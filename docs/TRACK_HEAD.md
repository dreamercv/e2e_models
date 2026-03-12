## TrackHead 模块说明（`sparse4d/track_head.py`）

### 1. 总体功能

- **作用**：在 Sparse4D 中负责**基于 BEV 检测结果做跨帧目标跟踪**。
- **输入**：
  - `seq_features`: `(B, T, N, C)`  
    - B: batch size  
    - T: 时间帧数（序列长度）  
    - N: 每帧 anchor / instance 数量（检测 head 输出的 query 数）  
    - C: instance feature 维度  
  - `seq_anchors`: `(B, T, N, 11)`  
    - 每个 anchor 对应一个 11 维 3D 框编码 `[x, y, z, log(w), log(l), log(h), sin(yaw), cos(yaw), vx, vy, vz]`。
  - `T_ego_his2cur`: `(B, T, 4, 4)`  
    - 第 `t` 帧的自车坐标系相对前一帧的变换矩阵（从 t-1 到 t）。
- **输出**：
  - `affinity`: `(B, T, N, N)`  
    - `affinity[b, t, i, j]` 表示：**当前帧 t 的第 i 个 instance 与上一帧 t-1 的第 j 个 instance 的匹配概率/亲和度**。
  - 可配合 `track_affinity_loss` 训练，配合 `decode_track` 解码出离散的 track id。

整体逻辑是：  
1. 用自车位姿 `T_ego_his2cur` 将历史帧的 anchor 对齐到当前帧坐标系。  
2. 将 instance feature 和 anchor 编码分别通过 MLP 投到同一 embedding 空间相加。  
3. 使用多头 cross-attention，让“当前帧 token”去 attend “上一帧 token”，得到一帧内的 `(N, N)` 亲和矩阵。  
4. 堆叠所有时间帧得到 `(B, T, N, N)`，再用 **匈牙利匹配** 或简单 argmax 将亲和矩阵转为离散 match / track id。

---

### 2. 坐标对齐：`align_anchors_to_frame`

```python
def align_anchors_to_frame(anchor: torch.Tensor, T_src2dst: torch.Tensor) -> torch.Tensor:
    """
    将 anchor 从源帧变换到目标帧。
    anchor: (B, N, 11) 编码格式 [x,y,z,log(w),log(l),log(h),sin(yaw),cos(yaw),vx,vy,vz]
    T_src2dst: (B, 4, 4) 源帧到目标帧的变换矩阵
    returns: (B, N, 11) 目标帧下的 anchor（编码格式不变）
    """
```

- 从 11 维编码中取出：
  - `center = [x, y, z]`
  - `size = [log(w), log(l), log(h)]`（保持不变，只是透传）
  - `sin_cos = [sin(yaw), cos(yaw)]`
  - `vel = [vx, vy, vz]`
- 使用 `T_src2dst` 的旋转和平移部分对 `center` 和 `vel` 做 3D 仿射变换：
  - `center_dst = R @ center_src + t`
  - `vel_dst = R @ vel_src`
- 对 yaw 的朝向向量 `[sin(yaw), cos(yaw)]` 使用平面旋转：
  - 取 `R[:2, :2]` 作为平面旋转，对 `[sin, cos]` 做线性变换得到新朝向。
- 最后将对齐后的 `[center, size, yaw_vec, vel]` 再拼回 11 维编码。

这个函数的核心是：**避免直接在角度空间做插值或差分，而是对 sin/cos 和速度矢量做线性变换**，数值上更稳定。

---

### 3. TrackHead 前向流程

```python
class TrackHead(nn.Module):
    def __init__(self, feat_dim=256, anchor_dim=11, num_heads=8, dropout=0.1, embed_dim=256):
        self.feat_proj = nn.Linear(feat_dim, embed_dim)
        self.anchor_proj = nn.Linear(anchor_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
```

- `feat_proj`: 将 detection head 提供的 instance feature 投到 `embed_dim` 维。
- `anchor_proj`: 将 anchor 的 11 维几何编码投到 `embed_dim` 维。
- `cross_attn`: 核心的跨帧注意力层，用于计算当前帧与上一帧之间的匹配。
- `norm`: 对 token 做 LayerNorm，稳定训练。

#### 3.1 `forward` 细节

```python
def forward(self, seq_features, seq_anchors, T_ego_his2cur):
    B, T, N, C = seq_features.shape
    affinity_list = []
    for t in range(T):
        if t == 0:
            # 第一帧没历史信息，默认单位矩阵（每个实例只和自己匹配）
            aff = torch.eye(N, device=device, dtype=seq_features.dtype).unsqueeze(0).expand(B, N, N)
            affinity_list.append(aff)
            continue

        feat_cur = seq_features[:, t]        # (B, N, C)
        anchor_cur = seq_anchors[:, t]       # (B, N, 11)
        feat_prev = seq_features[:, t - 1]   # (B, N, C)
        anchor_prev = seq_anchors[:, t - 1]  # (B, N, 11)
        T_prev2cur = T_ego_his2cur[:, t]     # (B, 4, 4)

        # 1) 将上一帧 anchor 对齐到当前帧坐标
        anchor_prev_aligned = align_anchors_to_frame(anchor_prev, T_prev2cur)

        # 2) 将 feature + anchor 编码成 token
        token_cur = self.norm(self.feat_proj(feat_cur) + self.anchor_proj(anchor_cur))               # (B, N, D)
        token_prev = self.norm(self.feat_proj(feat_prev) + self.anchor_proj(anchor_prev_aligned))    # (B, N, D)

        # 3) 多头 cross-attention：当前帧 query，上一帧 key/value
        attn_out, attn_weights = self.cross_attn(
            token_cur, token_prev, token_prev, need_weights=True
        )
        # attn_weights: (B, N_query, N_key) = (B, N, N)
        affinity_list.append(attn_weights)

    # 堆成 (B, T, N, N)
    return torch.stack(affinity_list, dim=1)
```

- 时间维上逐帧循环：
  - **t=0**：没有历史帧，亲和矩阵设为单位阵（每个实例只和自己对齐），方便后续逻辑统一处理。
  - **t>0**：对第 t 帧：
    - 把 `t-1` 帧 anchor 通过 `T_prev2cur` 对齐到当前帧坐标。
    - 将“当前帧 feature+anchor”作为 query，将“对齐后的上一帧 feature+anchor”作为 key/value。
    - `MultiheadAttention` 输出的 `attn_weights[b]` 形状为 `(N, N)`，每一行是一个当前实例对上一帧所有实例的 softmax 概率分布，刚好就是我们需要的亲和矩阵。

---

### 4. 跟踪 loss：`track_affinity_loss`

```python
def track_affinity_loss(affinity, gt_match, ignore_index=-1):
    """
    affinity: (B, T, N, N)
    gt_match: (B, T, N)，gt_match[b,t,i] = 上一帧中匹配的 index in [0, N)，或 ignore_index 表示新轨迹/忽略
    """
    B, T, N, _ = affinity.shape
    aff_flat = (affinity.reshape(-1, N) + 1e-8).log()
    gt_flat = gt_match.reshape(-1).long()
    valid = gt_flat != ignore_index
    if valid.sum() == 0:
        return affinity.new_tensor(0.0)
    return F.nll_loss(aff_flat[valid], gt_flat[valid], reduction="mean")
```

- 将 `(B, T, N, N)` 展平成 `(B*T*N, N)`，每一行是一帧内一个实例对上一帧 N 个实例的 log 概率。
- `gt_match[b,t,i]` 表示“当前帧 t 的第 i 个 instance 在上一帧 t-1 中对应的 index”（若为新轨迹或不参与训练，则为 `ignore_index`）。
- 对所有非 ignore 的位置，直接做 **NLLLoss**：
  - 类似于分类任务：  
    - `aff_flat[valid]`：预测概率分布  
    - `gt_flat[valid]`：标签（上一帧 index）
- 这样可以直接训练 cross-attention 输出的 attention 权重，使其在真值对应的上一帧 index 上概率更大。

---

### 5. 轨迹解码：`decode_track`

```python
def decode_track(affinity, seq_anchors, use_hungarian=True, score_threshold=None):
    """
    affinity: (B, T, N, N)
    seq_anchors: (B, T, N, 11)
    返回:
        track_ids: (B, T, N)  每个 instance 的 track id
        positions: (B, T, N, 3)  直接取 seq_anchors[..., :3]
    """
```

整体思路：**把 attention 亲和矩阵转成离散的轨迹 id**。

#### 5.1 初始化

- `track_ids[b, 0] = 0..N-1`：第一帧每个实例各自一个 ID。
- `next_id = N`：后续新出现的实例从 N 往上编号。

#### 5.2 t>0 帧的匹配

```python
for t in range(1, T):
    cost = -affinity[b, t].float().cpu().numpy()   # (N, N)
    if use_hungarian:
        row_idx, col_idx = linear_sum_assignment(cost)
        if score_threshold is not None:
            aff = affinity[b, t].cpu().numpy()
            for i, j in zip(row_idx, col_idx):
                if aff[i, j] < score_threshold:
                    # 亲和度太低，认为是新目标
                    track_ids[b, t, i] = next_id
                    next_id += 1
                else:
                    # 继承上一帧 j 的 id
                    track_ids[b, t, i] = track_ids[b, t - 1, j].item()
        else:
            for i, j in zip(row_idx, col_idx):
                track_ids[b, t, i] = track_ids[b, t - 1, j]
    else:
        best_j = affinity[b, t].argmax(dim=-1)  # 每行取 argmax
        ...
```

- **为什么需要匈牙利匹配（Hungarian）？**
  - 注意 `affinity[b, t]` 是一个 **完整的 (N, N) 矩阵**，每一行是“当前实例 i 对上一帧所有实例 j 的概率分布”。
  - 如果简单对每一行取 argmax：
    - 可能出现 **多个当前实例都匹配到同一个上一帧实例** 的情况（多对一）。
    - 这会导致轨迹 ID 混乱，无法保证一一对应的 assignment。
  - 匈牙利匹配（线性分配）在 cost 矩阵上求解的是 **全局最优的一一匹配**（或尽量接近），保证：
    - 每个当前实例最多对应一个上一帧实例；
    - 整体的匹配代价（负亲和度）最小。
  - 因此这里构造 `cost = -affinity`，用 `linear_sum_assignment` 在 CPU 上求解最优匹配，再根据 match 结果为当前帧实例分配上一帧轨迹或新建轨迹。

- **`score_threshold` 的作用**：
  - 即便匈牙利给出了匹配 `(i,j)`，但如果 `affinity[b,t,i,j]` 太低，说明“这个 i 很可能是新目标”，就会给 `i` 分配一个新的 track id，而不是继承 `j` 的 id。

---

### 6. 总结：TrackHead 的作用与流程

1. **输入**：检测 head 的时序 instance feature + 对应 anchor + 自车位姿变换矩阵。
2. **坐标对齐**：用 `T_ego_his2cur` 将上一帧 anchor 对齐到当前帧坐标，使匹配只关注语义变化而不是坐标系变换。
3. **token 构造**：feature 与几何编码分别投到 embedding 空间相加，得到“时空 aware 的 instance token”。
4. **跨帧注意力**：当前帧 token 作为 query，对齐后的上一帧 token 作为 key/value，通过 `MultiheadAttention` 得到 `(N, N)` 亲和矩阵。
5. **训练**：用 `track_affinity_loss` 与 `gt_match`（上一帧 index 标签）做 NLLLoss，直接监督 attention 权重。
6. **推理**：用 `decode_track` 对亲和矩阵做匈牙利匹配（可选阈值），得到轨迹 ID 序列与对应的 3D 位置。

这样设计的好处是：
- **统一在 BEV/instance 空间内做跟踪**，不直接依赖像素或 raw 点云；
- 利用了 detection head 的高维语义特征 + anchor 几何信息；
- 匈牙利匹配保证了全局一一 assignment，而不是贪心式的局部 argmax。

