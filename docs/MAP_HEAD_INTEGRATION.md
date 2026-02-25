# 在 BEV Backbone 后接入 MapTR 风格 map_head 的方案

当前流程：**时序图像 → BEV Backbone（含时序对齐）→ BEV 特征 → Head（检测）+ TrackHead（跟踪）**。  
目标：在 **Backbone 输出 BEV 特征之后**，再接一个 **MapTR 策略的 map_head**，实现检测 + 跟踪 + 地图元素三路输出。

---

## 一、MapTR 与当前结构的差异

| 项目 | MapTR 原版 | 你当前结构 |
|------|------------|------------|
| BEV 从哪来 | 在 **pts_bbox_head 内部**用 BEVFormer 等 encoder，从 **多视角图像特征** 生成 BEV | **已有独立 BEV Backbone**，直接输出 BEV 特征 (B*T, C, H, W) |
| Head 输入 | **mlvl_feats**：多尺度图像特征 (B, N_cam, C, H, W) | **feature_maps**：BEV 特征 (B*T, C, H_bev, W_bev) |

因此：**不要**把 MapTR 整头（含内部 BEV 编码）搬过来；只采用其 **“地图解码”策略**，即：**在已有 BEV 特征上做 map 的 query-based 解码 + 匹配 + loss**。

---

## 二、整体架构（接入 map_head 后）

```
时序图像 (B, T, N_cam, 3, H, W)
    ↓
BEV Backbone（视图变换 + 时序对齐）
    ↓
BEV 特征 feature_maps: (B*T, C, H_bev, W_bev)
    ├──→ Sparse4D Head  → 检测 (classification / prediction / quality)
    │         ↓
    │    seq_features, seq_anchors
    │         ↓
    │    TrackHead  → 跟踪 (track_affinity / track_ids / track_positions)
    │
    └──→ Map Head（MapTR 策略）→ 地图元素 (cls_scores, pts_preds)
```

- **Backbone 只跑一次**，同一份 `feature_maps` 同时喂给 **检测头** 和 **map_head**。  
- 检测 / 跟踪 的接口和现有代码一致；**新增**的是 map_head 的输入输出与 loss。

---

## 三、Map Head 设计要点（MapTR 策略）

### 3.1 输入

- **BEV 特征**：`feature_maps`，形状 `(B*T, C, H_bev, W_bev)`（与检测头一致）。
- **使用方式二选一**（推荐先做 **方案 A**）：
  - **方案 A（按“当前帧”建图）**：把 `feature_maps` 视为 `(B, T, C, H, W)`，只取 **每个样本的最后一帧** `feat = feature_maps.view(B, T, C, H, W)[:, -1]`，得到 `(B, C, H, W)` 再进 map_head。这样 map 只对“当前时刻”建图，实现简单，和多数 MapTR 设定一致。
  - **方案 B（每帧都出图）**：整块 `(B*T, C, H, W)` 进 map_head，输出 `(B*T, num_vec, ...)`，即每帧一组 map 预测；需要配套的 GT 也是每帧一份（list 长度 B*T）。

### 3.2 MapTR 策略核心（在 BEV 上复现）

1. **Map 实例表示**  
   - 每条地图线/元素用 **一条折线** 表示：`num_pts_per_vec` 个 2D 点（在 BEV 的 pc_range 内），例如 divider、crossing、boundary 等。
   - 类别数：`num_map_classes`（与检测的 num_classes 分开）。

2. **Query 设计**  
   - **num_vec** 个 map instance query（如 100/200），与 MapTR 一致。
   - 两种方式二选一：
     - **all_pts**：`num_query = num_vec * num_pts_per_vec`，每个 query 对应一个点，再按 instance 分组得到多条线。
     - **instance_pts**：`num_vec` 个 instance embedding + `num_pts_per_vec` 个 point embedding，相加得到 `num_vec * num_pts_per_vec` 个 query（与 MapTR 的 `query_embed_type='instance_pts'` 一致）。

3. **Decoder**  
   - **Transformer Decoder**：map queries 作为 query，**BEV 特征** 作为 memory。
     - 将 BEV 展平为 `(B, C, H*W)` 或保留 2D 做 2D 位置编码，作为 decoder 的 memory。
     - 可选：用 deformable attention 让 query 只 attend 到 BEV 的局部（与 MapTR 的 decoder 类似），减少计算。
   - 输出：每个 query 的 hidden state → 接 **cls branch**（每条 instance 一个类别）和 **pts branch**（每个点 2 维，再 reshape 成 (num_vec, num_pts_per_vec, 2)）。

4. **匹配与 Loss（与 MapTR 一致）**  
   - **Assigner**：匈牙利匹配，cost = cls_cost + bbox_cost（由 pts 的 minmax 得到 2D bbox）+ **Chamfer cost**（预测点集与 GT 点集的距离）。
   - **Loss**：
     - 分类：Focal 或 CE，对匹配上的 query 算。
     - 2D bbox：L1（可选，由 pts 的 minmax 得到）。
     - **ChamferDistance**：匹配上的预测折线点与 GT 折线点。
     - **Direction loss**：相邻预测点的方向与 GT 方向一致（如 cosine loss），加强几何约束。

5. **坐标与归一化**  
   - 与 MapTR 一致：在 `pc_range` 内做归一化再进 loss；预测时反归一化回世界/车体坐标。你的 BEV 已有 `bev_bounds`/grid，可复用或与 MapTR 的 `pc_range` 对齐。

### 3.3 输出形式（便于和现有 model 对接）

- **训练**：  
  - `map_cls_scores`: `(B, num_vec, num_map_classes)`  
  - `map_pts_preds`: `(B, num_vec, num_pts_per_vec, 2)`（归一化或 pc_range 内坐标，与 loss 约定一致）  
  - 可选：`map_bbox_preds`: `(B, num_vec, 4)`（由 pts minmax 得到，用于匹配/辅助 loss）

- **推理**：  
  - 对 `map_cls_scores` 做阈值过滤，对 `map_pts_preds` 反归一化，得到当前帧的 map 折线 list（每条线 = 一个 instance 的 (num_pts_per_vec, 2) + 类别）。

---

## 四、与现有 model 的代码对接方式

### 4.1 扩展 `build_sparse4d_bev_model`（或新函数）

- 增加可选参数 **`map_head=None`**。
- 在 `Sparse4DBEVModel.forward` 中：
  1. `feature_maps = self.bev_backbone(...)`（不变）
  2. `head_out, seq_features, seq_anchors = self.head(feature_maps, metas)`（不变）
  3. 若存在 `self.track_head`，用 `seq_features, seq_anchors, T_ego_his2curs` 算跟踪（不变）
  4. **若存在 `self.map_head`**：
     - 按方案 A：`bev_for_map = feature_maps.view(B, T, C, H, W)[:, -1]`（若当前 backbone 输出已是 B*T 合并，需已知 T 做 view）。
     - 调用 `map_out = self.map_head(bev_for_map, metas)`（或传入 `feature_maps` 和 `metas`，在 map_head 内部做取最后一帧）。
  5. 返回值在现有 4 元组基础上增加 **map_out**（可为 None）：  
     `return head_out, seq_features, seq_anchors, track_out, map_out`

### 4.2 Map Head 的 forward 约定

- **输入**：  
  - `bev_feature`: `(B, C, H, W)`（方案 A）或 `(B*T, C, H, W)`（方案 B）  
  - `metas`: 训练时需包含 map 的 GT（见下）。
- **输出**：  
  - 训练：`dict`，例如 `map_cls_scores`, `map_pts_preds`, `map_bbox_preds`（可选），以及 **`loss_map`**（在 head 内算好，或返回各分项由 model 汇总）。
  - 推理：同一 dict，但无 `loss_map`，只有预测；也可在 model 里根据 `map_out` 再 decode 成 list of polylines。

### 4.3 真值（metas）里为 map 准备的内容

- **gt_map_pts**：list，长度 B（方案 A）或 B*T（方案 B）。每个元素形状 `(N_gt_vec, num_pts_per_gt_vec, 2)`，BEV/pc_range 下的 2D 点坐标。
- **gt_map_labels**：list，长度同上。每个元素 `(N_gt_vec,)`，地图元素类别下标（divider、crossing、boundary 等）。
- 若使用 bbox 辅助匹配，可再提供 **gt_map_bboxes**（由 gt_map_pts 的 minmax 得到），或 head 内自动从 pts 算。

与现有检测/跟踪的 `gt_labels_3d`、`gt_bboxes_3d`、`gt_track_match` 并列放在 `metas` 中即可。

---

## 五、实现步骤建议

1. **新建 `map_head.py`（纯 PyTorch）**  
   - 类名如 `MapHead` 或 `MapTRStyleMapHead`。  
   - 实现：  
     - 用 BEV 特征做 memory 的 Transformer Decoder（可先不用 deformable，用普通 MHA）；  
     - MapTR 风格的 query（instance_pts 或 all_pts）；  
     - cls/reg(pts) 分支；  
     - 前向输出 `map_cls_scores`、`map_pts_preds`（及可选 bbox）；  
     - 内部或单独模块：匈牙利 assigner（cls + bbox + Chamfer cost）、Chamfer loss、direction loss。

2. **Assigner / Loss**  
   - 参考 MapTR 的 `maptr_assigner.py` 和 `maptr_head.py` 的 `loss_single`：  
     - 实现一个不依赖 mmdet 的 **MapTRAssigner**（scipy `linear_sum_assignment` + 自定义 cost）。  
     - 实现 ChamferDistance、PtsDirCosLoss（或 L1 on direction vector），与 MapTR 一致。

3. **在 model.py 中挂接**  
   - `build_map_head(...)` 返回 `MapHead` 实例。  
   - `build_sparse4d_bev_model(..., map_head=None)`，在 forward 中在 backbone 之后调用 map_head，并返回 map_out。

4. **DataLoader / metas**  
   - 在现有 dataloader 中增加对 map GT 的读取与整理（每帧或仅当前帧的 `gt_map_pts`、`gt_map_labels`），放入 `metas`，与 [DATA_FORMAT.md](DATA_FORMAT.md) 中现有约定一致。

5. **训练循环**  
   - 若 `map_out` 非空且含 `loss_map`，则 `total_loss += map_out["loss_map"]`（或分项加权重）。

---

## 六、小结

- **不**在 map_head 里再做一遍“图像→BEV”，而是 **直接吃 backbone 的 BEV 输出**，与检测/跟踪共享同一 BEV 特征。
- **只借鉴 MapTR 的“地图解码”**：固定数量 map instance queries、折线点表示、Transformer decoder 以 BEV 为 memory、匈牙利匹配 + Chamfer + direction loss。
- 接入点唯一：**Backbone 输出 → 复制一份给 map_head**；检测与跟踪逻辑不变，仅模型返回值多一个 `map_out`，便于你逐步实现和调试。

如果你愿意，我可以下一步在仓库里加一个 **最小可跑的 `map_head` 骨架**（含 decoder + 简单 assigner + Chamfer/dir loss 占位），再和 `model.py` 的接口接好，方便你直接填数据和调参。
