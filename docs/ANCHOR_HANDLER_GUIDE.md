## Anchor Handler / Anchor 先验：作用、原因与代码落点

本页总结我们今天讨论的核心点：
1) `anchor_handler` 是什么  
2) 为什么需要 anchor 先验（prior/query）  
3) 在代码中它如何“微调/对齐” anchor（尤其是 temporal cache + DN）  
4) refinement 在代码里是如何把预测落实为“anchor 增量 + 回加”

---

## 1. `anchor_handler` 是什么？

在 Sparse4D 中，`anchor_handler` 是一个**可插拔的模块**（官方通过配置传入），它至少要提供一个接口：
`anchor_handler.anchor_projection(anchor, T_src2dst_list, time_intervals=...)`

它的职责不是“算 loss”，而是做**几何/时间上的 anchor 对齐（projection / alignment）**，把：

- temporal 模块缓存下来的历史 anchor（`cached_anchor`）
- DN 去噪产生的 noisy anchor（`dn_anchor`）

按当前帧相对运动（由相机/ego 位姿变换矩阵给出）与时间间隔（`Δt`）投影到当前帧坐标系下，保证后续 temporal attention / matching / loss 的几何一致性。

---

## 3.1 `gnn`（图注意力）在 Sparse4D 中的作用：为什么要加

在 Sparse4D 的 decoder 循环里，`operation_order` 通常包含 `gnn`，即在进入 `deformable` 融合 BEV/图像信息之前，对**所有 instance/query tokens 做一次 self-attention**。

在官方 `sparse4d_head.py` 中，当 `op == "gnn"` 时会调用 attention：

- `query` / `key` / `value` 主要来自当前一轮的 `instance_feature`
- `query_pos = anchor_embed` 将 anchor 的几何先验编码（通过 `anchor_encoder(anchor)`）注入到 attention 中
- `attn_mask` 用于 DN token 的隔离/约束：保证 noisy anchors 不会完全污染 learnable instances 的注意力流

因此 `gnn` 的直观作用可以总结为：

1. **让不同 anchor 之间先相互“交流”**，获得集合内一致性（例如减少重复检测、增强区分度）。
2. **把空间先验作为位置编码引入 self-attention**，让注意力不仅依据特征相似度，也依赖 anchor 几何关系。
3. **为后续 `deformable` 融合做更好的特征预处理**：deformable 通常会在 BEV 上采样/融合局部信息，gnn 先做全局 token 交互能提升可判别性。

简言之：`gnn` 是 Sparse4D 的“query 集合内部更新器”，`deformable` 是“query 与外部 BEV/图像的交互器”；二者配合能提升检测质量与训练稳定性。

---

## 2. 为什么 anchor 需要先验（prior）？

### 2.1 anchor 的本质：learnable query 原型

官方会从 `npy` 读取一组离散的 3D 框原型（k-means 聚类得到的 nuScenes anchor 原型），anchor 在编码维度上是 11 维：

`[X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ]`

在 `InstanceBank` 里，这组先验会被注册为可学习参数（`self.anchor = nn.Parameter(...)`），同时每个 anchor 对应一份可学习的实例特征（`instance_feature`）。

### 2.2 为什么不用“从零预测 anchor”？

Sparse4D 的 decoder/ refinement 并不是直接回归绝对框，而是围绕 anchor query 做 refine：

- anchor 给 decoder 提供“当前 query 应该长什么样”（尺度、朝向、速度量级）
- decoder 只需要学习“在该先验附近如何调整”

这会显著降低学习难度，让训练更稳定、更符合 DETR/Sparse query 的建模方式。

---

## 3. `anchor_handler` 在代码中的作用链路（temporal cache & DN）

### 3.1 temporal cache：`InstanceBank.get()`

官方 `InstanceBank.get()` 中的关键逻辑（对应我们 debug 时关心的投影）：

1. 计算当前帧与缓存帧的时间差 `time_interval = metas["timestamp"] - history_time`
2. 用 `self.mask` 控制时间间隔是否可信（`abs(time_interval) <= max_time_interval`）
3. 若存在 `anchor_handler`，构造历史帧到当前帧的变换矩阵 `T_temp2cur`
4. 调用：

`cached_anchor = anchor_handler.anchor_projection(cached_anchor, [T_temp2cur], time_intervals=[-time_interval])[0]`

这一步就是 `anchor_handler` 的核心价值：**把历史 anchor 投影回当前帧坐标系，并做速度/时间补偿**。

我们的实现里也有同样的结构（在 `projects/e2e_models/sparse4d/instance_bank.py` 里）：
- `if self.cached_anchor is not None and batch_size == self.cached_anchor.shape[0]:`
- `self.cached_anchor = self.anchor_handler.anchor_projection(self.cached_anchor, [T_temp2cur], time_intervals=[-time_interval])[0]`

### 3.2 DN noisy anchors：同样需要投影

DN 训练会先构造 `dn_metas["dn_anchor"]`（噪声 anchor）以及对应的 `dn_reg_target / dn_cls_target`。

由于 DN anchors 也要参与 temporal 部分的几何对齐，因此也会走同一个 projection：

`dn_anchor = anchor_handler.anchor_projection(dn_anchor.flatten(1,2), [T_temp2cur], time_intervals=[-time_interval])[0]`

并再 reshape 回 `[batch_size, num_dn_group, num_dn, state_dim]`。

---

## 4. `anchor_projection` 如何进行几何与时间补偿？

在官方 `detection3d_blocks.py` 中，`SparseBox3DKeyPointsGenerator.anchor_projection` 的核心思想是：

1. **用速度对中心点做时间平移补偿**
2. **用 `T_src2dst` 变换矩阵完成刚体旋转 + 平移**
3. 对 yaw 用 sin/cos 的旋转方式更新
4. 对速度向量也做旋转

关键代码语义（对应我们理解投影的步骤）：

- `center = anchor[..., [X, Y, Z]]`
- 若有 `time_interval`：
  - `translation = vel * time_interval`
  - `center = center - translation`
- 旋转/平移：
  - `center = R * center + t`
- yaw 更新：
  - 用 `T_src2dst[:2,:2]` 作用到 `[COS_YAW, SIN_YAW]` 向量
- vel 更新：
  - 用旋转部分把 vel 向量旋转过去

这保证了“anchor 在时间推进 Δt 以后”的位置、朝向、速度是自洽的。

---

## 5. anchor 是如何被“微调（refine）”出来的？（预测不是直接替换先验）

微调发生在 `SparseBox3DRefinementModule.forward()`。

参考官方代码逻辑（我们也在 `sparse4d/head.py` 的 `op == "refine"` 分支调用它）：

1. 特征融合：
   - `feature = instance_feature + anchor_embed`
2. refinement 网络预测一个 `output`：
   - `output = self.layers(feature)`，输出维度包含 refine_state 指定的参数
3. **关键：只对 refine_state 这些维度做“增量回加”**：
   - `output[..., refine_state] = output[..., refine_state] + anchor[..., refine_state]`

因此 refinement 并不是“输出最终 anchor”，而是：
**预测增量（delta），再对 anchor 中相应维度回加，得到 refined anchor。**

在我们当前 `sparse4d/head.py` 的调用形式里：
```python
anchor, cls, qt = self.layers[i](
    instance_feature, anchor, anchor_embed,
    time_interval=time_int,
    return_cls=...
)
```
其中 `self.layers[i]` 就是 refinement 模块（或其堆叠）。

---

## 6. 一句话把整个逻辑串起来

1. `anchor` 先验（来自 k-means npy）提供 query 原型：尺度/朝向/速度的“合理起点”
2. temporal 模块用 `anchor_handler.anchor_projection` 把历史/noisy anchor 投影到当前帧坐标系（含速度与 Δt 补偿）
3. decoder 的 refinement 学的是：**在该先验附近预测增量 delta**
4. refinement 在代码中用 `delta + anchor` 得到 refined anchor/box
5. 最终 decode 输出预测结果用于 loss / 可视化 / 推理

---

## 7. 你排查 DN/Loss 问题时可重点核对的点（实用检查清单）

- 你的 `anchor_handler` projection 是否同时作用于 `cached_anchor` 与 `dn_anchor`
- `time_interval` 的符号与数值是否与投影实现一致（例如 `time_intervals=[-time_interval]`）
- `dn_valid_mask / dn_cls_target / dn_reg_target` 的结构是否与 matcher/采样器的预期维度完全一致
- refinement 的“delta + anchor”是否对齐到你当前 `refine_state` 的维度索引

---

## 8. `BEVFeatureAggregation`：它如何把 BEV 特征聚合到 instance 上？（以及你现在为何是中心点采样）

`BEVFeatureAggregation` 位于 `sparse4d/bev_aggregation.py`，它的核心职责是：

> 对每个 instance/anchor，把 BEV 特征图在 anchor 对应的位置（或关键点集合）用 `grid_sample` 采样出来，再聚合成 instance 特征，作为后续 `refine` 的输入。

### 8.1 在 `sparse4d/build_model.py` 中如何实例化？

你在 `projects/e2e_models/sparse4d/build_model.py` 里实例化 `BEVFeatureAggregation` 时，关键配置是：

1) `kps_generator`：

```python
kps = SparseBox3DKeyPointsGenerator(
    embed_dims=embed_dims,
    num_learnable_pts=0,
    fix_scale=((0.0, 0.0, 0.0),)
)
```

2) `BEVFeatureAggregation`：

```python
bev_agg = BEVFeatureAggregation(
    embed_dims=embed_dims,
    bev_bounds=bev_bounds,
    kps_generator=kps,
    proj_drop=dropout,
    residual_mode="add",
)
```

由于 `num_learnable_pts=0` 且 `fix_scale` 里只有一个点 `(0,0,0)`，因此 `SparseBox3DKeyPointsGenerator.num_pts == 1`。

### 8.2 你现在到底是“中心点采样”还是“关键点采样”？

`SparseBox3DKeyPointsGenerator` 在生成 key points 时逻辑是：

- 先根据 anchor 尺寸得到 `size`
- 再用 `fix_scale` 乘上 `size` 得到局部偏移 key_points

当 `fix_scale=((0,0,0),)` 时，这个局部偏移恒为 0，最终 key point 退化为：

> key_points 的位置就是 anchor 的中心（`[X,Y,Z]`）

`BEVFeatureAggregation.forward()` 只使用 `key_points[..., :2]` 的 x,y 来做 BEV 网格采样，因此：

> 虽然代码结构支持“关键点采样”，但因为 `num_pts=1 且该点=中心点”，所以你的当前设置本质上是“中心点采样”。

### 8.3 `BEVFeatureAggregation` 的计算流程（对应代码）

在 `bev_aggregation.py` 里大致流程是：

1) 生成 key points（中心点或关键点）
2) 把 key points 的 x,y 映射到 BEV 网格的归一化坐标 `[-1, 1]`
3) 用 `torch.nn.functional.grid_sample` 在 BEV 特征图上采样
4) 将采样结果 reshape 成 `(B, N, num_pts, C)`，对 `num_pts` 聚合（当前 num_pts=1 时等价于不聚合）
5) `output_proj` 线性投影回 embed_dims，然后用残差把它加回 `instance_feature`

其中采样用的是：

```python
sampled = torch.nn.functional.grid_sample(
    bev_map, grid, mode="bilinear",
    padding_mode="zeros", align_corners=True
)
```

聚合用的是：

```python
features = sampled.sum(dim=2)
```

残差连接用的是 `residual_mode="add"`：

```python
output = output + instance_feature
```

### 8.4 为什么采样粒度会影响 loss？

`BEVFeatureAggregation` 输出的 instance 特征会直接影响后续 decoder 的分类/回归（refine、dn loss 等）。

当你使用“中心点采样（num_pts=1）”时：

- instance 的先验 anchor（你目前 anchor_init 默认全 0，且 temporal cache 也较简化）稍有偏差时，
- BEV 上采到的局部特征就可能落在“目标框覆盖区域之外”，导致 instance_feature 对真实目标的条件信息变弱，
- 后续 refine 学习到的 delta 会更难、梯度噪声更大，
- 从而表现为 loss 更敏感、更容易变大或波动（尤其 DN 与正样本对齐不稳时）。

相反，如果使用真正的多关键点采样（`num_pts>1`，关键点覆盖目标框的不同位置），聚合后更稳健，instance_feature 更能反映“框覆盖区域”的语义和几何信息，loss 往往会更稳定。

