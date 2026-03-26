# BEV Backbone 几何链路说明与 Dynamic 分支训练问题备忘

本文档整理自代码审查结论，供后续修改与验证时对照。**当前仓库代码是否已按文中建议修改，请以实际文件为准。**

---

## 一、`backbone/bev_backbone.py`：自车 3D 点 → 增强图像平面 → 填回 BEV

### 1. 设计意图（与你的描述对照）

整体流程与「在自车/雷达坐标系下初始化 3D 网格 → 投影到各相机 → 叠加上数据增强（`post_rot` / `post_trans`）后的像素坐标 → 用 `grid_sample` 取图像特征 → 铺到 BEV」一致。

### 2. `Splat.bev2eachroi` 步骤摘要

1. **3D 格点**：`create_voxels` 在 `xbound/ybound/zbound` 上生成 `(x, y, z)`。
2. **外参到相机系**：`p_cam = R @ p + t`，与数据里 `lidar2camera` 的 `rot/tran` 用法一致（若 LiDAR 与自车 BEV 不完全重合，需额外 ego→lidar 变换，此处未体现）。
3. **可见性**：`z <= 0` 的点被置为无效大坐标，避免错误投影。
4. **透视与鱼眼**：归一化平面 `(x/z, y/z, 1)` → `projectPoints_fisheye` + 内参 `K`。
5. **增强后图像平面**：`post_rot3 @ p + post_tran3`，与 `dataset.py` 中 `img_transform` 得到的增强矩阵一致。
6. **采样**：像素坐标按 `input_size`（如 `(H,W)=(128,384)`）归一化到 `[-1,1]`，供 `F.grid_sample` 使用。

### 3. 建议核对项（非必改，验证时留意）

| 项 | 说明 |
|----|------|
| `lidar` 与 `ego` | 标定若为 `lidar2camera`，而 BEV 定义在纯自车系，需确认二者是否对齐。 |
| `grid_sample` 与 `normalize_coords` | `normalize_coords` 使用 `(shape - 1)` 分母，更接近 `align_corners=True`；若 `grid_sample` 使用默认 `align_corners=False`，边界处可能有细微偏差。可选：统一为 `align_corners=True` 或改用与 `False` 一致的归一化公式。 |
| 鱼眼 `r=0` | `t/r` 在光轴附近可能数值不稳定，必要时加小 `epsilon`。 |
| `depths` 计算 | `bev2eachroi` 末尾对 `depths` 的 `view` 未参与返回，属死代码或预留。 |
| `get_cam_pos_embedding` | 依赖 `data_aug_conf`、`downsample` 等，在 `BEVBackbone` 上若未定义，调用会报错；当前若未使用可忽略。 |
| `bev_emb` 的 `ground_points = self.points[2]` | 为 Z 维第 3 个切片，语义上不一定是「地面高度」，若需严格接地点应对齐 `zbound`。 |

---

## 二、Dynamic 训练：box loss 下降，cls / cns / yns 不降、无检测结果

训练入口：`train.py` → `Model.forward` → `forward_dynamic_branch` → `det3d_head` 前向 + `det3d_head.loss`。问题主要集中在 **损失定义与解码**，而非 dataloader 或 `backward/step` 顺序本身。

### 1. `loss_cls` 的 `avg_factor` 与回归 mask 混用

**现象**：分类 focal 的监督对象是「匈牙利匹配后、类别目标非 ignore 的 query」，而归一化若误用 **`num_pos`（经 `cls_threshold_to_reg` 过滤后的回归正样本数）**，会导致 **cls 损失尺度与真实监督样本数不一致**，容易出现 box 在优化、cls/cns/yns 曲线异常或几乎不动。

**建议修改方向**：`cls_loss` 使用 **独立因子**，例如与 focal 一致的 valid 样本数（如 `(cls_target_flat >= 0)` 且不等于 `ignore_index` 的数量，并 `clamp(min=1)`），再作为 `avg_factor`。

**涉及文件**：`sparse4d/head.py`（`loss` 内主分支循环）。

### 2. `cls_threshold_to_reg` 与分类先验冲突

**背景**：`SparseBox3DRefinementModule` 中分类分支常按 **前景先验约 0.01** 初始化 bias，对应 **sigmoid 约 0.01**。

**问题**：若 **`cls_threshold_to_reg = 0.05`**，训练初期大量 **已匹配 query** 会因 `max(sigmoid(cls)) < 0.05` 被挡在 **回归分支**（含 box / cns / yns）之外，造成 **回归与分类优化目标错位**。

**建议**：默认关闭该过滤（例如 `-1` 且仅在 `> 0` 时启用），或改为 **极小阈值**（如 `0.001`），或 **warmup 后再打开**。

**涉及文件**：`sparse4d/build_model.py` 中构造 `Sparse4DHead` 时传入的 `cls_threshold_to_reg`。

### 3. 解码：`score = cls_sigmoid * centerness_sigmoid`

**背景**：`sparse4d/decoder.py` 在存在 `quality` 时，会将 top-k 分类分数与 **centerness** 相乘。

**问题**：训练初期 centerness 目标 `exp(-||Δxyz||)` 往往较小，分支容易学到 **很负的 logits**，`sigmoid(cns)` 接近 0，**最终分数被压到接近 0**，表现为可视化「没有检测结果」，即使分类在缓慢学习。

**建议**：对 **quality 最后一层 bias** 做合理初始化（例如 centerness / yawness 使用 `bias_init_with_prob(0.5)` 一类），避免一上来把 decode 分数乘没；必要时用 **仅 cls** 解码做对比实验。

**涉及文件**：`sparse4d/detection3d_blocks.py`（`SparseBox3DRefinementModule.init_weight`）、`sparse4d/decoder.py`。

### 4. 其它可继续排查的方向（若上述仍不足）

- **匈牙利代价权重**：`sparse4d/dn_sampler.py` 中 `cls_weight`、`box_weight` 比例是否合适。
- **DN**：`config` 中 `use_dn: False` 时无 `dn_*` 分支损失，仅主分支；若需稳定早期训练可对比打开 DN 的效果。
- **学习率与总 loss 各项权重**：观察 tensorboard 中各 `det3d_*` 项量级是否差多个数量级。

---

## 三、验证清单（可自行打勾）

- [ ] BEV 投影与数据集 `lidar2camera`、`post_rot/post_trans` 是否同一套约定（左手/右手、轴方向）。
- [ ] 修改 `avg_factor` 后，`loss_cls_*` 是否随训练平稳下降。
- [ ] 关闭或减小 `cls_threshold_to_reg` 后，`loss_box` / `loss_cns` / `loss_yns` 是否与 cls 更一致。
- [ ] 调整 quality 初始化或解码策略后，tensorboard 可视化是否出现合理框（注意 `score_thresh`）。

---

*文档仅作备忘，具体以你本地修改与实验结果为准。*
