# BEV Backbone 几何与 Dynamic 检测 Loss 问题说明

本文档汇总对工程中 `backbone/bev_backbone.py` 的几何链路检查结论，以及 **dynamic 分支训练时 box loss 下降但 cls / cns / yns loss 不下降、解码无框** 的可能原因与建议修改方向，便于后续自行改代码验证。

---

## 一、`bev_backbone.py`：自车 3D 点 → 增强图像 → 特征落到 BEV

### 1. 设计意图（与你的描述对应）

1. 在自车/雷达坐标系（与 `grid_conf` 的 `xbound/ybound/zbound` 一致）上建立 3D 体素格点 `create_voxels`。
2. 用外参 `rots/trans`（数据中为 `lidar2camera`）将点变到相机系：`p_cam = R @ p + t`。
3. 透视除法 → 鱼眼畸变 + 内参 `intrins/distorts` → 得到「未增强」像素平面上的齐次坐标。
4. 再施加数据增强仿射 `post_rot3/post_tran3`（与 `dataset.py` 中 `img_transform` 一致），得到 **增强后图像** 上的坐标。
5. 按特征图尺寸 `input_size`（如 `(128,384)`）归一化到 `[-1,1]`，用 `grid_sample` 从图像特征上采样，多高度层拼通道后再与 BEV 融合。

整体上，**「自车 3D 点 → 增强后图像坐标系 → 取特征写回 BEV」** 这条链路与 LSS/BEVDet 类做法一致。

### 2. 实现上需自行核对的点

| 项目 | 说明 |
|------|------|
| 坐标系命名 | 数据里写的是 `lidar2camera`；若 BEV 真在 **ego** 而 ego 与 lidar 未对齐，需要补 **ego→lidar**（或统一标定定义）。 |
| `grid_sample` 与 `normalize_coords` | `normalize_coords` 使用 `(shape - 1)` 作分母，等价于 **align_corners=True** 的像素映射；若 `grid_sample` 使用默认 `align_corners=False`，边界与亚像素上会有轻微不一致，建议两者统一。 |
| 鱼眼 `projectPoints_fisheye` | `r=0`（光轴附近）可能出现除零/数值问题，必要时加小 `epsilon`。 |
| 死代码 | `bev2eachroi` 末尾对 `depths` 的 `view` 未参与返回，可删或留给深度相关扩展。 |
| `get_cam_pos_embedding` | 依赖未在 `BEVBackbone` 上定义的 `data_aug_conf`、`downsample`；若未调用可忽略，若调用会报错。 |
| `bev_emb` | `ground_points = self.points[2]` 是 **Z 维第 3 个切片**，语义上不一定是「地面高度」，若要对齐路面应对齐 `zbound`。 |

---

## 二、Dynamic 训练：box 降、cls/cns/yns 不降、无检测结果

### 1. 调用链（入口 `train.py`）

- `train.py` → `Model.forward` → `forward_dynamic_branch` → `det3d_head(...)` + `det3d_head.loss(out, metas)`。
- 损失在 `sparse4d/head.py` 的 `Sparse4DHead.loss` 中组装；分类为 `FocalLoss`，回归与 quality 为 `SparseBox3DLoss`（含可选 centerness / yawness）。

### 2. 问题 1：`loss_cls` 的 `avg_factor` 与 focal 监督样本不一致

**现象**：回归（box）在优化，但分类与其它项尺度/梯度行为异常。

**原因要点**：

- `FocalLoss` 在 **所有「匈牙利匹配上且 `cls_target` 有效」** 的 query 上计算。
- 原实现里 `cls_loss` 的 `avg_factor` 使用了 **`num_pos`**，而 `num_pos` 来自 **回归 mask**，该 mask 在 `cls_threshold_to_reg > 0` 时还会再要求 **`cls.max().sigmoid() > 阈值`**。
- 即：**分子**覆盖「全部匹配 query」，**分母**却是「过置信度阈值的子集」，归一化与真实监督数量不一致，易导致 cls 项梯度尺度异常。

**建议修改**：cls 单独使用与 focal 一致的 valid 样本数作为 `avg_factor`（例如对 `cls_target_flat` 上非 `ignore_index` 的匹配数做 `sum`，并 `clamp(min=1)`）。

### 3. 问题 2：`cls_threshold_to_reg` 与分类先验冲突

**原因要点**：

- `SparseBox3DRefinementModule` 中分类分支常用 **前景先验约 0.01** 的 bias 初始化，各类别 sigmoid 量级约 **0.01**。
- 若 **`cls_threshold_to_reg = 0.05`**，则训练初期大量 **已匹配** 样本会被挡在回归分支外（box / cns / yns 只在 mask 为 True 上算），而 cls 仍在匹配集合上算，**优化目标错位**：box 可能主要靠少数样本或后期才稳定对齐。

**建议修改**：默认关闭该过滤（例如阈值 **&lt; 0** 时不启用），或改为 **极小阈值**（如 `1e-3`），或 **warmup 后再打开**。

### 4. 问题 3：解码分数 = `cls_sigmoid × centerness_sigmoid`

**原因要点**（见 `sparse4d/decoder.py`）：

- 若存在 `quality`，最终 top-k 分数会乘以 **centerness 的 sigmoid**。
- 训练初期 centerness 目标 `exp(-||Δxyz||)` 往往偏小，容易把 quality  logits 压得很负 → **最终 score 接近 0**，表现像「没有检测结果」，即使 cls 在缓慢学习。

**建议修改**：对 quality 最后一层 bias 做合理先验初始化（例如与 **0.5** 先验对应的 `bias_init_with_prob`），避免一上来就把 decode 分数乘没；调试阶段也可暂时关闭「乘 centerness」对比实验。

### 5. 其它可排查项（未改代码前可对照）

- **匈牙利代价权重**：`sparse4d/dn_sampler.py` 中 `cls_weight` / `box_weight` 影响匹配质量。
- **`use_dn`**：配置中若为 `False`，则无 denoising 分支 loss，仅主分支；与「是否要用 DN」的预期一致即可。
- **类别数与标签**：`num_classes` 与 `det_class_names` / GT 下标一致。

---

## 三、建议的代码改动清单（供你本地验证时对照）

以下为「建议方向」，可按需逐项打开/验证：

1. **`sparse4d/head.py`**：`loss_cls(..., avg_factor=与 focal valid 匹配数一致)`，勿与回归 `num_pos`（含 `cls_threshold_to_reg`）混用。
2. **`sparse4d/build_model.py`（或构建 `Sparse4DHead` 处）**：`cls_threshold_to_reg` 设为 **不启用**（如 `-1`）或极小正数。
3. **`sparse4d/detection3d_blocks.py`**：`SparseBox3DRefinementModule.init_weight` 中为 **quality** 分支（CNS/YNS）设置合理 bias 先验。
4. **`bev_backbone.py`**：`grid_sample` 与 `normalize_coords` 的 **`align_corners` 一致**。
5. **数据**：确认 `lidar2camera` 与 BEV/自车坐标定义一致。

---

## 四、版本说明

- 文档基于对工程逻辑的阅读整理；**具体行号以你当前分支/工作区文件为准**。
- 若你之后在其它目录（非本 worktree）改代码，请以实际文件内容为准同步本文档。
