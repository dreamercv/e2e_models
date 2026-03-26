# BEV Backbone 与 Dynamic 分支检测损失问题说明

本文档整理对 `backbone/bev_backbone.py` 的几何链路核对结论，以及 Dynamic 训练时「box loss 下降但 cls / cns / yns 不下降、无检测结果」的可能原因与建议修改方向，便于你本地逐项验证。**未强制改代码**，仅作记录。

---

## 一、`bev_backbone.py`：流程是否与「自车 3D → 增强图像 → BEV 采样」一致

### 1.1 主流程（结论：与描述一致）

`Splat.bev2eachroi` 大致为：

1. 在 BEV 网格配置 `xbound / ybound / zbound` 上生成自车（数据侧与 LiDAR 对齐时即「车体」）坐标系下的 3D 格点 `create_voxels`。
2. 外参：`p_cam = R @ p + t`，与数据里 `lidar2camera` 的 `rot` / `tran` 用法一致。
3. 透视归一化、`projectPoints_fisheye` + 内参，得到投影像素（齐次）。
4. **数据增强**：`post_rot3 @ p + post_tran3`，与 `dataset.py` 里 `img_transform` 生成的 `post_rot3` / `post_tran3` 一致（先投影到相机像素，再叠 resize/crop/flip/rotate）。
5. 按特征图 `input_size`（如 `(H,W)=(128,384)`）归一化到 `[-1,1]`，再 `grid_sample` 取图像特征，多高度层在通道维拼接后经卷积融合。

### 1.2 使用与实现时需注意

| 点 | 说明 |
|----|------|
| 坐标系 | 数据字段为 `lidar2camera`。若标注中 **LiDAR 系与自车 BEV 不一致**，需增加 ego→lidar 外参，否则存在系统偏差。 |
| `grid_sample` 与归一化 | `normalize_coords` 使用 `(shape - 1)` 作分母，等价于 **align_corners=True** 的像素映射；若 `grid_sample` 使用默认 `align_corners=False`，边界处会有细微不一致，建议两者统一。 |
| 鱼眼 | `r=0` 时 `t/r` 可能数值不稳定，光轴附近可加小 `epsilon` 或分支处理。 |
| 死代码 | `bev2eachroi` 末尾对 `depths` 的 reshape 未参与返回，可删或留给深度相关扩展。 |
| `get_cam_pos_embedding` | 依赖未在 `BEVBackbone` 上定义的 `data_aug_conf`、`downsample`，若调用会报错；当前若未使用可忽略。 |
| `bev_emb` | `ground_points = self.points[2]` 是 **Z 维第 3 个切片**，未必对应物理「地面」高度，仅作位置编码时需与 `zbound` 语义对齐。 |

---

## 二、Dynamic 训练：`train.py` → `Model` → `det3d_head.loss`

训练入口：`train.py` 中 `model(...)` → `forward_dynamic_branch` → `self.det3d_head(bev_feat, metas)` 与 `self.det3d_head.loss(out, metas)`。总损失为各 `det3d_*` 张量之和（见 `model/models.py`）。

### 2.1 现象

- **box loss** 在下降；
- **cls / cns / yns** 几乎不下降或行为异常；
- 解码/可视化 **几乎没有检测结果**。

### 2.2 原因分析（建议验证）

#### （1）`loss_cls` 的 `avg_factor` 与 focal 监督集合不一致（核心）

在 `sparse4d/head.py` 的 `loss` 中，曾对 **分类 focal** 使用与回归相同的 `num_pos`，而 `num_pos` 来自经 **`cls_threshold_to_reg` 过滤后的 mask**（与「匈牙利匹配上且参与回归的 query」一致）。

但 **FocalLoss** 实际在 **所有 `cls_target != ignore_index` 的匹配 query** 上计算（含未过置信度阈值的匹配）。

若分母误用「过阈后的回归正样本数」，会导致 **分类损失的归一化与真实监督数量不一致**，梯度尺度失真，易出现 **box 在优化、cls/cns/yns 表现异常**。

**建议修改方向**：分类分支使用 **独立的 `avg_factor`**，例如与 focal 中 `valid` 样本数一致（如对 `cls_target_flat` 做与 `FocalLoss` 相同的 valid 统计后再 `clamp(min=1)`）。

#### （2）`cls_threshold_to_reg` 与分类先验冲突

`SparseBox3DRefinementModule` 中分类分支常用 **前景先验约 0.01** 的 bias 初始化，对应 **sigmoid 约 0.01**。

若同时设置 **`cls_threshold_to_reg = 0.05`**，则训练早期 **大量已匹配的 query** 因 `max(sigmoid(cls)) < 0.05` 被挡在 **回归分支**（含 box / cns / yns）之外，造成：

- 回归只在一小部分样本上更新；
- 分类仍在全部匹配上算 focal；

优化目标错位，表现为 **回归与分类/质量头不同步**。

**建议**：默认关闭该过滤（如阈值 **&lt; 0** 或 **0** 表示不启用），或改为 **极小值（如 1e-3）** / **warmup 后再打开**。

#### （3）解码分数被 centerness 压制（「无框」）

`sparse4d/decoder.py` 中，若存在 `quality`，最终分数大致为：

**`score ≈ cls_sigmoid × centerness_sigmoid`**

训练初期 centerness 目标 `exp(-||Δxyz||)` 往往较小，质量分支 logits 易被压到很负 → **最终分数接近 0**，可视化阈值过滤后像「没有检测」。

**建议**：对 quality 最后一层 bias 做合理先验（例如 `bias_init_with_prob(0.5)` 对应 cen/yawness），避免 decode 阶段把分数乘没；调试阶段也可暂时 **不关 centerness 相乘** 或 **降低对 cen 的依赖** 以对比。

#### （4）其它可顺带排查项

- **匈牙利代价权重**：`sparse4d/dn_sampler.py` 中 `cls_weight` / `box_weight` 比例是否导致匹配偏向 box、分类代价信号弱。
- **`use_dn`**：配置中若关闭 denoising，则无 `dn_*` 分支损失，仅主分支；与「是否仅主分支在学」相关。
- **类别数与标签**：`num_classes` 与 `dataset` 中 `det_class_names` 映射一致，避免标签越界被 `clamp` 错类。

---

## 三、建议修改清单（供你本地逐项打开/验证）

以下为「方向性」清单，**按你的节奏改一版再训**即可。

1. **`sparse4d/head.py`**：`loss_cls` 的 `avg_factor` 改为与 focal valid 样本数一致，勿与回归 `num_pos`（经 `cls_threshold` 过滤）混用。
2. **`sparse4d/build_model.py`（或 config 中传入 `Sparse4DHead` 的参数）**：将 `cls_threshold_to_reg` 设为 **不启用** 或 **极小正数**，避免与 0.01 先验冲突。
3. **`sparse4d/detection3d_blocks.py`**：`SparseBox3DRefinementModule.init_weight` 中为 **quality（CNS/YNS）** 设置合理 bias 先验。
4. **`sparse4d/decoder.py`（可选）**：调试时确认 `score = cls * cen` 是否把分数压没；必要时临时改为仅用 cls 或提高 cen 先验。
5. **`backbone/bev_backbone.py`（可选）**：统一 `grid_sample` 的 `align_corners` 与 `normalize_coords` 公式。

---

## 四、相关文件路径（本仓库）

| 主题 | 路径 |
|------|------|
| BEV 投影与采样 | `backbone/bev_backbone.py` |
| 增强与标定 | `dataset/dataset.py`（`img_transform`、`lidar2camera`） |
| 训练入口 | `train.py` |
| 检测头与 loss 组装 | `sparse4d/head.py` |
| 匈牙利匹配 | `sparse4d/dn_sampler.py` |
| 分类/回归/质量损失 | `sparse4d/losses.py` |
| refine 与 quality 头 | `sparse4d/detection3d_blocks.py` |
| 解码与 score | `sparse4d/decoder.py` |
| Head 构建默认 | `sparse4d/build_model.py` |

---

*文档生成用于记录讨论结论；若与当前分支代码行号不一致，请以仓库内实际实现为准。*
