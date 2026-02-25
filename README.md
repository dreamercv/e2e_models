## 项目简介

本目录下实现了一个**纯 PyTorch 的端到端 BEV 感知模型**，在同一套 BEV 特征上同时完成：

- **3D 目标检测**（基于 Sparse4D 思路）
- **跨帧目标跟踪**（TrackHead + 匈牙利匹配）
- **矢量车道线 / 地图要素检测**（参考 MapTR 的矢量 MapHead）

实现目标是：

- 保留官方 `Sparse4D-main` 与 `MapTR-main` 的核心思想与接口风格；
- 完全去除 `mmcv` / `mmdet` 依赖，方便在纯 PyTorch 环境下调试、扩展与二次开发；
- 统一 BEV backbone，做到**一次 BEV 编码，三种任务共享**。

> 说明：目前已通过 `test_model.py` 的完整 forward、自监督 loss 计算以及各模块 decoder / decode 的形状检查，确保**所有模块在假数据上可以无错误跑通**。性能与官方实现的数值一致性暂未验证。

---

## 整体架构

主入口文件为 `model.py`，核心模块结构如下：

- `backbone/bev_backbone.py`
  - 负责多相机多帧图像 → 时序对齐的 BEV 特征 (`feature_maps`，形状约为 `(B*T, C, H, W)`)，内部完成相机几何变换与时序对齐。
- `sparse4d/` 目录
  - `instance_bank.py`：Sparse4D 的锚点与实例特征缓存（此处设置为**无跨 forward 时序 cache**）。
  - `detection3d_blocks.py`：3D 框编码器、关键点生成器、refine 模块等。
  - `bev_aggregation.py`：在 BEV 上进行多尺度 / 多关键点特征聚合，替代官方在图像平面上的采样。
  - `head.py`：`Sparse4DHead`，实现检测 + DN（Denoising） + loss 逻辑。
  - `decoder.py`：`SparseBox3DDecoder`，将 head 输出解码为 3D 框。
  - `dn_sampler.py`：`DenoisingSampler`，实现 DN 采样、匈牙利匹配与 DN 损失相关的 target 生成。
  - `track_head.py`：`TrackHead`、`track_affinity_loss`、`decode_track`，实现跨帧关联。
- `maptr/` 目录
  - `deformable_attn.py`：多尺度 Deformable Attention 的纯 PyTorch 版本（不依赖 C++/CUDA 扩展）。
  - `decoder.py`：`DetrTransformerDecoderLayer`、`MapTRDecoder`，实现 MapTR 风格的 query 解码器。
  - `bbox_coder.py`：`MapTRNMSFreeCoder`，实现 NMS-free 的后处理与规范化坐标反归一化。
  - `assigner_loss.py`：`MapTRAssigner` + 各种 loss（Chamfer、方向余弦、Focal 等）。
  - `map_head.py`：`MapHead`，将 BEV 特征映射为矢量 map 查询、预测 bbox 和点集，并计算 loss / 匈牙利匹配 / decode。
- `model.py`
  - 提供统一构建函数：`build_sparse4d_bev_head`、`build_track_head`、`build_map_head`/`build_map_head_from_maptr_config`、`build_sparse4d_bev_model`。
  - 内部类 `Sparse4DBEVModel` 将 backbone + 检测 head + track_head + map_head 组装成一个可训练 / 推理的整体模型。
- `test_model.py`
  - 构造假输入与假 GT，跑一次完整 forward，检查：
    - 检测 / 跟踪 / MapTR 各自的 loss 标量是否可计算；
    - DN 分支 loss 是否能算；
    - 各模块 decoder / decode 输出的形状是否符合预期。

---

## 关键函数与模块说明

### 1. `build_sparse4d_bev_head`（3D 目标检测）

定义位置：`model.py`

作用：

- 以纯 PyTorch 方式构建 Sparse4D 风格的检测 head：
  - 使用 `InstanceBank` 存储 anchor 与实例特征；
  - 通过 `SparseBox3DEncoder` 编码 anchor；
  - 用 `MHAWrapper` + `SimpleFFN` + `BEVFeatureAggregation` + `SparseBox3DRefinementModule` 堆叠形成多层 decoder（`operation_order`）；
  - 可选构建 `DenoisingSampler`，实现 DETR 风格的 DN 训练；
  - 可选构建 `SparseBox3DDecoder`，在推理阶段从分类 / 回归输出 decode 出 boxes_3d。

与官方 `Sparse4D-main` 的主要差异：

- **完全去除 mmcv/mmdet**，不再使用 registry / build_from_cfg，而是直接 PyTorch 模块组合。
- **时序处理位置不同**：
  - 官方 Sparse4D 在 head 内部保存时序 cache、跨 forward 更新。
  - 本实现将时序全部放到 backbone 内，head 接收到的是 `(B*T, C, H, W)`，自身不做跨 forward 的 cache/更新，`InstanceBank` 设置 `num_temp_instances=0`。
- **特征采样位置不同**：
  - 官方在**图像特征平面**上进行 keypoints 采样。
  - 本实现通过 `BEVFeatureAggregation` 在 **BEV 上采样/聚合**，简化几何处理并统一多任务接口。
- **DN / loss 路径**：
  - DN：使用自实现的 `DenoisingSampler`，逻辑参照官方 target.py / dn_sampler 的思想。
  - Loss：`Sparse4DHead.loss()` 里，将 head 输出与 GT 做匈牙利匹配并计算分类 / 回归 / 质量（centerness / yawness）loss。

当前状态：

- 在 `test_model.py` 中启用 `use_dn=True`，已验证：
  - 普通 loss：`loss_cls_*`、`loss_box_*`、`loss_cns_*`、`loss_yns_*` 可正常计算；
  - DN loss：`loss_cls_dn_*`、`loss_box_dn_*`、`loss_cns_dn_*`、`loss_yns_dn_*` 也能计算（数值很大是因为随机权重 + 假数据，见终端打印说明）。
- Decoder 测试：
  - 通过 `SparseBox3DDecoder.decode` 在 eval 模式下成功跑通，输出为每帧 `{'boxes_3d', 'scores_3d', 'labels_3d', 'cls_scores', 'instance_ids'}`。
  - 在自测脚本中为了方便观察，将 `score_threshold` 设为 0 或较小值即可看到非空解码结果。

### 2. `build_track_head` / `TrackHead`（目标跟踪）

定义位置：`model.py` / `sparse4d/track_head.py`

作用：

- 接收 head 输出的 `seq_features` / `seq_anchors`（形状约 `(B, T, N, C)` / `(B, T, N, box_dim)`），以及 `T_ego_his2curs`（历史到当前的自车变换矩阵），输出：
  - `track_affinity`：形状 `(B, T, N, N)` 的帧间亲和矩阵。
- 损失与解码：
  - `track_affinity_loss`：对 `track_affinity` 与 `gt_track_match`（亲和 GT）计算 loss；
  - `decode_track`：在 eval 模式下对亲和矩阵做匈牙利匹配，输出：
    - `track_ids`: `(B, T, N)`，
    - `track_positions`: `(B, T, N, 3)`。

与官方 `Sparse4D-main` 的差异：

- 接口及思想与官方一致（亲和矩阵 + 匈牙利匹配），但：
  - 这里的特征来自统一的 **BEV backbone**，而非图像 backbone；
  - 实现完全基于 PyTorch，无 mmcv / mmdet。
- 当前测试：
  - 在 `test_model.py` 中，`loss_track` 计算正常；
  - `decode_track` 在 eval 模式下输出的 `track_ids` / `track_positions` 形状与预期一致。

### 3. `build_map_head` / `build_map_head_from_maptr_config` / `MapHead`（矢量 Map / 车道线检测）

定义位置：`model.py` / `maptr/map_head.py`

作用：

- 参考 `MapTR-main` 的 `pts_bbox_head`，在 BEV 特征上实现矢量化的 map 头：
  - 使用 `MapTRDecoder`（多层 Transformer decoder）对查询进行多次 refine；
  - 预测每条矢量的类别 / bbox / 点集；
  - 使用 `MapTRAssigner` 做一对一的 Hungarian 匹配；
  - 使用 Chamfer loss、方向余弦 loss 等计算训练损失；
  - 在 eval 时通过 `MapTRNMSFreeCoder.decode` 输出矢量化多边形 / polyline。

`build_map_head_from_maptr_config`：

- 按官方 `maptr_nano_r18_110e.py` 中 `pts_bbox_head` 的 decoder / bbox_coder / pos_encoding 配置构建 `MapHead`，包括：
  - decoder 层数 / heads 数 / FFN 维度 / im2col_step；
  - BEV 网格大小（row_num_embed / col_num_embed）；
  - `post_center_range`、`max_num` 等 bbox_coder 参数；
  - `pc_range` 等点云范围。

与官方 `MapTR-main` 的关键差异（详见 `docs/MAP_HEAD_CONFIG_CHECK.md`）：

- **Decoder 结构**：
  - 官方：`MultiheadAttention (self)` + `CustomMSDeformableAttention (cross)`。
  - 本实现：`MultiheadAttention` 用于 self / cross，暂未接入 deformable cross-attention。
- **BBox coder**：
  - 官方 `MapTRNMSFreeCoder` 有完整的 post_center_range 过滤、max_num 等后处理。
  - 本实现已实现基本的 NMS-free coder 和 pc_range 反归一化逻辑，但一些细节（voxel_size 等）做了简化。
- **Positional Encoding**：
  - 官方在 BEV 上使用 `LearnedPositionalEncoding`。
  - 本实现中对应部分已支持 row / col 嵌入，但整体与官方完全一致性仍有差异。
- **Loss 与 Assigner**：
  - 官方使用 **OrderedPtsL1Cost + PtsL1Loss**（严格的有序点匹配）；
  - 本实现使用 **Chamfer distance** 作为 pts cost 和 pts loss，更接近“无序点集”匹配；
  - `loss_dir` 权重默认值与官方 config 有差异（官方 0.005，本实现初始为 2.0，已在文档中提示可调）。

当前状态：

- 在 `test_model.py` 的自测中：
  - `loss_map_cls_*` / `loss_map_bbox_*` / `loss_map_pts_*` / `loss_map_dir_*` 均能正常计算（标量）；
  - eval 模式下 `map_head.decode` 输出：
    - `bboxes`: `(num_vec, 4)`，
    - `scores`: `(num_vec,)`，
    - `labels`: `(num_vec,)`，
    - `pts`: `(num_vec, num_pts_per_vec, 2)`。
- 与官方数值/精度尚未严格对齐，但**前向 / 损失 / 匹配 / decode 路径已全部打通**。

### 4. `build_sparse4d_bev_model` / `Sparse4DBEVModel`

定义位置：`model.py`

作用：

- 将 BEV backbone、检测 head、可选 track_head、可选 map_head 组合为一个统一的 PyTorch `nn.Module`：

```python
model = build_sparse4d_bev_model(
    bev_backbone=backbone,
    head=head,
    track_head=track_head,
    map_head=map_head,
    num_temporal_frames=num_temporal_frames,
)
```

`forward` 流程：

1. `bev_backbone` 接收多帧多相机图像及几何信息 → 输出 `(B*T, C, H, W)`；
2. `Sparse4DHead` 接收 `feature_maps` 与 `metas`：
   - train 模式下仅输出检测相关的特征与 logit，loss 通过 `head.loss()` 计算；
   - eval 模式下可配合 `SparseBox3DDecoder` 输出 3D 框；
3. `TrackHead` 接收 `seq_features` / `seq_anchors` / `T_ego_his2curs`，输出 `track_affinity`：
   - train 模式下计算 `loss_track`；
   - eval 模式下 `decode_track` 输出 `track_ids` / `track_positions`；
4. `MapHead` 接收最后一帧 BEV `(B, C, H, W)` 与 `metas`：
   - train 模式下输出 `loss_map`；
   - eval 模式下 `decode` 输出 `map_polylines`。

---

## 与官方项目的整体差异总结

### 相比 `Sparse4D-main`

- **依赖**：不依赖 `mmcv` / `mmdet`，所有模块用纯 PyTorch 重写。
- **时序处理方式**：
  - 官方：head 内部有跨 frame 的 cache / update，本实现将时序全部交由 BEV backbone 处理，head 只看 `(B*T, C, H, W)`。
- **特征采样**：
  - 官方在图像平面采样，本实现直接在 BEV 上采样聚合（`BEVFeatureAggregation`）。
- **DN 与 loss**：
  - 保留 DN 思想与 FocalLoss + SparseBox3DLoss 组合；
  - 但 GT/预测的 batch 组织方式（B vs B*T）略有不同，代码中已通过扩展 GT / DN target 的方式适配当前架构。

### 相比 `MapTR-main`

详见 `docs/MAP_HEAD_CONFIG_CHECK.md`，简要归纳：

- **Decoder**：暂未接入 Deformable cross-attention，而是用标准 `MultiheadAttention` 代替，结构更简洁。
- **BBox coder / pos_encoding**：实现了核心逻辑，但在 voxel_size / pos_encoding 等细节上做了简化，与官方 config 并非 1:1 对齐。
- **Loss / Assigner**：最大的区别在于 **有序点 L1（官方） vs Chamfer（本实现）**，会影响匹配方式与梯度形式。
- **权重与超参**：`loss_dir` 等项的权重初始值与官方不同，可根据需要在 config 层面调整以靠近官方行为。

整体而言，本项目更偏向于：

- **保持结构与接口风格相似**，方便迁移 / 对比；
- 在实现上做了适度精简与纯 PyTorch 化，便于调试与集成；
- 目前主要验证的是**形状正确、梯度连通、loss/匹配逻辑可运行**，而不是完全复现官方数值。

---

## 使用与测试

### 环境与依赖

- Python 3.8+
- PyTorch（版本与原 Sparse4D 项目一致或相近）
- 其它依赖参见上层仓库的 `requirements` / `environment` 配置。

### 快速自测：`test_model.py`

在当前目录下运行：

```bash
cd projects/e2e_models
python test_model.py
```

脚本会执行：

1. 构建一个包含：
   - BEV backbone
   - Sparse4D 检测 head（含 DN 与 decoder）
   - TrackHead
   - MapTR MapHead
   的完整模型；
2. 使用假的输入与 GT 跑 **一次 train forward**：
   - 打印 head 输出的 key 与 `seq_features` / `seq_anchors` 形状；
   - 打印检测 / 跟踪 / MapTR 的各项 loss（仅作数值 / 形状 sanity check，不代表合理大小）；
3. 切换到 eval 模式再跑 **一次 forward**，并：
   - 调用 `SparseBox3DDecoder.decode` 检查检测框解码输出的形状；
   - 调用 `map_head.decode` 检查 map_polylines 的形状；
   - 调用 `decode_track` 检查 `track_ids` / `track_positions` 的形状。

> 注意：由于模型未训练、输入为随机假数据，loss 数值可能非常大，decode 后框 / polyline 也不具备物理意义，这些只用于验证代码路径是否正确打通。

### 调试与扩展建议

- 若要**对齐官方 Sparse4D / MapTR 的数值表现**：
  - 建议参考 `docs/MAP_HEAD_CONFIG_CHECK.md` 中的对照表，逐项调整 decoder 结构、loss 配置与权重；
  - 特别是 MapTR 的 `loss_pts` / `assigner` 的 `OrderedPtsL1Cost`，若有需要可将本实现的 Chamfer 替换为有序点 L1。
- 若只关心**工程集成与端到端联调**：
  - 可以保持当前简化版本，通过真实数据训练，先观察整体联动是否符合预期，再根据需求逐步与官方结构靠拢。

---

## 当前观察结论

- **Forward 与 loss**：检测 / 跟踪 / MapTR + DN 分支在假数据上均可稳定 forward 与反向（loss 标量可算）。
- **Decoder / decode**：Sparse4D 检测 decoder、MapTR decode、跟踪 decode_track 均在 eval 模式下跑通，输出形状与设计一致。
- **与官方的一致性**：
  - 结构与接口大体对齐，但在 decoder 细节、loss 形式、权重配置上仍有明显差异；
  - 当前实现更适合作为一个**工程化、易调试的纯 PyTorch 版本**，而不是严格的官方复现。

