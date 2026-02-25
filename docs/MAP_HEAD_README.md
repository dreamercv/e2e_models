# Map Head 说明文档（MapTR 风格）

本模块 `map_head.py` 实现 MapTR 风格的地图元素检测头：**纯 PyTorch，不依赖 mmcv**。输入 BEV 特征，输出固定数量的地图折线（类别 + 点集），支持匈牙利匹配、多顺序 GT、Chamfer/方向 loss 与解码。

---

## 一、整体结构

```
BEV 特征 (B, C, H, W)
    → bev_proj → memory (B, H*W, embed_dims)
    → Transformer Decoder（map queries 做 cross-attention）
    → 每层：ref 迭代细化 → reg_branches 得到点 → transform_box 得到 bbox + pts
    → cls_branches（按 instance 聚合点后分类）
    → 输出：map_cls_scores, map_bbox_preds, map_pts_preds（均为 list，每层一个）
```

- **输入**：`bev_feature` 形状 `(B, C, H, W)`，例如 backbone 最后一帧的 BEV。
- **输出**：见下文「输出形状」；训练时用 `MapHead.loss(...)` 算 loss，推理时用 `MapHead.decode(...)` 得到折线列表。

---

## 二、与官方 MapTR 的 Tricks 对照

| 项目 | 官方 MapTR | 本实现 | 说明 |
|------|------------|--------|------|
| **匈牙利匹配** | cls + bbox L1 + **IoU** + Chamfer | cls + bbox L1 + Chamfer | 已实现匈牙利；IoU cost 未加，可后续在 assigner 中加 |
| **多顺序 GT** | gt_shifts_pts，匹配时对 order 取 min，得到 order_index | 同：`gt_pts` 为 (N_gt, num_orders, num_pts, 2)，Chamfer 取 min，order_index 用于取 target | 已实现 |
| **Loss 分类** | Focal 等 | CE + 正负样本权重（pos/neg 平衡） | 已实现 |
| **Loss 回归** | bbox L1 + **GIoU** + Chamfer + 方向 | bbox L1 + Chamfer + 方向 | 已实现 L1/Chamfer/方向；未实现 bbox GIoU |
| **Chamfer loss** | 有 | 有，pred/target 点集，可 src/dst 权重 | 已实现 |
| **方向 loss** | PtsDirCosLoss（相邻点方向余弦） | direction_cosine_loss，dir_interval 控制相邻点 | 已实现 |
| **Reference 迭代细化** | with_box_refine，每层 ref = sigmoid(inv_sigmoid(ref)+delta) | 同 | 已实现 |
| **点集 → bbox** | minmax | transform_box：点集 minmax 得 x1,y1,x2,y2 | 已实现 |
| **分类按 instance 聚合** | hs.view(..., num_vec, num_pts, C).mean(2) 再分类 | 同 | 已实现 |
| **Query** | instance_pts（instance_embed + pts_embed） | use_instance_pts=True 时同 | 已实现 |
| **坐标归一化** | pc_range 内 [0,1] | normalize_2d_pts / normalize_2d_bbox，denormalize 解码 | 已实现 |
| **多层 aux loss** | 每层 decoder 都算 loss，前面层权重 0.5 | 同，aux_loss_weight | 已实现 |
| **预测点数 ≠ GT 点数** | 插值对齐再 Chamfer | F.interpolate(..., mode='linear') 对齐 | 已实现 |

结论：**匈牙利匹配、多顺序 GT、Chamfer、方向 loss、ref 迭代、decode 等核心流程均已实现**；可选增强为 assigner 加 IoU cost、loss 加 bbox GIoU。

---

## 三、匈牙利匹配（Assigner）

- **MapTRAssigner** 对每个 batch 样本单独做一次匹配。
- **代价**：`cost = cls_weight * cls_cost + reg_weight * reg_cost + pts_weight * pts_cost`
  - **cls_cost**：对 GT 类别取预测概率的负值，形状 (num_vec, num_gts)。
  - **reg_cost**：bbox 四维（x1,y1,x2,y2）归一化后的 L1 距离矩阵 (num_vec, num_gts)。
  - **pts_cost**：Chamfer 距离矩阵 (num_vec, num_gts)；若 GT 有多顺序，先对每个 order 算 Chamfer，再取 `min(dim=order)` 得到 pts_cost 和 **order_index**。
- **匈牙利**：`scipy.optimize.linear_sum_assignment(cost)` 得到 pred↔gt 一一对应。
- **输出**：`assigned_gt_inds (B, num_vec)`：0 表示背景，1-based 为匹配到的 GT 下标；`order_index (B, num_vec)`：匹配到的 GT 使用的顺序下标（多顺序时有效）。

---

## 四、Loss 计算

`MapHead.loss(pred_dict, gt_bboxes_list, gt_labels_list, gt_pts_list)` 返回字典，键如 `loss_map_cls`、`loss_map_bbox`、`loss_map_pts`、`loss_map_dir`，以及各中间层的 `loss_map_cls_d0` 等。

1. **loss_map_cls**：分类。根据 `assigned_gt_inds` 为每个 query 赋 label（正样本用匹配的 GT 类别，负样本用 num_classes）；正负样本加权 CE，负样本权重 = num_pos / num_neg。
2. **loss_map_bbox**：仅正样本，归一化 bbox 的 L1。
3. **loss_map_pts**：仅正样本，Chamfer(pred_pts, target_pts)；target 根据 **order_index** 取对应顺序的 GT 点；若 pred 与 GT 点数不同先插值。
4. **loss_map_dir**：仅正样本，相邻点方向余弦 loss（1 - cos），`dir_interval` 控制取哪些相邻点，权重 `loss_dir_weight`。
5. **多层**：最后一层权重 1.0，其余层权重 `aux_loss_weight`（如 0.5）。

---

## 五、Decode 解码

`MapHead.decode(pred_dict, score_threshold=0.5)`：

1. 取最后一层：`cls_scores = pred_dict["map_cls_scores"][-1]`，`pts_preds = pred_dict["map_pts_preds"][-1]`。
2. 分类：`probs = cls_scores.sigmoid()`，`max_scores, labels = probs.max(dim=-1)`。
3. 过滤：`keep = max_scores >= score_threshold`。
4. 坐标：对保留的 `pts_preds` 做 **denormalize_2d_pts** 得到物理坐标（pc_range 内）。
5. 返回：`list` 长度 B，每元素为 `{"labels", "scores", "pts"}`（均为该样本保留的折线的 tensor）。

---

## 六、输出形状（forward）

| 键 | 类型 | 形状（默认 num_vec=100, num_pts_per_vec=20, num_classes=3） |
|----|------|-------------------------------------------------------------|
| map_cls_scores | list of Tensor | 每元素 (B, num_vec, num_classes)，如 (B, 100, 3) |
| map_bbox_preds | list of Tensor | 每元素 (B, num_vec, 4)，归一化 x1,y1,x2,y2 |
| map_pts_preds | list of Tensor | 每元素 (B, num_vec, num_pts_per_vec, 2)，归一化点 |
| init_ref | Tensor | (B, num_vec*num_pts_per_vec, 2) |
| inter_refs | list of Tensor | 每元素 (B, num_vec*num_pts_per_vec, 2) |

list 长度 = num_decoder_layers + 1（最后一层为最终输出）。

---

## 七、使用方式

```python
from sparse4d_bev_torch import build_map_head

map_head = build_map_head(
    bev_feat_dim=256,
    embed_dims=256,
    num_vec=100,
    num_pts_per_vec=20,
    num_pts_per_gt_vec=20,
    num_classes=3,
    num_decoder_layers=6,
    bev_bounds=([-80.0, 120.0], [-40.0, 40.0]),
)
# 前向
pred = map_head(bev_feature, metas)
# 训练
losses = map_head.loss(pred, gt_map_bboxes, gt_map_labels, gt_map_pts)
# 推理
polylines = map_head.decode(pred, score_threshold=0.5)
```

真值格式与 DataLoader 约定见 **[MAP_DATA_FORMAT.md](MAP_DATA_FORMAT.md)**。
