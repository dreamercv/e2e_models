# 纯 PyTorch MapHead 与 config (maptr_nano_r18_110e.py) 对照

本文档逐项对比 `sparse4d_bev_torch/map_head.py` 与官方 config 中 **decoder / bbox_coder / positional_encoding / loss_cls / loss_bbox / loss_iou / loss_pts / loss_dir** 及 assigner 是否一致。

---

## 一、Config 中的定义摘要

```python
# maptr_nano_r18_110e.py
decoder=dict(
    type='MapTRDecoder',
    num_layers=2,
    return_intermediate=True,
    transformerlayers=dict(
        type='DetrTransformerDecoderLayer',
        attn_cfgs=[
            dict(type='MultiheadAttention', embed_dims=256, num_heads=4, dropout=0.1),
            dict(type='CustomMSDeformableAttention', embed_dims=256, num_levels=1, im2col_step=192),
        ],
        feedforward_channels=512, ffn_dropout=0.1,
        operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'))),
bbox_coder=dict(
    type='MapTRNMSFreeCoder',
    post_center_range=[...], pc_range=point_cloud_range, max_num=50, voxel_size=voxel_size, num_classes=3),
positional_encoding=dict(
    type='LearnedPositionalEncoding',
    num_feats=128, row_num_embed=80, col_num_embed=40),
loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
loss_bbox=dict(type='L1Loss', loss_weight=0.0),
loss_iou=dict(type='GIoULoss', loss_weight=0.0),
loss_pts=dict(type='PtsL1Loss', loss_weight=5.0),
loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.005)),
assigner: cls_cost=FocalLossCost weight=2.0, reg_cost=BBoxL1Cost weight=0.0,
          iou_cost=IoUCost giou weight=0.0, pts_cost=OrderedPtsL1Cost weight=5.
```

---

## 二、逐项对照结果

| 项目 | Config (maptr_nano_r18_110e) | 本实现 (map_head.py) | 是否一致 |
|------|------------------------------|----------------------|----------|
| **decoder** | MapTRDecoder，2 层；每层 **MultiheadAttention (self)** + **CustomMSDeformableAttention (cross)**，FFN | **nn.TransformerDecoder**，每层标准 **MultiheadAttention**（self+cross 均为 MHA），无 Deformable | ❌ **不一致**：缺 Deformable cross-attention |
| **bbox_coder** | MapTRNMSFreeCoder：post_center_range 过滤、max_num=50、voxel 等后处理 | 无 bbox_coder；仅有 **decode()**：score 阈值 + denormalize 得到 polylines | ❌ **不一致**：无 NMS/range 等后处理 |
| **positional_encoding** | LearnedPositionalEncoding(128, row=80, col=40)，加在 BEV 上 | **未使用**：BEV 仅 `bev_proj` + flatten，无 BEV 位置编码 | ❌ **不一致**：缺 BEV 位置编码 |
| **loss_cls** | **FocalLoss**，sigmoid，gamma=2，alpha=0.25，weight=2.0 | **CrossEntropy** + 正负样本权重平衡（无 Focal） | ❌ **不一致**：Focal vs CE |
| **loss_bbox** | L1Loss，**weight=0.0** | L1（仅正样本），有权重 | ⚠️ **语义一致**：config 未用 bbox loss；本实现有，可设 0 |
| **loss_iou** | GIoULoss，**weight=0.0** | **未实现** | ⚠️ **一致**：config 未用；本实现无该项 |
| **loss_pts** | **PtsL1Loss**，weight=5.0（**有序点对点 L1**：\|pred−target\|，点一一对应） | **Chamfer loss**（点集对点集，无顺序一一对应） | ❌ **不一致**：L1 点对点 vs Chamfer |
| **loss_dir** | PtsDirCosLoss，**weight=0.005** | direction_cosine_loss，默认 **loss_dir_weight=2.0** | ❌ **不一致**：权重差 400 倍（0.005 vs 2.0） |
| **assigner** | FocalLossCost(2) + BBoxL1Cost(0) + IoUCost(0) + **OrderedPtsL1Cost(5)** | cls_cost + **reg_cost(bbox L1)** + **Chamfer cost(pts)**；无 IoU cost | ❌ **部分一致**：config 用 pts 的 **OrderedPtsL1Cost**（展平点 L1 距离矩阵），本实现用 **Chamfer** 作为 pts cost |

---

## 三、细节说明

### 3.1 Decoder

- **Config**：Cross-attention 为 **CustomMSDeformableAttention**，以 reference 为采样中心在 BEV 上做可变形注意力。
- **本实现**：使用 **nn.TransformerDecoderLayer**，cross-attention 为普通 **MultiheadAttention**，无 deformable。
- **影响**：收敛/精度可能不同；本实现更简单、易调试。

### 3.2 loss_pts 与 assigner 的 pts_cost

- **Config**  
  - **OrderedPtsL1Cost**：pred `(num_query, num_pts, 2)`、gt `(num_gt, num_orders, num_pts, 2)` → 展平后 L1 距离矩阵，对多 order 取 min。  
  - **PtsL1Loss**：匹配后 pred 与 target **同顺序、同点数**，逐点 L1：`|pred - target|`。
- **本实现**  
  - **Assigner**：pts cost 为 **Chamfer**（点集对点集）。  
  - **Loss**：**Chamfer**（点集对点集）。
- **结论**：config 是“有序点 L1”，本实现是“无序点集 Chamfer”；两者在数学和梯度上都不等价。

### 3.3 loss_dir 权重

- Config：`loss_weight=0.005`。  
- 本实现：`loss_dir_weight=2.0`（且再乘在 direction_cosine_loss 上）。  
- 若要对齐 config，应设 **loss_dir_weight=0.005**。

### 3.4 loss_cls

- Config：**FocalLoss**（sigmoid，gamma=2，alpha=0.25）。  
- 本实现：**CrossEntropy** + 正负样本权重。  
- 要对齐需在 map_head 中改为 FocalLoss（或接出与官方相同的 Focal 接口）。

### 3.5 Bbox 表示

- 官方：归一化与回归用 **cxcywh**（transform_box 里 bbox_xyxy_to_cxcywh）。  
- 本实现：全程 **x1,y1,x2,y2**（xyxy），未用 cxcywh。  
- 仅影响 bbox 的归一化/反归一化形式，不改变“点集 minmax 得到 bbox”的逻辑。

---

## 四、结论与建议

| 项目 | 与 config 一致性 | 建议 |
|------|------------------|------|
| decoder | ❌ 无 Deformable | 若要对齐官方，需接 Deformable cross-attention；否则保持 MHA 即可。 |
| bbox_coder | ❌ 无 | 推理阶段可按需加 post_center_range、max_num 等后处理。 |
| positional_encoding | ❌ 无 | 可在 BEV 上加强 LearnedPositionalEncoding 与 config 一致。 |
| loss_cls | ❌ Focal vs CE | 建议改为 FocalLoss（gamma=2, alpha=0.25），weight=2.0。 |
| loss_bbox | ⚠️ config=0 | 本实现可保留；若严格对齐 config，将 bbox loss 权重设为 0。 |
| loss_iou | ⚠️ config=0 | 可不实现；若需与代码结构一致，可加 GIoU 并设 weight=0。 |
| loss_pts | ❌ L1 vs Chamfer | 要对齐 config 需改为 **有序点 L1**（PtsL1Loss）；保留 Chamfer 则与 config 不一致。 |
| loss_dir | ❌ 权重 2.0 vs 0.005 | 建议设 **loss_dir_weight=0.005** 与 config 一致。 |
| assigner pts_cost | ❌ OrderedPtsL1 vs Chamfer | 要对齐 config 需改为 **OrderedPtsL1Cost**（展平点 L1 矩阵 + 多 order 取 min）。 |

**总结**：当前纯 PyTorch MapHead **与 maptr_nano_r18_110e.py 的 decoder / bbox_coder / positional_encoding / loss_cls / loss_pts / loss_dir 及 assigner 的 pts_cost 并不完全一致**。若要以该 config 复现或对齐官方结果，需要按上表逐项调整（尤其是 decoder 结构、loss_pts/assigner 的 L1 vs Chamfer、loss_dir 权重、loss_cls 的 Focal）。
