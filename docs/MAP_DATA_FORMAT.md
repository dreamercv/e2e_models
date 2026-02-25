# Map Head 真值格式与 DataLoader 约定

本文档说明：**地图真值应包含哪些内容**、**各字段的精确格式**、**多顺序 GT 如何构造**，以及 **DataLoader 如何输出才能直接用于 MapHead.loss**。

---

## 一、模型侧接口

MapHead 训练时需要的真值通过 `metas` 传入，并在 model 中调用：

```python
map_losses = map_head.loss(map_pred, gt_map_bboxes, gt_map_labels, gt_map_pts)
```

其中 `gt_map_bboxes`、`gt_map_labels`、`gt_map_pts` 来自 `metas["gt_map_bboxes"]` 等，且为 **list 长度 = B**（B 为当前 batch 中“用于 map 的”样本数，通常为 backbone 取最后一帧 BEV 时的 batch 大小）。

---

## 二、地图真值内容与格式（metas）

| 键 | 类型 | 格式说明 |
|----|------|----------|
| **gt_map_bboxes** | `list` of `torch.Tensor` | **长度 = B**。第 b 个元素对应第 b 个样本（当前帧/最后一帧），形状 `(N_b, 4)`，**x1, y1, x2, y2**（在 **pc_range** 内，与 BEV 范围一致，如 x∈[-80,120], y∈[-40,40]）。该样本无地图线时可为空 tensor `(0, 4)`。 |
| **gt_map_labels** | `list` of `torch.Tensor` | **长度 = B**。第 b 个元素形状 `(N_b,)`，dtype `long`，值为**类别下标**（0 到 num_classes-1，如 0=divider, 1=ped_crossing, 2=boundary）。与 gt_map_bboxes 的 N_b 一一对应。 |
| **gt_map_pts** | `list` of `torch.Tensor` | **长度 = B**。第 b 个元素为**折线点集**，支持两种形状：<br>• **单顺序**：`(N_b, num_pts, 2)`，每条线 num_pts 个点，(x,y) 在 pc_range 内。<br>• **多顺序**：`(N_b, num_orders, num_pts, 2)`。num_orders 表示同一条线的多种起点/方向（见下文「多顺序 GT」）。<br>无地图线时可为 `(0, num_pts, 2)` 或 `(0, num_orders, num_pts, 2)`。 |

**约定**：

- 坐标系与 BEV 一致（一般为自车/ego 坐标系），单位与你的场景一致（通常为米）。
- **pc_range** 与 `build_map_head(..., bev_bounds=...)` 一致，如 `[-80, -40, 120, 40]` 表示 xmin, ymin, xmax, ymax。
- **bbox** 可由折线点集 minmax 得到：`x1,y1 = pts.min(dim=0); x2,y2 = pts.max(dim=0)`，再保证 (x1,y1) 为左上、(x2,y2) 为右下即可。

---

## 三、多顺序 GT（解决“顺序歧义”）

同一条线可以有两种点序（如 A→B 或 B→A），匹配时需要对每种顺序算 Chamfer，取最小代价并得到 **order_index**。真值需提供 **num_orders** 种点序。

### 3.1 线段（开曲线，如 lane divider 的一段）

- **num_orders = 2**：正序 + 反序。
- 正序：原始采样点 `pts (num_pts, 2)`。
- 反序：`pts.flip(0)`。
- 构造后形状：`(N_b, 2, num_pts, 2)`。

### 3.2 闭合折线（如 crosswalk 多边形）

- **num_orders = num_pts**（或固定如 4）：循环移位，每种起点一种顺序。
- 第 k 种顺序：`torch.roll(pts, shifts=-k, dims=0)`。
- 构造后形状：`(N_b, num_orders, num_pts, 2)`。

### 3.3 代码示例（单样本）

```python
def build_gt_map_pts_one_sample(polylines, num_pts=20, num_orders=2):
    """
    polylines: list of (num_pts_orig, 2) 或 (num_pts_orig, 2) 的数组，每条线点数可不同
    返回: (N, num_orders, num_pts, 2)，N=len(polylines)
    """
    out = []
    for pts in polylines:
        pts = torch.as_tensor(pts, dtype=torch.float32)
        if pts.shape[0] != num_pts:
            pts = F.interpolate(pts.unsqueeze(0).permute(0, 2, 1), size=num_pts, mode='linear', align_corners=True).squeeze(0).permute(1, 0)
        if num_orders == 2:
            order0 = pts
            order1 = pts.flip(0)
            out.append(torch.stack([order0, order1], dim=0))
        else:
            orders = [torch.roll(pts, -k, dims=0) for k in range(num_orders)]
            out.append(torch.stack(orders, dim=0))
    return torch.stack(out, dim=0)
```

---

## 四、与 MapHead 参数的对应关系

| MapHead 参数 | 含义 | 真值需满足 |
|--------------|------|------------|
| num_pts_per_gt_vec | GT 折线点数（用于匹配与 loss） | gt_map_pts 的 num_pts 维可与此不同，loss 内会做插值；为一致可先插成 num_pts_per_gt_vec |
| num_classes | 类别数 | gt_map_labels 取值在 [0, num_classes-1] |
| pc_range | BEV 范围 | gt_map_bboxes、gt_map_pts 的 (x,y) 均在此范围内 |

---

## 五、DataLoader 输出格式（与 detection 一起用）

若同一 batch 既做检测又做地图，model 中会取**最后一帧 BEV** 给 map_head，因此 **map 真值只需对应“当前帧”**，list 长度为 B。

```python
batch = {
    "x": ...,                    # (B, T, N_cam, 3, H, W)
    "rots": ..., "trans": ..., "intrins": ..., "distorts": ...,
    "post_rot3": ..., "post_tran3": ..., "theta_mats": ...,
    "T_ego_his2curs": ...,       # (B, T, 4, 4)
    "metas": {
        "gt_labels_3d": [...],   # 检测，list 长 B*T
        "gt_bboxes_3d": [...],
        "gt_track_match": ...,   # 可选
        # ----- 地图（仅当使用 MapHead 且训练时） -----
        "gt_map_bboxes": list of Tensor,   # 长度 B，每元素 (N_b, 4) x1,y1,x2,y2
        "gt_map_labels": list of Tensor,   # 长度 B，每元素 (N_b,)
        "gt_map_pts": list of Tensor,      # 长度 B，每元素 (N_b, num_orders, num_pts, 2) 或 (N_b, num_pts, 2)
    },
}
```

**注意**：map 真值对应的是“当前时刻”的地图（通常取 T 的最后一帧或当前帧），因此若你的标注是按“当前帧”给的，list 长度就是 B；若标注是按 T 帧给的，需要取最后一帧的 map GT 组成 list 长度 B。

---

## 六、示例：Map Dataset + collate_fn

```python
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class MapDataset(Dataset):
    """假设每条样本有：当前帧图像/BEV 所需数据 + 当前帧地图标注。"""
    def __init__(self, ..., num_pts=20, num_orders=2):
        self.num_pts = num_pts
        self.num_orders = num_orders

    def __getitem__(self, idx):
        # 读取当前帧的地图标注
        polylines = self.get_polylines(idx)   # list of (L_i, 2)
        labels = self.get_labels(idx)          # (N,)
        bboxes = self.get_bboxes(idx)         # (N, 4) 或由 polylines minmax 得到
        pts = self._polylines_to_gt_pts(polylines)
        return {
            "x": ...,
            "gt_map_bboxes": bboxes,
            "gt_map_labels": labels,
            "gt_map_pts": pts,
        }

    def _polylines_to_gt_pts(self, polylines):
        out = []
        for pts in polylines:
            pts = torch.as_tensor(pts, dtype=torch.float32)
            if pts.shape[0] != self.num_pts:
                pts = F.interpolate(
                    pts.unsqueeze(0).permute(0, 2, 1),
                    size=self.num_pts, mode="linear", align_corners=True
                ).squeeze(0).permute(1, 0)
            if self.num_orders == 2:
                out.append(torch.stack([pts, pts.flip(0)], dim=0))
            else:
                out.append(torch.stack([torch.roll(pts, -k, dims=0) for k in range(self.num_orders)], dim=0))
        return torch.stack(out, dim=0) if out else torch.zeros(0, self.num_orders, self.num_pts, 2)

def collate_fn(batch_list):
    B = len(batch_list)
    out = {k: [b[k] for b in batch_list] for k in batch_list[0] if k != "metas"}
    out["metas"] = {
        "gt_map_bboxes": [b["gt_map_bboxes"] for b in batch_list],
        "gt_map_labels": [b["gt_map_labels"] for b in batch_list],
        "gt_map_pts": [b["gt_map_pts"] for b in batch_list],
    }
    return out
```

训练时：

```python
pred = map_head(bev_for_map, metas)
losses = map_head.loss(
    pred,
    metas["gt_map_bboxes"],
    metas["gt_map_labels"],
    metas["gt_map_pts"],
)
```

---

## 七、小结表

| 真值/输入 | 形状/类型 | 说明 |
|-----------|-----------|------|
| gt_map_bboxes | list 长 B，每元素 (N_b, 4) | x1,y1,x2,y2，pc_range 内 |
| gt_map_labels | list 长 B，每元素 (N_b,) | 类别下标 |
| gt_map_pts | list 长 B，每元素 (N_b, num_orders, num_pts, 2) 或 (N_b, num_pts, 2) | 折线点，pc_range 内；多顺序用于匹配消歧 |
| pc_range | [xmin, ymin, xmax, ymax] | 与 bev_bounds 一致 |

按上述格式制作地图真值并保证 `metas` 中 list 长度为 B，即可直接用于 `MapHead.loss` 与端到端训练。
