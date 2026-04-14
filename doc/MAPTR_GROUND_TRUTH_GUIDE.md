# MapTR 真值制作指南（本仓库）

本文说明如何为当前工程中的 MapTR 风格 `MapHead` 制作训练真值：坐标系、张量形状、多顺序点集、与本仓库 `dataset.py` 的对接方式，以及自定义数据集的落地步骤。

---

## 一、真值在模型里怎么用

训练时 `Model.forward_static_branch` 从 `metas["static"]` 读取：

| 键名 | 含义 |
|------|------|
| `gt_labels_map3D` | 每条地图线段的类别 |
| `gt_bboxes_map3D` | 每条线在 BEV 上的轴对齐包围盒 |
| `gt_pts_map3D` | 每条线的折线点（支持多顺序） |

随后被展平为与 `B*T` 维 BEV 对齐的 list，传入 `MapHead.loss(...)`。字段命名沿用数据集输出名，与 `docs/MAP_DATA_FORMAT.md` 中的 `gt_map_*` 语义一致（仅键名不同）。

---

## 二、坐标系与范围

- **坐标系**：与 BEV / 自车一致，一般为 **ego 坐标系**，\(x\) 通常为车辆前进方向，\(y\) 为左侧（以你 `grid_conf` 与标定约定为准）。
- **单位**：米（与 `grid_conf` 一致）。
- **pc_range（2D）**：由 `config["grid_conf"]` 推导：
  - `xbound = [xmin, xmax, dx]` → \(x \in [xmin, xmax)\)
  - `ybound = [ymin, ymax, dy]` → \(y \in [ymin, ymax)\)
- 真值中的点与框应落在该范围内；训练时会在 `MapHead` 内归一化到 \([0,1]\)。

**必须与 `map_3d_head` / `build_map_head` 使用的 `grid_conf` 一致**，否则 loss 与可视化会整体偏移。

---

## 三、三条张量的精确定义

### 3.1 `gt_labels_map3D`

- **形状**：单样本内为 `(N,)`，`dtype=torch.long`。
- **语义**：第 \(i\) 条地图元素的类别下标，取值 `0 .. num_classes-1`。
- **本仓库默认 3 类**（与 `config["map_3d_head"]["num_classes"]` 一致）：
  - `0`：divider（车道线类）
  - `1`：ped_crossing（人行横道等）
  - `2`：boundary（路沿 / 不可行驶区域等）

若你自定义类别，需同时改 `num_classes` 与标签映射。

### 3.2 `gt_bboxes_map3D`

- **形状**：`(N, 4)`，`float32`。
- **格式**：**xyxy**，即 `[x1, y1, x2, y2]`，与折线点在同一坐标系。
- **推荐计算方式**：对每条线的点集取 min/max：
  - `x1 = pts[:,0].min()`, `x2 = pts[:,0].max()`
  - `y1 = pts[:,1].min()`, `y2 = pts[:,1].max()`

### 3.3 `gt_pts_map3D`（核心）

- **形状**：`(N, num_orders, num_pts, 2)`。
- **N**：地图线段（实例）条数。
- **num_pts**：每条线固定采样点数，默认 **20**（`dataset` 中 `config.get("map_num_pts", 20)`）。
- **num_orders**：同一条线的多种点序，用于匹配时消除方向/起点歧义（与官方 MapTR 的 shift 思想一致）。
- **最后一维**：`(x, y)`，物理坐标，在 pc_range 内。

**无地图时**：`N=0`，可为空 tensor，例如：

- `gt_labels`: `(0,)`
- `gt_bboxes`: `(0, 4)`
- `gt_pts`: `(0, num_orders, num_pts, 2)`

---

## 四、多顺序 `num_orders` 怎么定

本仓库 `get_anno_map3D` 的实现要点：

- 统一 `num_orders = 2 * num_pts`（例如 `num_pts=20` → `num_orders=40`）。
- **开曲线**（如 `line_3d`）：几何上只需 **正序 + 反序**；其余 order 用正序 **重复填充** 到 `num_orders`。
- **闭合折线**（如 `polygon_3d`）：用 **正向循环移位** `num_pts` 种 + **反向再循环移位** `num_pts` 种，共 `2*num_pts` 种，无需重复填充。

### 4.1 简化版（自定义数据时可直接采用）

若你暂时只有“一条线一个方向”的标注，可只构造 **2 个 order**（正序 + 反序），其余 order 用正序重复填充到 `2*num_pts`，与当前 `dataset.py` 行为一致。

### 4.2 最小示例（单条开曲线）

```python
import numpy as np
import torch

num_pts = 20
num_orders = 2 * num_pts  # 与本仓库一致

# 原始折线 (N_raw, 2)，先重采样到 num_pts（线性插值）
def resample_polyline(coords_xy, num_pts):
    n = coords_xy.shape[0]
    if n <= 1:
        raise ValueError("need at least 2 points")
    t_old = np.linspace(0.0, 1.0, n, dtype=np.float32)
    t_new = np.linspace(0.0, 1.0, num_pts, dtype=np.float32)
    x_new = np.interp(t_new, t_old, coords_xy[:, 0])
    y_new = np.interp(t_new, t_old, coords_xy[:, 1])
    return np.stack([x_new, y_new], axis=1).astype(np.float32)

pts = resample_polyline(your_polyline_xy, num_pts)  # (num_pts, 2)
pts_t = torch.from_numpy(pts).float()
orders = [pts_t, torch.flip(pts_t, [0])]
while len(orders) < num_orders:
    orders.append(pts_t)
gt_one_line = torch.stack(orders, dim=0)  # (num_orders, num_pts, 2)
```

多条线时，对每条线堆叠为 `(N, num_orders, num_pts, 2)`。

---

## 五、本仓库从 JSON 到真值的流程（`get_anno_map3D`）

参考：`dataset/dataset.py` 中 `get_anno_map3D`。

1. **读当前帧标注**：默认取时间窗 `recs` 的 **最后一帧** `recs[-1]` 的 label JSON。
2. **解析 `groups`**：按 `type` 映射到 3 类（如 `lane_line` → divider，`road_marker_line` + category==3 → crosswalk，`road_edge` / `non_drivable_area` → boundary）。
3. **几何**：`LidarFusion` + `line_3d` / `polygon_3d`，点取 `(x,y)`；可按业务过滤可见性（代码里用 `properties.v` 与相机下标过滤）。
4. **范围**：与 `grid_conf` 的 x/y 边界比较；可只保留部分在范围内的点。
5. **重采样**：每条线插值到固定 `num_pts`。
6. **bbox**：对重采样后的点做 min/max。
7. **多顺序**：按 `line_3d` / `polygon_3d` 分支构造 `gt_pts`，见上文。

你若使用 **非本 JSON 格式**，只需在 `Dataset` 里实现等价逻辑，最终输出上述三个 tensor 即可。

---

## 六、自定义数据集：推荐落地步骤

1. **定义类别表**  
   与 `num_classes` 一致，并写死 `class_id → 0..C-1`。

2. **统一坐标到 ego**  
   若标注在世界系或地图系，先乘 `T_world2ego`（或 `T_map2ego`），再生成折线。

3. **每条线 → 点序列**  
   至少 2 个点；建议按行驶方向或标注规范统一“顺/逆时针”，再重采样到 `num_pts`。

4. **构造多顺序**  
   开曲线：正序 + 反序 + 填充到 `2*num_pts`；闭合：循环移位 + 反向循环移位（见第四节）。

5. **bbox**  
   由 `gt_pts` 某一 order（如 order 0）做点 min/max 得 xyxy。

6. **在 `Dataset.__getitem__` 中返回**  
   `gt_labels_map3D` / `gt_bboxes_map3D` / `gt_pts_map3D`（单样本内为 tensor；`custom_collate` 会保持为 list of samples）。

7. **配置项检查**  
   - `task_flag["map3D"] = True`  
   - `gt_names["static"]` 包含上述三个键（见 `config/config.py`）  
   - `grid_conf` 与标定一致  
   - 可选 `config["map_num_pts"] = 20` 与 `num_pts_per_gt_vec` 对齐

---

## 七、与官方 MapTR 的 GT 对应关系

- 官方常用 **固定点数折线 + 多 shift 点集**（`gt_shift_pts_pattern` 等），本质是 **同一条线的多种顺序**。
- 本仓库用 **显式 `num_orders` 维** 表达，匹配时由 `MapTRAssigner` 在 order 维上取最优。
- 若你从官方 NuScenes 格式转换，需把 `fixed_num_sampled_points` / `shift_*` 转为本仓库的 `(N, num_orders, num_pts, 2)`。

---

## 八、自检清单

- [ ] 随机抽样本可视化：点集落在 BEV 范围内，与图像语义一致。  
- [ ] `gt_bboxes` 与 `gt_pts` 同一条线一致（min/max）。  
- [ ] `gt_labels` 取值在 `[0, num_classes-1]`。  
- [ ] 空场景：`N=0` 的 tensor 形状正确，训练不报错。  
- [ ] `num_pts` 与 `map_3d_head` 的 `num_pts_per_gt_vec` 一致或接受 loss 内插值。  

---

## 九、相关文档与代码

| 资源 | 路径 |
|------|------|
| 真值格式与 metas 约定 | `docs/MAP_DATA_FORMAT.md` |
| MapTR 实现总览 | `doc/MAPTR_IMPLEMENTATION_FULL_GUIDE.md` |
| 标注 JSON 示例说明 | `dataset/demo_map.md` |
| 真值生成逻辑 | `dataset/dataset.py` → `get_anno_map3D` |
| 匹配与归一化 | `maptr/assigner_loss.py` |

按上述格式制作真值后，即可与当前 `MapHead` 训练流程直接对接。
