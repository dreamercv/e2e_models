# 相机位置编码（按 `dataset.py` 343–487 伪代码思路，适配本工程）

本文档严格按数据层注释中的 **两段流程** 组织，与当前工程约定一致：**全环视鱼眼**、标定 `lidar2camera`、增强参数 `post_rot3/post_tran3`、特征尺度 `final_dim` 与 `dowmsample`（config 中键名为 `dowmsample`）。

合并进 `dataset.py` 前可在独立脚本中 import 本节代码调试。

---

## 1. 伪代码与工程实现的逐步对照

### 1.1 `get_cam_pos_embedding` 思路（注释 344–433 行）

| 伪代码步骤 | 本工程实现 |
|------------|------------|
| `rays`：`(fH,fW)` 网格上 `(xs, ys, 1)` | 与 `Dataset.gen_rays()` 相同：`final_dim // ds` 分辨率，`linspace(0, W-1, fW)`、`linspace(0, H-1, fH)`，栈成 `(fH,fW,3)` |
| `points = rays - post_trans` 再 `inv(post_rots) @` | **增强逆变换**：`p_pre = inv(post_rot3) @ (p_aug - post_tran3)`（列向量）；实现里等价使用行向量形式 `p_pre_row = (p_aug - t)_row @ inv(post_rot)^T`，与 `bev_backbone.Splat.bev2eachroi` 中 `p_aug = post @ p_pre + t` 互逆 |
| 按 front/back/left/right 分相机 + `cv2.undistortPoints` / `fisheye.undistortPoints` | **不再按相机名分支**：每个样本已有自己的 `intrins/distorts`，统一走与 `Splat.projectPoints_fisheye` **互逆** 的 Torch 鱼眼反投影（见下文），避免与 BEV 投影模型不一致 |
| `points1 = inv(NewCameraMatrix) @ ...` 再 `positional_encoding(..., :2)` | 反投影得到相机系射线方向 `d_cam`，再 `d_ego = R^T @ d_cam`（`R` 为 `lidar2camera` 的旋转，与 `get_image_data` 中 `rots` 一致），对 `d_ego[..., :2]` 做 `positional_encoding` |

### 1.2 `gen_camera_pos_encoding` 思路（注释 435–485 行）

| 伪代码步骤 | 本工程实现 |
|------------|------------|
| 用 `post_img` 的 `H,W` 在 `(fH,fW)` 上 `meshgrid` | **两种等价用法**：(A) 与 `gen_rays` 一致，用 `final_dim` 与 `0..W-1` 网格（与 backbone 输入对齐）；(B) 用当前张量实际 `H,W` + `linspace(0.5, H-0.5, fH)`（注释写法），在 **已增强图像** 分辨率上对齐子像素中心 |
| `intrin_inv @ pixel` 得 `cam_dirs` 再归一化 | **针孔近似不适用鱼眼**：改为 `pixels_to_cam_dirs_fisheye`（与 1.1 相同内核） |
| 注释中 `ego_dirs = post_rot @ cam_dirs` | **纠正**：`post_rot` 在数据里是 **图像增强**，不是车体旋转；车体/自车方向应使用 **`d_ego = R_{lidar2cam}^T @ d_cam`**（方向向量，与平移无关） |
| `positional_encoding(dirs_xy, L)` → `(N,C,fH,fW)` | 与 `BEVBackbone.positional_encoding` 一致：`L` 个频带、`sin/cos`、输出通道 `C = 4 * L` |

---

## 2. `positional_encoding`（与 `BEVBackbone` 一致）

```python
import numpy as np
import torch


def positional_encoding(input_xy: torch.Tensor, L: int) -> torch.Tensor:
    """
    input_xy: (..., 2)  通常为 ego 系射线在 xy 平面上投影或前两维
    返回: (..., 4*L)   与 backbone/bev_backbone.py BEVBackbone.positional_encoding 一致
    """
    shape = input_xy.shape
    freq = 2 ** torch.arange(L, dtype=torch.float32, device=input_xy.device) * np.pi
    spectrum = input_xy[..., None] * freq
    sin, cos = spectrum.sin(), spectrum.cos()
    input_enc = torch.stack([sin, cos], dim=-2)
    input_enc = input_enc.view(*shape[:-1], -1)
    return input_enc
```

---

## 3. 鱼眼像素 → 相机系单位射线（与 `Splat.projectPoints_fisheye` 互逆）

```python
import torch
import torch.nn.functional as F


def _radial_factor(r: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    r = torch.clamp(r, min=eps)
    k1, k2, p1, p2, k3, k4, k5, k6 = dist.unbind(-1)
    t = torch.atan2(r, torch.ones_like(r))
    return t * (1 + k1 * t ** 2 + k2 * t ** 4 + k3 * t ** 6 + k4 * t ** 8) / r


def pixels_to_cam_dirs_fisheye(
    uv1: torch.Tensor,
    intrin_inv: torch.Tensor,
    dist: torch.Tensor,
) -> torch.Tensor:
    """uv1 (...,3), intrin_inv (...,3,3), dist (...,8) -> 单位方向 (...,3)，相机系。"""
    q = torch.matmul(intrin_inv, uv1.unsqueeze(-1)).squeeze(-1)
    a, b, w = q[..., 0], q[..., 1], q[..., 2]
    a, b = a / (w + 1e-6), b / (w + 1e-6)
    s = torch.sqrt(a * a + b * b + 1e-12)
    lo = torch.zeros_like(s)
    hi = torch.full_like(s, 2.0)
    for _ in range(24):
        mid = (lo + hi) * 0.5
        g = _radial_factor(mid, dist)
        f = mid * g - s
        hi = torch.where(f > 0, mid, hi)
        lo = torch.where(f <= 0, mid, lo)
    r = (lo + hi) * 0.5
    g = _radial_factor(r, dist)
    xn = a / (g + 1e-6)
    yn = b / (g + 1e-6)
    d = torch.stack([xn, yn, torch.ones_like(xn)], dim=-1)
    return F.normalize(d, dim=-1)
```

---

## 4. 流程一：`get_cam_pos_embedding` 风格（`self.rays` + 增强逆 + 鱼眼 + ego）

与注释一致：先构造与 `Dataset.gen_rays()` 相同的 `rays`，再对 **每张图** 的 `post_rots/post_trans` 做逆，然后鱼眼反投影、`R^T` 转到自车，最后 `positional_encoding`。

```python
def cam_dirs_ego_from_rays_style(
    rays_hw3: torch.Tensor,
    post_rots: torch.Tensor,
    post_trans: torch.Tensor,
    intrins: torch.Tensor,
    distorts: torch.Tensor,
    lidar2cam_rots: torch.Tensor,
) -> torch.Tensor:
    """
    rays_hw3: (fH, fW, 3)  与 Dataset.gen_rays() 输出一致
    post_rots: (N,3,3), post_trans: (N,3)
    intrins, distorts, lidar2cam_rots: (N,3,3), (N,8), (N,3,3)
    返回 d_ego: (N, fH*fW, 3)
    """
    device = post_rots.device
    dtype = post_rots.dtype
    fH, fW, _ = rays_hw3.shape
    N = post_rots.shape[0]
    pix = rays_hw3.to(device=device, dtype=dtype).view(1, fH, fW, 3).expand(N, fH, fW, 3)
    pix_flat = pix.reshape(N, -1, 3)
    inv_post = torch.inverse(post_rots)
    t = post_trans.unsqueeze(1)
    pre = torch.bmm(pix_flat - t, inv_post.transpose(1, 2))

    intrin_inv = torch.inverse(intrins)
    dist_exp = distorts.unsqueeze(1).expand(N, fH * fW, 8)
    intrin_inv_exp = intrin_inv.unsqueeze(1).expand(N, fH * fW, 3, 3)
    d_cam = pixels_to_cam_dirs_fisheye(pre, intrin_inv_exp, dist_exp)
    d_ego = torch.bmm(lidar2cam_rots.transpose(1, 2), d_cam.unsqueeze(-1)).squeeze(-1)
    return d_ego


def cam_pos_embed_flow1(
    rays_hw3: torch.Tensor,
    post_rots: torch.Tensor,
    post_trans: torch.Tensor,
    intrins: torch.Tensor,
    distorts: torch.Tensor,
    lidar2cam_rots: torch.Tensor,
    L: int,
) -> torch.Tensor:
    """返回 (N, 4*L, fH, fW)，与注释中 get_cam_pos_embedding 最终拼接前单相机张量同维。"""
    N = post_rots.shape[0]
    fH, fW = rays_hw3.shape[0], rays_hw3.shape[1]
    d_ego = cam_dirs_ego_from_rays_style(
        rays_hw3, post_rots, post_trans, intrins, distorts, lidar2cam_rots
    )
    dirs_xy = d_ego[..., :2].view(N, fH, fW, 2)
    emb = positional_encoding(dirs_xy, L)
    return emb.permute(0, 3, 1, 2).contiguous()
```

**在 `Dataset` 里用法示例**（在 `__init__` 已有 `self.rays = self.gen_rays()` 时）：

```python
# embed = cam_pos_embed_flow1(
#     self.rays, post_rots, post_trans, intrins, distorts, rots, L=4
# )
```

---

## 5. 流程二：`gen_camera_pos_encoding` 风格（按 `post_img` 尺寸下采样网格）

与注释一致：用 **增强后图像** 高宽 `H,W` 生成 `(fH,fW)` 子像素网格（可选 `0.5..H-0.5`），再走同一套鱼眼 + `R^T` + `positional_encoding`。

```python
def build_pixel_grid_feature_hw(
    H: int,
    W: int,
    fH: int,
    fW: int,
    device: torch.device,
    dtype: torch.dtype,
    align_center: bool = True,
) -> torch.Tensor:
    """
    返回 (fH, fW, 3) 齐次像素坐标，与注释中 meshgrid 一致。
    align_center=True: linspace(0.5, H-0.5, fH)（注释写法）
    False: linspace(0, H-1, fH)（与 gen_rays / backbone 一致）
    """
    if align_center:
        ys = torch.linspace(0.5, H - 0.5, fH, device=device, dtype=dtype)
        xs = torch.linspace(0.5, W - 0.5, fW, device=device, dtype=dtype)
    else:
        ys = torch.linspace(0, H - 1, fH, device=device, dtype=dtype)
        xs = torch.linspace(0, W - 1, fW, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    ones = torch.ones_like(grid_x)
    return torch.stack([grid_x, grid_y, ones], dim=-1)


def cam_pos_embed_flow2(
    post_img_N3HW: torch.Tensor,
    post_rots: torch.Tensor,
    post_trans: torch.Tensor,
    intrins: torch.Tensor,
    distorts: torch.Tensor,
    lidar2cam_rots: torch.Tensor,
    fH: int,
    fW: int,
    L: int,
    align_center: bool = True,
) -> torch.Tensor:
    """
    post_img: (N,3,H,W) 仅用于取 H,W；可与训练输入张量同尺寸
    返回 (N, 4*L, fH, fW)
    """
    _, _, H, W = post_img_N3HW.shape
    device = post_img_N3HW.device
    dtype = post_img_N3HW.dtype
    grid = build_pixel_grid_feature_hw(H, W, fH, fW, device, dtype, align_center)
    rays_hw3 = grid
    return cam_pos_embed_flow1(
        rays_hw3, post_rots, post_trans, intrins, distorts, lidar2cam_rots, L
    )
```

---

## 6. 与 `Dataset` 字段对齐的张量形状

| 来源 | 典型形状 | 说明 |
|------|----------|------|
| `get_image_data` | `post_rots: (n_cam, 3, 3)` 等 | 单 clip 多相机时先 `stack`，collate 后 batch 维再扩 |
| 模型侧 | `(b*m*t*n, ...)` | 与 `rearrange` 后 `x` 一致时，在 `forward` 前对 `post_rots` 等 `flatten` 成 `(N,3,3)` 再调用上述函数 |

---

## 7. 输出与 2D 特征拼接

- 单路输出：`cam_pos_embed` 为 **`(N, 4*L, fH, fW)`**。
- 与 `image_backbone` 输出 **`(N, C, fH, fW)`** 拼接：`torch.cat([feat, cam_pos_embed], dim=1)`，需同步修改后续 `Conv2d` 输入通道或加 `1×1` 降维。
- 若仅作几何先验，建议 **`torch.no_grad()`** 计算，省显存、稳定训练。

---

## 8. 与 `CAMERA_POS_EMBEDDING_FISHEYE.md` 的关系

- **本文**：按「`get_cam_pos_embedding` / `gen_camera_pos_encoding`」两段伪代码命名，便于对照 `dataset.py` 注释。
- **另一篇**：同一鱼眼模型与 `bev_backbone` 对齐的集中说明；实现内核可复用同一套 `pixels_to_cam_dirs_fisheye`。

调试完成后将本节函数迁入 `utils/cam_pos_embed.py` 或在 `Dataset` 中封装方法再合并主分支即可。
