# 环视鱼眼相机位置编码（Camera Position Embedding）

本文档给出与当前工程 **数据管线、`bev_backbone.Splat` 几何约定一致** 的相机位置编码实现思路与**可复制代码**，便于在 `dataset` 或 `image_backbone` 侧调试；调试通过后再合并进主工程。

**目标**：在下采样特征分辨率 `(fH, fW) = (final_dim[0]//ds, final_dim[1]//ds)` 上，对每个像素构造一条 **自车（与 lidar 对齐）坐标系下的视线方向**，再对方向的分量做 **sin/cos 多频位置编码**，得到 `(N, C, fH, fW)`，可与 2D 特征 `concat`，使网络感知「当前特征来自环视中哪一个相机、在图像上的哪一片区域、对应自车系大致朝向哪里」。

---

## 1. 与伪代码（`dataset.py` 343–487 行）的对应关系

| 伪代码思路 | 本工程适配 |
|------------|------------|
| `rays`：特征图网格像素 `(u,v,1)` | 与 `Dataset.gen_rays()` 一致：`linspace(0, W-1, fW)`、`linspace(0, H-1, fH)` |
| 先 `inverse(post_rot)`、`post_trans` 还原到增强前 | 与 `Splat.bev2eachroi` **相反顺序**：增强图像上一点 `p_aug` → `p_pre = inv(post_rot3) @ (p_aug - post_tran3)`（列向量约定见下） |
| 针孔：`inv(K) @ 像素` 得相机射线 | **全程鱼眼**：在 `p_pre`（齐次像素）上用与 `projectPoints_fisheye` **互逆** 的映射得到相机系单位方向，而不是针孔 `inv(K)` 一步 |
| `ego_dirs = R * cam_dirs` | 外参来自标定 `lidar2camera`：`p_cam = R @ p_ego + t`，**方向向量**满足 `d_cam = R @ d_ego`，故 `d_ego = R^T @ d_cam`（`R` 与 `bev_backbone` 中 `rots` 一致） |
| `positional_encoding(dirs_xy, L)` | 与 `BEVBackbone.positional_encoding` 相同：对二维输入做 `2^i * π * x` 的 sin/cos，拼成 `2 * L * 2 = 4L` 维（每个坐标维度 `L` 个频率） |

**注意**：原注释里 `gen_camera_pos_encoding` 把 `post_rot` 写成「相机到 ego」是 **命名易混**；当前数据里 `post_rot3/post_tran3` 来自 `img_transform`，表示 **二维图像增强** 的齐次变换，不是车体姿态。

---

## 2. 齐次坐标约定（与 `bev_backbone` 一致）

- 列向量：`p = [u, v, w]^T`。
- 增强：`p_aug = post_rot3 @ p_pre + post_tran3`（`post_rot3` 为 `3×3`，`post_tran3` 最后一维常为 0）。
- 反变换：`p_pre = inv(post_rot3) @ (p_aug - post_tran3)`。

像素平面：`p_aug = [u, v, 1]^T`，`u` 列、`v` 行，与 OpenCV / `grid_sample` 常用约定一致。

---

## 3. 鱼眼：与 `projectPoints_fisheye` 互逆

正向（见 `backbone/bev_backbone.py` `Splat.projectPoints_fisheye`）在 **归一化平面** 上：

- 输入齐次 `(x_n, y_n, 1)`（即 `X/Z, Y/Z`）。
- `r = sqrt(x_n^2 + y_n^2)`，`t = atan2(r, 1)`。
- `radial = t * (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8) / r`（代码中对 `r` 需加小量避免除零）。
- `(x_d, y_d) = (x_n * radial, y_n * radial)`，再 `K @ [x_d, y_d, 1]^T` 得到像素。

**反向**：已知像素 `(u,v)`，先 `q = inv(K) @ [u,v,1]^T`，取 `a = q0/q2, b = q1/q2`，则 `(a,b) = (x_n * radial(r), y_n * radial(r))`，且 `sqrt(a^2+b^2) = r * radial(r)`。对标量方程 `r * radial(r) = s`（`s = sqrt(a^2+b^2)`）用 **二分或牛顿法** 求 `r`，再 `x_n = a / radial(r)`, `y_n = b / radial(r)`，归一化得单位射线方向 `d_cam ∝ [x_n, y_n, 1]`（再除以范数）。

这样与训练时 BEV 投影使用 **同一套** 鱼眼多项式，避免针孔近似带来的左右/环视不一致。

---

## 4. 输出形状与拼接建议

- 输入：`post_rots (N,3,3)`, `post_trans (N,3)`, `intrins (N,3,3)`, `distorts (N,8)`，`rots (N,3,3)` 为 `lidar2camera` 的 `R`（与 `dataset.get_image_data` 一致）。
- 输出：`cam_pos_embed`，形状 **`(N, C, fH, fW)`**，其中 `C = 4 * L`（与 `BEVBackbone.positional_encoding` 对 2D 输入一致：`L` 个频率 × sin/cos × 2 个分量）。
- 与 `image_backbone` 输出 `(N, 256, fH, fW)` 拼接时：`torch.cat([feat, cam_pos_embed], dim=1)` → 通道 `256 + C`；需在模型里改 `in_channels` 或加 `1×1` 投影回 256。

**环视**：`N = batch 中相机维」展开后的数量（例如 `b*m*t*n` 与当前 `x` 对齐），每张图独立算一遍网格，无需再按旧伪代码分 front/back/left/right 写死相机索引；若需 **相机 ID embedding**，可额外 `+ cam_id_embed[cam_idx]`。

---

## 5. 参考实现（可直接复制到独立 `.py` 调试）

以下模块 **不修改** 现有仓库文件；合并时可将函数挂到 `Dataset` 上或收到 `utils/cam_pos_embed.py`。

```python
# camera_pos_embedding_fisheye.py — 参考实现，与 bev_backbone.Splat 鱼眼模型一致
import numpy as np
import torch
import torch.nn.functional as F


def positional_encoding_2d(input_xy: torch.Tensor, L: int) -> torch.Tensor:
    """
    与 BEVBackbone.positional_encoding 一致：input_xy (..., 2) -> (..., 4*L)
    """
    shape = input_xy.shape
    device, dtype = input_xy.device, input_xy.dtype
    freq = (2 ** torch.arange(L, device=device, dtype=dtype)) * np.pi
    spectrum = input_xy[..., None] * freq  # (..., 2, L)
    sin, cos = spectrum.sin(), spectrum.cos()
    enc = torch.stack([sin, cos], dim=-2)  # (..., 2, 2, L)
    enc = enc.view(*shape[:-1], -1)
    return enc


def _radial_factor(r: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    """projectPoints_fisheye 中的 radial = t * poly(t) / r"""
    eps = 1e-6
    r = torch.clamp(r, min=eps)
    k1, k2, p1, p2, k3, k4, k5, k6 = dist.unbind(-1)
    t = torch.atan2(r, torch.ones_like(r))
    rad = t * (1 + k1 * t ** 2 + k2 * t ** 4 + k3 * t ** 6 + k4 * t ** 8) / r
    return rad


def pixels_to_cam_dirs_fisheye(
    uv1: torch.Tensor,
    intrin_inv: torch.Tensor,
    dist: torch.Tensor,
) -> torch.Tensor:
    """
    uv1: (..., 3) 齐次像素 [u,v,1]
    intrin_inv: (..., 3, 3)
    dist: (..., 8)
    返回单位方向 (..., 3) 相机坐标系，前向为 +Z（与投影链一致）
    """
    # q = inv(K) @ [u,v,1] 对应投影前 (xn*rad, yn*rad, 1)
    q = torch.matmul(intrin_inv, uv1.unsqueeze(-1)).squeeze(-1)
    a, b, w = q[..., 0], q[..., 1], q[..., 2]
    a, b = a / (w + 1e-6), b / (w + 1e-6)
    s = torch.sqrt(a * a + b * b + 1e-12)

    # 解 s = r * radial(r)，r 在 [0, rmax] 上单调近似成立；二分法
    lo = torch.zeros_like(s)
    hi = torch.full_like(s, 2.0)  # 视场限制可调
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


def cam_ray_ego_fisheye(
    post_rot: torch.Tensor,
    post_tran: torch.Tensor,
    intrin: torch.Tensor,
    dist: torch.Tensor,
    lidar2cam_rot: torch.Tensor,
    fH: int,
    fW: int,
    ogfH: int,
    ogfW: int,
) -> torch.Tensor:
    """
    post_rot: (N,3,3), post_tran: (N,3)
    intrin: (N,3,3), dist: (N,8)
    lidar2cam_rot: (N,3,3) 与 dataset 中 rots 一致
    返回 ego / lidar 对齐坐标系下的单位方向 (N, fH*fW, 3)
    """
    device = post_rot.device
    dtype = post_rot.dtype
    N = post_rot.shape[0]
    xs = torch.linspace(0, ogfW - 1, fW, device=device, dtype=dtype)
    ys = torch.linspace(0, ogfH - 1, fH, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    ones = torch.ones_like(grid_x)
    pix = torch.stack([grid_x, grid_y, ones], dim=-1).view(1, fH, fW, 3).expand(N, fH, fW, 3)

    pix_flat = pix.reshape(N, -1, 3)
    # 行向量约定：p_pre_row = (p_aug - t)_row @ inv(post_rot)^T，与 bev 中 p_aug = post @ p_pre + t 互逆
    inv_post = torch.inverse(post_rot)
    t = post_tran.unsqueeze(1)
    pre = torch.bmm(pix_flat - t, inv_post.transpose(1, 2))

    intrin_inv = torch.inverse(intrin)
    dist_exp = dist.unsqueeze(1).expand(N, fH * fW, 8)
    intrin_inv_exp = intrin_inv.unsqueeze(1).expand(N, fH * fW, 3, 3)

    d_cam = pixels_to_cam_dirs_fisheye(pre, intrin_inv_exp, dist_exp)
    # d_ego = R^T @ d_cam（p_cam = R @ p_ego + t => 方向 d_cam = R @ d_ego）
    R = lidar2cam_rot
    d_ego = torch.bmm(R.transpose(1, 2), d_cam.unsqueeze(-1)).squeeze(-1)
    return d_ego  # (N, fH*fW, 3)


def camera_pos_embed_from_batch(
    post_rots: torch.Tensor,
    post_trans: torch.Tensor,
    intrins: torch.Tensor,
    distorts: torch.Tensor,
    rots: torch.Tensor,
    final_dim: tuple,
    ds: int,
    L: int = 4,
) -> torch.Tensor:
    """
    post_rots, post_trans, intrins, distorts, rots: 已与单相机样本对齐 (N, ...)
    final_dim: (H_img, W_img) 与 config final_dim 一致
    返回: (N, 4*L, fH, fW)
    """
    ogfH, ogfW = final_dim
    fH, fW = ogfH // ds, ogfW // ds
    d_ego = cam_ray_ego_fisheye(
        post_rots,
        post_trans,
        intrins,
        distorts,
        rots,
        fH,
        fW,
        ogfH,
        ogfW,
    )
    dirs_xy = d_ego[..., :2].view(d_ego.shape[0], fH, fW, 2)
    emb = positional_encoding_2d(dirs_xy, L)
    return emb.permute(0, 3, 1, 2).contiguous()
```

**说明**：

- `cam_ray_ego_fisheye` 里采用 **行向量** 写法：`p_pre = (p_aug - t) @ inv(post_rot)^T`，与 `bev_backbone` 中 `p_aug = post_rot @ p_pre + t`（列向量）等价；若验证时出现左右镜像，与 `Splat.bev2eachroi` 对同一像素反推一点对比即可。
- 二分上界 `hi=2` 仅示例，极广鱼眼可略增大；若 `s` 超出模型有效范围，应对该像素置零或 mask。

---

## 6. 合并进工程时的检查清单

1. **形状**：与 `forward_img_backbone` 输出的 `(b*m*t*n, 256, fH, fW)` 在 **同一 `(fH,fW)`** 上对齐（`final_dim` 与 `dowmsample` 与 config 一致）。
2. **设备**：与 `rots` 等同批上 GPU。
3. **梯度**：位置编码若仅作「几何提示」，可用 `torch.no_grad()` 与 `bev_backbone.get_cam_pos_embedding` 草稿一致，省显存。
4. **验证**：固定一帧，将 `d_ego` 的 `xy` 画成箭头场（俯视）或检查前相机主方向是否大致沿 +X/+Y（取决于你车体坐标定义）。

---

## 7. 版本说明

- 文档基于当前仓库 `dataset.py`、`backbone/bev_backbone.py` 阅读整理；合并代码前请以你分支上的实际张量形状为准做一次单步 `print(shape)` 对齐。
