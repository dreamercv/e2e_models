#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpenCV fisheye 投影等价实现（学习用，不接入训练）。

对应关系：
- OpenCV: cv2.fisheye.projectPoints(X_cam, rvec=0, tvec=0, K, D[:4])
- 这里的 torch 版：输入相机系 3D 点 + K + D(k1,k2,k3,k4)，输出像素坐标

模型是 OpenCV fisheye（Kannala-Brandt / equidistant）：
1) 归一化平面: x = X/Z, y = Y/Z
2) r = sqrt(x^2 + y^2), theta = atan(r)
3) theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
4) scale = theta_d / r (r=0 时置 1)
5) x_d = x * scale, y_d = y * scale
6) [u,v,1]^T = K * [x_d, y_d, 1]^T
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import cv2


def project_points_fisheye_torch(
    points_camera: torch.Tensor,
    camera_intrinsic: torch.Tensor,
    distortion_coeffs: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    与 cv2.fisheye.projectPoints 等价的 torch 实现（rvec=tvec=0）。

    Args:
        points_camera: (N, 3), 相机坐标系点
        camera_intrinsic: (3, 3), K
        distortion_coeffs: (>=4,), 取前4个 [k1,k2,k3,k4]
        eps: 数值稳定项

    Returns:
        projected_points: (N, 2), 像素坐标 (u, v)
    """
    assert points_camera.ndim == 2 and points_camera.shape[1] == 3
    assert camera_intrinsic.shape == (3, 3)
    assert distortion_coeffs.numel() >= 4

    k1, k2, k3, k4 = distortion_coeffs[:4]
    X = points_camera[:, 0]
    Y = points_camera[:, 1]
    Z = points_camera[:, 2]

    x = X / (Z + eps)
    y = Y / (Z + eps)

    r2 = x * x + y * y
    r = torch.sqrt(r2 + eps)
    theta = torch.atan(r)

    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4
    theta_d = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)

    scale = theta_d / r
    scale = torch.where(r > eps, scale, torch.ones_like(scale))

    x_d = x * scale
    y_d = y * scale

    ones = torch.ones_like(x_d)
    xyz_d = torch.stack([x_d, y_d, ones], dim=-1)  # (N,3)
    uvw = xyz_d @ camera_intrinsic.t()  # 行向量右乘
    projected_points = uvw[:, :2] / (uvw[:, 2:3] + eps)
    return projected_points


def project_points_fisheye_opencv(
    points_camera: np.ndarray,
    camera_intrinsic: np.ndarray,
    distortion_coeffs: np.ndarray,
) -> np.ndarray:
    """
    直接调用 OpenCV fisheye，便于对照。
    """
    pts = points_camera.reshape(-1, 1, 3).astype(np.float32)
    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)
    uv, _ = cv2.fisheye.projectPoints(
        pts,
        rvec,
        tvec,
        camera_intrinsic.astype(np.float32),
        distortion_coeffs[:4].astype(np.float32),
    )
    return uv.reshape(-1, 2)


def compare_torch_and_opencv(
    points_camera: np.ndarray,
    camera_intrinsic: np.ndarray,
    distortion_coeffs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 torch 结果、opencv 结果、两者误差，方便学习验证。
    """
    pts_t = torch.from_numpy(points_camera.astype(np.float32))
    k_t = torch.from_numpy(camera_intrinsic.astype(np.float32))
    d_t = torch.from_numpy(distortion_coeffs.astype(np.float32))

    uv_torch = project_points_fisheye_torch(pts_t, k_t, d_t).cpu().numpy()
    uv_cv = project_points_fisheye_opencv(points_camera, camera_intrinsic, distortion_coeffs)
    err = np.linalg.norm(uv_torch - uv_cv, axis=1)
    return uv_torch, uv_cv, err


if __name__ == "__main__":
    # 简单示例：你可以替换成自己的 K、D、points_camera
    K = np.array(
        [[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    D = np.array([-0.01, 0.001, -0.0001, 0.00001], dtype=np.float32)
    P = np.array(
        [
            [1.0, 0.0, 8.0],
            [2.0, 1.0, 12.0],
            [-1.0, 0.5, 10.0],
            [0.2, -0.3, 5.0],
        ],
        dtype=np.float32,
    )

    uv_t, uv_c, e = compare_torch_and_opencv(P, K, D)
    print("torch:\n", uv_t)
    print("opencv:\n", uv_c)
    print("pixel error:\n", e)
    print("max error:", e.max(), "mean error:", e.mean())
