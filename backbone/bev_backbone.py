# -*- encoding: utf-8 -*-
'''
@File         :bev_backbone_1.py
@Date         :2026/02/12 16:14:32
@Author       :Binge.Van
@E-mail       :afb5szh@bosch.com
@Version      :V1.0.0
@Description  :

'''

import os, sys

import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import colorsys
import torch.nn.functional as F
import torch

from torch import nn


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx


class Splat(nn.Module):
    def __init__(self, grid_conf, input_size=(128, 384)):
        super(Splat, self).__init__()
        self.input_size = input_size
        self.grid_conf = grid_conf

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'], self.grid_conf['ybound'], self.grid_conf['zbound'])
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)
        self.voxels = self.create_voxels()

    def create_voxels(self):
        xs = torch.linspace(self.bx[0] - self.dx[0] / 2, self.bx[0] - self.dx[0] / 2 + self.dx[0] * (self.nx[0] - 1),
                            self.nx[0], dtype=torch.float).view(1, 1, self.nx[0]).expand(self.nx[2], self.nx[1],
                                                                                         self.nx[0])
        ys = torch.linspace(self.bx[1] - self.dx[1] / 2, self.bx[1] - self.dx[1] / 2 + self.dx[1] * (self.nx[1] - 1),
                            self.nx[1], dtype=torch.float).view(1, self.nx[1], 1).expand(self.nx[2], self.nx[1],
                                                                                         self.nx[0])
        zs = torch.linspace(self.bx[2] - self.dx[2] / 2, self.bx[2] - self.dx[2] / 2 + self.dx[2] * (self.nx[2] - 1),
                            self.nx[2], dtype=torch.float).view(self.nx[2], 1, 1).expand(self.nx[2], self.nx[1],
                                                                                         self.nx[0])
        voxels = torch.stack((xs, ys, zs), -1)
        return nn.Parameter(voxels, requires_grad=False)

    def bev2eachroi(self, points, rots, trans, intrins, distorts, post_rots, post_trans):
        B, N, _, _ = rots.shape  # [bs*sq,roi,3,3]
        Z, Y, X, _ = points.shape
        P = Z * Y * X
        points = points.view(1, 1, Z * Y * X, 3).expand(B, N, P, 3).clone()

        points = rots.view(B, N, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        points = points + trans.view(B, N, 1, 3)
        # points = points.unsqueeze(-2).matmul(torch.inverse(Rs).view(B, N, 1, 3, 3)).squeeze(-2)

        x, y, z = torch.where(points[:, :, :, 2] < 0)
        points[x, y, z, 0] = 9999 * torch.ones_like(x, dtype=points.dtype)
        points[x, y, z, 1] = 9999 * torch.ones_like(x, dtype=points.dtype)

        depths = points[..., 2:]
        points = torch.cat((points[..., :2] / depths, torch.ones_like(depths)), -1)

        intrins = intrins.view(B, N, 1, 3, 3)
        distorts = distorts.view(B, N, 1, 8)
        points = self.projectPoints_fisheye(points, distorts, intrins)

        # points1 = points[:, 0:1]
        # intrins1 = intrins[:, 0:1].view(B, 1, 1, 3, 3)
        # distorts1 = distorts[:, 0:1].view(B, 1, 1, 8)
        # points[:, 0:1] = self.projectPoints_fisheye(points1, distorts1, intrins1)

        points = post_rots.view(B, N, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        points = points + post_trans.view(B, N, 1, 3)
        points = points.view(B, N, Z, Y, X, 3).permute(0, 1, 2, 4, 3, 5)[..., :2]

        # # 可视化验证
        # image_all_000 = feature.cpu().numpy()[0].transpose(1,2,0)
        # alll = []
        # for i in range(6):
        #     pts  = points[0,0,i].reshape(-1, 2).cpu().numpy()
        #     image_all = image_all_000.copy()
        #     for m in range(pts.shape[0]):

        #         try:
        #             cv2.circle(image_all, (int(pts[m, 0]), int(pts[m, 1])), 1, (255, 255, 255), 2)
        #         except:
        #             pass
        #     alll.append(image_all)
        # cv2.imwrite("local_map.jpg", np.concatenate(alll,0))

        features_shape = torch.Tensor([self.input_size[0], self.input_size[1]]).to(points.device)
        points = self.normalize_coords(coords=points, shape=features_shape)
        depths = depths.view(B, N, Z, Y, X, 1).permute(0, 1, 2, 5, 4, 3).squeeze().view(B * N, Z, X, Y)
        points = points.view(B * N * Z, X, Y, 2)

        return points

    def projectPoints_fisheye(self, proj_points, dist_coeffs, ori_intrin):
        """
        :param data: (torch.tensor, shape=[N, cams, C, 3]) 3D points in camera coordinates.
        :param K: (torch.tensor, shape=[N, cams, 3, 3]) Camera matrix.
        :param dist_coeffs: (torch.tensor, shape=[N, cams, 1, 8]) Distortion coefficients.
        : 径向畸变系数(k1,k2),切向畸变系数(p1,p2),径向畸变系数(k3,k4,k5,k6)
        :return: (torch.tensor, shape=[N, 2]) Projected 2D points.
        """
        # Apply dist_coeffs
        import math
        r = torch.sqrt(torch.sum(proj_points[..., :2] ** 2, dim=-1, keepdim=True)).squeeze(-1)
        k1, k2, p1, p2, k3, k4, k5, k6 = dist_coeffs.unbind(-1)  # 
        t = torch.atan2(r, torch.ones_like(r))
        radial = t * (1 + k1 * t ** 2 + k2 * t ** 4 + k3 * t ** 6 + k4 * t ** 8) / r  # torch.Size([5, 1, 76800])

        proj_points[..., 0] = proj_points[..., 0] * radial
        proj_points[..., 1] = proj_points[..., 1] * radial

        # Apply camera matrix
        proj_points = ori_intrin.matmul(proj_points.unsqueeze(-1)).squeeze(-1)

        return proj_points

    def normalize_coords(self, coords, shape):
        """
        Normalize coordinates of a grid between [-1, 1]
        Args:
            coords [torch.Tensor(..., 2)]: Coordinates in grid
            shape [torch.Tensor(2)]: Grid shape [H, W]
        Returns:
            norm_coords [torch.Tensor(.., 2)]: Normalized coordinates in grid
        """
        min_n = -1
        max_n = 1
        shape = torch.flip(shape, dims=[0])  # Reverse ordering of shape 512,128

        # Subtract 1 since pixel indexing from [0, shape - 1]
        norm_coords = coords / (shape - 1) * (max_n - min_n) + min_n
        return norm_coords  # [-1,1]



class BEVBackbone(nn.Module):
    def __init__(self, channels=256,grid_conf=None, input_size=(128, 384),num_temporal=7,num_cams = 6):
        super(BEVBackbone, self).__init__()
        self.num_temporal = num_temporal
        self.grid_conf = grid_conf
        self.input_size = input_size
        self.splat = Splat(grid_conf, input_size)
        self.points = self.splat.create_voxels()
        ground_points = self.points[2]
        self.bev_emb = self.positional_encoding(ground_points[...,:2],4).permute(2,1,0)
        self.height = self.points.shape[0]
        self.num_cams = num_cams
        # self.bev_backbone = nn.Sequential(
        #     nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        #     nn.MaxPool2d(2, 2)
        # )
        self.fusion_height = nn.Sequential(
            nn.Conv2d(channels * self.height+self.bev_emb.shape[0], channels*3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels*3),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels*3, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.multiview_fusion = nn.Sequential(
            nn.Conv2d(channels * self.num_cams, channels*3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels*3),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels*3, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.algin_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        # nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rots, trans, intrins, distorts, post_rot3, post_tran3, theta_mats):

        """
        # B,5*6,3,128,384
        # B*5*6,3,128,384
        B, img_num 6,  c h w
        x_img.reshape(6, B, img_num, *x_img.shape[2:])
        """
        S = self.num_temporal
        # B, S, N, C, H, W = x.shape
        rots = rots.flatten(0, 1)
        trans = trans.flatten(0, 1)
        intrins = intrins.flatten(0, 1)
        distorts = distorts.flatten(0, 1)
        post_rot3 = post_rot3.flatten(0, 1)
        post_tran3 = post_tran3.flatten(0, 1)
        # points1 = torch.tensor(points1).view(seq_len * 3, 32, 96, 3).unsqueeze(1).unsqueeze(-1)
        # x = self.bev_backbone(x)
        points = self.points.to(x.device) 
        bev_emb = self.bev_emb.to(x.device)
        grid = self.splat.bev2eachroi(points, rots.to(points.dtype), trans.to(points.dtype), intrins.to(points.dtype),
                                      distorts.to(points.dtype), post_rot3.to(points.dtype),
                                      post_tran3.to(points.dtype)).to(x.dtype)
        B, X, Y, _ = grid.shape #B = bs=3 * mode=2 * seq_len=5 *3（相机数） * 6(高度)
        B = int(B / self.height)
        indexs = [[self.height * i + j for i in range(B)] for j in range(self.height)]
        outs = [F.grid_sample(input=x, grid=grid[index, ...], mode="nearest", padding_mode="zeros") for index in indexs]
        # index_0 = [6 * i + 0 for i in range(B)]
        # index_1 = [6 * i + 1 for i in range(B)]
        # index_2 = [6 * i + 2 for i in range(B)]
        # index_3 = [6 * i + 3 for i in range(B)]       theta_mats = 6 * 5 * 2 * 3
        # index_4 = [6 * i + 4 for i in range(B)]
        # index_5 = [6 * i + 5 for i in range(B)]     b m t n c h w -> (b m t n) c h w # 3 2 5 3
        # output_0 = F.grid_sample(input=x, grid=grid[index_0, ...], mode="nearest", padding_mode="zeros")
        # output_1 = F.grid_sample(input=x, grid=grid[index_1, ...], mode="nearest", padding_mode="zeros")
        # output_2 = F.grid_sample(input=x, grid=grid[index_2, ...], mode="nearest", padding_mode="zeros")
        # output_3 = F.grid_sample(input=x, grid=grid[index_3, ...], mode="nearest", padding_mode="zeros")
        # output_4 = F.grid_sample(input=x, grid=grid[index_4, ...], mode="nearest", padding_mode="zeros")
        # output_5 = F.grid_sample(input=x, grid=grid[index_5, ...], mode="nearest", padding_mode="zeros")
        out_put = torch.cat(outs,dim=1)  # B, C, H, W
        out_put = torch.cat([out_put, bev_emb[None].expand(out_put.shape[0], -1, -1, -1)], dim=1) # torch.Size([25, 384, 150, 100]) # 90 256 200 80 # 90 = 3 * 2 * 5 * 3(三个相机)
        out_put = self.fusion_height(out_put)
        out_put = out_put.reshape(-1,self.num_cams*out_put.shape[1],*out_put.shape[-2:])
        out_put = self.multiview_fusion(out_put)
        out_put = out_put.view(-1, S, *out_put.shape[-3:])
        algin_features = []
        for i in range(S):
            if i == 0:
                algin_features.append(out_put[:, i])
            else:
                algin_feature = self.warp_feature(out_put[:, i], theta_mats[:, i].to(out_put.dtype))
                algin_features.append(self.algin_fusion(torch.cat([algin_features[-1], algin_feature], 1)))
        algin_features = torch.stack(algin_features).permute(1, 0, 2, 3, 4).flatten(0, 1)
        return algin_features  # 输出时序帧信息，每一帧都和前一阵融合这样子达到了渐进的融合方式

    def warp_feature(self, features, theta):
        B, C, H, W = features.size()
        grids = F.affine_grid(theta, torch.Size((B, C, H, W)), align_corners=True)
        cropped_features = F.grid_sample(features, grids, align_corners=True)
        return cropped_features

    def positional_encoding(self, input, L):  # [B,...,N]
        shape = input.shape
        freq = 2 ** torch.arange(L, dtype=torch.float32, device=input.device) * np.pi  # [L]
        spectrum = input[..., None] * freq  # [B,...,N,L]
        sin, cos = spectrum.sin(), spectrum.cos()  # [B,...,N,L]
        input_enc = torch.stack([sin, cos], dim=-2)  # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1], -1)  # [B,...,2NL]
        return input_enc


    def get_cam_pos_embedding(self, intrins, post_rots, post_trans, distorts, L):
        with torch.no_grad():
            ogfH, ogfW = self.data_aug_conf['final_dim']
            fH, fW = ogfH // self.downsample, ogfW // self.downsample
            xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float, device=intrins.device).view(1, 1, fW).expand(1, fH,
                                                                                                                 fW)
            ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float, device=intrins.device).view(1, fH, 1).expand(1, fH,
                                                                                                                 fW)
            ds = torch.ones_like(xs)
            rays = torch.stack((xs, ys, ds), -1)

            # for distort_img using torch by ZY@[2023-04-19]
            if distorts.shape[0] != 0:
                img_size = (fW, fH)
                alpha = 1
                ori_K = intrins[0, 0, :, :].cpu().numpy()
                distort = distorts[0, 0, :].cpu().numpy()
                dis_img = torch.squeeze(rays).cpu().numpy()

                NewCameraMatrix, _ = cv2.getOptimalNewCameraMatrix(ori_K, distort, img_size, alpha, img_size, 0)
                map1, map2 = cv2.initUndistortRectifyMap(ori_K, distort, None, NewCameraMatrix, img_size, cv2.CV_32FC1)
                rays = cv2.remap(dis_img, map1, map2, cv2.INTER_LINEAR)
                rays = torch.tensor(rays, device=intrins.device)
                NewCameraMatrix = torch.tensor(NewCameraMatrix, device=intrins.device).expand(intrins.shape)
                rays = rays.unsqueeze(0)

            B, N, _ = post_trans.shape
            # undo post-transformation
            # B x N x D x H x W x 3
            points = rays - post_trans.view(B, N, 1, 1, 1, 3)
            points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

            if distorts.shape[0] != 0:
                points = torch.inverse(NewCameraMatrix).view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
            else:
                points = torch.inverse(intrins).view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)

            cam_pos_embedding = self.positional_encoding(points[..., :2], L).view(B * N, fH, fW, -1).permute(0, 3, 1, 2)
            # 最好再把原始的x,y也保留并concat起来
            return cam_pos_embedding


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    R = np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]
    ])

    return R


def TransformationmatrixEgo(orientation, position):
    w, x, y, z = orientation
    rotation_matrix = quaternion_to_rotation_matrix(x, y, z, w)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = position
    return transform


def egopose_alginhistory2current(ego_poses):
    T_ego_his2curs = []
    T_ego2wld_cur = TransformationmatrixEgo(ego_poses[-1]["orientation"], ego_poses[-1]["position"])
    for i, ego_pose in enumerate(ego_poses):
        T_ego2wld_his = TransformationmatrixEgo(ego_pose["orientation"], ego_pose["position"])
        pose_diff = np.linalg.inv(T_ego2wld_cur) @ T_ego2wld_his
        T_ego_his2curs.append(pose_diff)
    return np.array(T_ego_his2curs)


def gen_theta_mat(ego_poses):
    sx = 200 / 2
    sy = 80 / 2
    theta_mats = []
    for i, ego_pose in enumerate(ego_poses):
        T_ego2wld_cur = TransformationmatrixEgo(ego_pose["orientation"], ego_pose["position"])
        if i == 0:
            pose_diff = np.eye(4)
        else:
            pose_diff = np.linalg.inv(T_ego2wld_pre) @ T_ego2wld_cur
        T_ego2wld_pre = T_ego2wld_cur
        yaw = np.arctan2(pose_diff[1, 0], pose_diff[0, 0])
        dx = pose_diff[0, 3]
        dy = pose_diff[1, 3]
        cos = np.cos(yaw)
        sin = np.sin(yaw)
        eye = np.zeros((3, 3), dtype=np.float32)
        rel_pose = eye.copy()
        rel_pose[2, 2] = 1
        rel_pose[0, 0], rel_pose[0, 1], rel_pose[0, 2] = cos, -sin, dx
        rel_pose[1, 0], rel_pose[1, 1], rel_pose[1, 2] = sin, cos, dy
        pre_mat = np.array([[0., 1. / sy, 0.],
                            [1. / sx, 0., 1 / 5],
                            [0., 0., 1.]])

        post_mat = np.array([[0., sx, -sx / 5],
                             [sy, 0., 0.],
                             [0., 0., 1.]])
        theta_mat = (pre_mat @ (rel_pose @ post_mat))[:2, :][None]
        theta_mats.append(theta_mat)
    return np.concatenate(theta_mats, 0)
