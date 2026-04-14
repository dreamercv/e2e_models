# -*- encoding: utf-8 -*-
'''
@File         :test_vqvae.py
@Date         :2026/04/09 14:26:29
@Author       :Binge.Van
@E-mail       :afb5szh@bosch.com
@Version      :V1.0.0
@Description  :

'''


import os,sys
import torch
from torch import nn
from trajectory_dynamic.trajectory_predictor import build_dynamic_trajectory_predictor

def main():
    pass



if __name__ == '__main__':
    device = torch.device('cuda')

    B, N = 4, 55 ###
    traj_dim = 110 ###
    instance_feat_dim = 256####
    anchor_dim = 11###

    nbooks = 1

    traj = build_dynamic_trajectory_predictor(
        obj_num = N,
        instance_feat_dim = instance_feat_dim,
        anchor_dim = anchor_dim,
        d_model = 256,
        nhead = 4,
        num_layers = 3,
        traj_dim = traj_dim,
        vqvae_num_slots = 1024,
        e_dim = 256,
        n_e = 512,
        nbooks = nbooks,
        freeze_vqvae = False,
    ).to(device)
    
    
    instance_feature = torch.randn(B, N, instance_feat_dim, device=device, requires_grad=True)
    anchor = torch.randn(B, N, anchor_dim, device=device, requires_grad=True)
    gt_traj = torch.randn(B, N, traj_dim, device=device)
    traj_mask = torch.ones(B, N, traj_dim, device=device)
    instance_mask = torch.ones(B, N, device=device)
    instance_mask[:, 80:] = 0

    out = traj(
        instance_feature,
        anchor,
        instance_mask,
        gt_traj=gt_traj,
        traj_mask=traj_mask,
        decode_trajectory=True,
    )

    assert out['loss'] is not None
    out['loss'].backward()

    assert out['decoded_traj'] is not None
    assert out['decoded_traj'].shape == (B, N, traj_dim)
    assert out['pred_indices'].shape == (B, N, nbooks)
    print('loss', float(out['loss']))
    print('decoded_traj', tuple(out['decoded_traj'].shape))
    print('pred_indices', tuple(out['pred_indices'].shape))