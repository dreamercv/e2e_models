# -*- encoding: utf-8 -*-
'''
@File         :trajectory_predictor.py
@Date         :2026/04/09 14:55:57
@Author       :Binge.Van
@E-mail       :afb5szh@bosch.com
@Version      :V1.0.0
@Description  :

'''


import os,sys

import torch
from torch import nn

from .vqvae import VQVAE
from .transformer_encoder import TrajectoryEncoder,ce_loss

from sparse4d.dn_sampler import DenoisingSampler as Sampler

class TrajectoryPredictor(nn.Module):
    """
    训练：用冻结 VQ-VAE 对 GT 轨迹得到 teacher indices，对 index transformer 做 CE；
    同时用 argmax 预测 indices 走 decode 得到预测未来轨迹（不参与反向，或仅供日志）。

    槽位 N 默认 100，需与已训好的 TrajectoryVQVAE.num_slots、Sparse4D 目标数一致。
    """

    def __init__(
        self,
        traj_encoder,
        vqvae,
        sampler=None,
        freeze_vqvae: bool = True,
        use_slot_embedding: bool = True,
        gt_cls_key = None,
        gt_reg_key = None,
        gt_reg_key_mask = None


    ):
        super().__init__()

        
        

        self.traj_encoder = traj_encoder
        self.vqvae = vqvae
        self.sampler = sampler
        self.freeze_vqvae = freeze_vqvae

        self.n_e = self.vqvae.n_e
        self.nbooks = self.vqvae.nbooks
        self.n_e_i = self.n_e // self.nbooks
        self.max_slots = self.vqvae.num_slots
        self.d_model = self.traj_encoder.d_model
        self.instance_num = self.traj_encoder.instance_num

        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.gt_reg_key_mask = gt_reg_key_mask
        

        
        self.code_heads = nn.ModuleList(
            [nn.Linear(self.d_model, self.n_e_i) for _ in range(self.nbooks)]
        )

        self.use_slot_embedding = use_slot_embedding
        if use_slot_embedding:
            self.slot_embed = nn.Embedding(self.max_slots, self.d_model)
            self.slot_ids = torch.arange(self.instance_num, dtype=torch.long)

        if freeze_vqvae:
            self.vqvae.requires_grad_(False)
            self.vqvae.eval()

    @property
    def num_slots(self) -> int:
        return self.vqvae.num_slots

    @torch.inference_mode()
    def teacher_indices_from_trajectory(
        self,
        gt_traj: torch.Tensor,
        traj_mask: torch.Tensor,
        instance_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.freeze_vqvae:
            self.vqvae.eval()
        out = self.vqvae(gt_traj, traj_mask, instance_mask)
        return out['indices']


    # def loss(self,):
    #     # vq loss/ vqvae 重建loss/ 分类loss / transformer 重建loss


    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        # instance_mask: torch.Tensor,
        # gt_traj  = None,
        # traj_mask  = None,
        teacher_indices = None,
        decode_trajectory: bool = True,
        det_out = None,
        data_gt = None
    ) :
        """
        instance_feature: (B, N, 256)
        anchor: (B, N, 11)
        instance_mask: (B, N)
        gt_traj / traj_mask: 若提供且 teacher_indices 为 None，则用冻结 VQ-VAE 生成教师 indices。
        teacher_indices: (B, N, nbooks)，可直接监督（无效槽位为 -1）。
        """
        gt_traj,traj_mask,instance_mask = self.get_traj_target(det_out,data_gt)
        b, n = instance_feature.shape[:2]
        slot_ids =self.slot_ids.unsqueeze(0).expand(b, n).to(instance_feature.device)
        slot_embed = self.slot_embed(slot_ids)


        vqvae_out = self.vqvae(gt_traj, traj_mask, instance_mask)
        # vqvae_out  ={f"vqvae_{k}":v for k,v in vqvae_out.items()}



        x = self.traj_encoder(instance_feature, anchor, instance_mask,slot_embed)
        logits = torch.stack([h(x) for h in self.code_heads], dim=2)
        pred_indices = logits.argmax(dim=-1).long()
        vb = instance_mask.bool().unsqueeze(-1).expand_as(pred_indices)
        pred_indices = pred_indices.masked_fill(~vb, -1)

        decoded = self.vqvae.decode_from_indices(pred_indices, instance_mask)

        cls_loss = ce_loss(logits, vqvae_out["indices"], instance_mask)
        traj_loss = (((decoded - gt_traj)**2) * traj_mask).sum() / (traj_mask.sum() + 1e-6)

        dyn_traj_loss = {
            'traj_loss': traj_loss,
            'cls_loss': cls_loss,
            'recon_loss': vqvae_out["recon_loss"],
            'vq_loss': vqvae_out["vq_loss"],
        }
        dyn_traj = {
            
            'recon_traj': vqvae_out["x_hat"],
            'decoded_traj': decoded,
        }
        return dyn_traj_loss,dyn_traj
        
    @staticmethod
    def _flatten_gt_list(gt_list):
        """
        将 data 层输出的形如 [ [t0帧GT, t1帧GT, ...],  [...], ... ] 的二级 list
        展平为 Sparse4D 期望的一维 list（长度 = B*T）。
        若本身已是一维 list，则直接返回。
        """
        if not isinstance(gt_list, list) or len(gt_list) == 0:
            return gt_list
        first = gt_list[0]
        if isinstance(first, list):
            flat = []
            for seq in gt_list:
                flat.extend(seq)
            return flat
        return gt_list

    def get_traj_target(self,det_out,data):
        cls = det_out["classification"][-1] # 2 * 10 * 100 * 6
        reg = det_out["prediction"][-1] #2*10*100*11
        gt_cls = data.get(self.gt_cls_key)
        gt_reg = data.get(self.gt_reg_key)
        gt_reg_mask = data.get(self.gt_reg_key_mask, None)

        gt_cls = self._flatten_gt_list(gt_cls)
        gt_reg = self._flatten_gt_list(gt_reg)
        if gt_reg_mask is not None:
            gt_reg_mask = self._flatten_gt_list(gt_reg_mask) # b*t n 10

        _, _, _,traj_target = self.sampler.sample(
            cls, reg, gt_cls, gt_reg,reg_masks=gt_reg_mask
        )
        device = cls.device
        traj_target = traj_target.to(device)
        traj_mask =  traj_target.new_zeros(traj_target.shape) 
        traj_mask[traj_target<100] = 1
        instance_mask = (traj_target < 50).any(dim=-1).to(traj_target.dtype)
        return traj_target,traj_mask,instance_mask



def build_dynamic_trajectory_predictor(
    obj_num = 50,
    vqvae_num_slots = 1024,
    instance_feat_dim: int = 256,
    anchor_dim: int = 11,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 3,
    traj_dim: int = 100,
    e_dim: int = 256,
    n_e: int = 512,
    nbooks: int = 1,
    freeze_vqvae: bool = False,
):
    """默认 100 槽位，超参与已训 VQ-VAE 检查点需一致。"""
    vqvae = VQVAE(
        num_slots=vqvae_num_slots,
        traj_dim=traj_dim,
        e_dim=e_dim,
        n_e=n_e,
        nbooks=nbooks,
    )
    traj_encoder = TrajectoryEncoder(
        instance_num = obj_num,
        instance_feat_dim=instance_feat_dim,
        anchor_dim=anchor_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
    )
    sampler = Sampler()
    return TrajectoryPredictor(traj_encoder,
                                vqvae, 
                                sampler=sampler,
                                freeze_vqvae=freeze_vqvae,
                                gt_cls_key = "gt_labels_det3D",
                                gt_reg_key = "gt_bboxes_det3D",
                                gt_reg_key_mask = "gt_bboxes_det3D_mask"
                            )


if __name__ == '__main__':
    main()