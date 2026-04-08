# Full pipeline: TrajectoryIndexTransformer -> VQ indices -> TrajectoryVQVAE.decode (100 slots).

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .trajectory_index_transformer import (
    TrajectoryIndexTransformer,
    trajectory_vq_index_ce_loss,
)
from .trajectory_vqvae import TrajectoryVQVAE

__all__ = [
    'TrajectoryVQPredictionPipeline',
    'build_trajectory_vq_pipeline_100',
]


class TrajectoryVQPredictionPipeline(nn.Module):
    """
    训练：用冻结 VQ-VAE 对 GT 轨迹得到 teacher indices，对 index transformer 做 CE；
    同时用 argmax 预测 indices 走 decode 得到预测未来轨迹（不参与反向，或仅供日志）。

    槽位 N 默认 100，需与已训好的 TrajectoryVQVAE.num_slots、Sparse4D 目标数一致。
    """

    def __init__(
        self,
        index_transformer: TrajectoryIndexTransformer,
        vqvae: TrajectoryVQVAE,
        freeze_vqvae: bool = True,
    ):
        super().__init__()
        self.index_transformer = index_transformer
        self.vqvae = vqvae
        self.freeze_vqvae = freeze_vqvae
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

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        instance_mask: torch.Tensor,
        gt_traj: Optional[torch.Tensor] = None,
        traj_mask: Optional[torch.Tensor] = None,
        teacher_indices: Optional[torch.Tensor] = None,
        decode_trajectory: bool = True,
    ) -> Dict[str, Any]:
        """
        instance_feature: (B, N, 256)
        anchor: (B, N, 11)
        instance_mask: (B, N)
        gt_traj / traj_mask: 若提供且 teacher_indices 为 None，则用冻结 VQ-VAE 生成教师 indices。
        teacher_indices: (B, N, nbooks)，可直接监督（无效槽位为 -1）。
        """
        logits = self.index_transformer(instance_feature, anchor, instance_mask)
        pred_indices = logits.argmax(dim=-1).long()
        vb = instance_mask.bool().unsqueeze(-1).expand_as(pred_indices)
        pred_indices = pred_indices.masked_fill(~vb, -1)

        loss: Optional[torch.Tensor] = None
        if teacher_indices is None and gt_traj is not None and traj_mask is not None:
            teacher_indices = self.teacher_indices_from_trajectory(
                gt_traj, traj_mask, instance_mask
            )
        if teacher_indices is not None:
            loss = trajectory_vq_index_ce_loss(logits, teacher_indices, instance_mask)

        decoded: Optional[torch.Tensor] = None
        if decode_trajectory:
            if self.freeze_vqvae:
                with torch.no_grad():
                    decoded = self.vqvae.decode_from_indices(pred_indices, instance_mask)
            else:
                decoded = self.vqvae.decode_from_indices(pred_indices, instance_mask)

        return {
            'loss': loss,
            'logits': logits,
            'pred_indices': pred_indices,
            'teacher_indices': teacher_indices,
            'decoded_traj': decoded,
        }


def build_trajectory_vq_pipeline_100(
    instance_feat_dim: int = 256,
    anchor_dim: int = 11,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 3,
    traj_dim: int = 100,
    e_dim: int = 128,
    n_e: int = 512,
    nbooks: int = 1,
    freeze_vqvae: bool = True,
) -> TrajectoryVQPredictionPipeline:
    """默认 100 槽位，超参与已训 VQ-VAE 检查点需一致。"""
    vqvae = TrajectoryVQVAE(
        num_slots=100,
        traj_dim=traj_dim,
        e_dim=e_dim,
        n_e=n_e,
        nbooks=nbooks,
    )
    index_net = TrajectoryIndexTransformer(
        instance_feat_dim=instance_feat_dim,
        anchor_dim=anchor_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        n_e=n_e,
        nbooks=nbooks,
        max_slots=100,
    )
    return TrajectoryVQPredictionPipeline(index_net, vqvae, freeze_vqvae=freeze_vqvae)


def _demo_one_step():
    """一次 forward：算 CE loss，argmax indices 送入 VQ-VAE decoder 解压轨迹。需要 CUDA（quantize.EmaCodebookMeter 依赖 GPU）。"""
    if not torch.cuda.is_available():
        print('demo 跳过: 需要 CUDA（VectorQuantizer 内 EmaCodebookMeter 使用 .cuda()）')
        return

    device = torch.device('cuda')
    B, N = 4, 100
    traj_dim = 100
    e_dim, n_e, nbooks = 128, 512, 1

    vqvae = TrajectoryVQVAE(
        num_slots=N,
        traj_dim=traj_dim,
        e_dim=e_dim,
        n_e=n_e,
        nbooks=nbooks,
    ).to(device)

    index_net = TrajectoryIndexTransformer(
        instance_feat_dim=256,
        anchor_dim=11,
        d_model=256,
        nhead=8,
        num_layers=3,
        n_e=n_e,
        nbooks=nbooks,
        max_slots=N,
    ).to(device)

    pipeline = TrajectoryVQPredictionPipeline(index_net, vqvae, freeze_vqvae=True).to(device)

    instance_feature = torch.randn(B, N, 256, device=device, requires_grad=True)
    anchor = torch.randn(B, N, 11, device=device, requires_grad=True)
    gt_traj = torch.randn(B, N, traj_dim, device=device)
    traj_mask = torch.ones(B, N, traj_dim, device=device)
    instance_mask = torch.ones(B, N, device=device)
    instance_mask[:, 80:] = 0

    out = pipeline(
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


if __name__ == '__main__':
    _demo_one_step()
