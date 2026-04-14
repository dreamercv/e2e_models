# Offline VQ-VAE for per-slot future trajectories (e.g. B x N x 100, 100 = 50 x xy).
# Uses VectorQuantizer from quantize.py; instance_mask (B, N) marks real vs padded slots.

from typing import Optional

import torch
import torch.nn as nn

from .quantize import VectorQuantizer

__all__ = ['VQVAE', 'Encoder', 'Decoder']


class Encoder(nn.Module):
    """(B, N, traj_dim) with traj_dim = 2 * num_future_steps -> (B, N, e_dim)."""

    def __init__(self, traj_dim: int = 100, e_dim: int = 128, hidden: int = 64):
        super().__init__()
        assert traj_dim % 2 == 0, 'traj_dim must be 2 * num_steps'
        self.num_steps = traj_dim // 2
        self.conv = nn.Sequential(
            nn.Conv1d(2, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.to_latent = nn.Linear(hidden, e_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, traj_dim)
        b, n, _ = x.shape
        h = x.view(b * n, self.num_steps, 2).transpose(1, 2)  # (B*N, 2, T)
        h = self.conv(h)
        h = h.mean(dim=-1)
        z = self.to_latent(h).view(b, n, -1)
        return z


class Decoder(nn.Module):
    """(B, N, e_dim) -> (B, N, traj_dim)."""

    def __init__(self, traj_dim: int = 100, e_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.traj_dim = traj_dim
        self.net = nn.Sequential(
            nn.Linear(e_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, traj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, n, e = z.shape
        return self.net(z.reshape(b * n, e)).view(b, n, self.traj_dim)


class VQVAE(nn.Module):
    """
    Args:
        num_slots: N (e.g. 100), must match input dim 1.
        traj_dim: last dim (e.g. 100 = 50 future steps * xy).
        e_dim: bottleneck dim for VectorQuantizer (must be divisible by nbooks).
        n_e, beta, nbooks: see VectorQuantizer.
    Forward:
        x: (B, N, traj_dim)
        traj_mask: (B, N, traj_dim), 1 = valid xy to reconstruct
        instance_mask: (B, N), 1 = real instance, 0 = slot padded to N
    """

    def __init__(
        self,
        num_slots: int = 100, #槽位数量，理解应该要大于anchor的数量即可
        traj_dim: int = 100,#50*2
        e_dim: int = 128, #encoder输出维度，每个条目的特征维度
        n_e: int = 512,#码本总条目，可以理解是kmeans的类别数
        beta: float = 0.25,
        nbooks: int = 1,#码本数量
        enc_hidden: int = 64,
        dec_hidden: int = 256,
    ):
        super().__init__()
        assert e_dim % nbooks == 0, 'e_dim must be divisible by nbooks'
        assert n_e % nbooks == 0, 'n_e must be divisible by nbooks'
        self.num_slots = num_slots
        self.traj_dim = traj_dim
        self.e_dim = e_dim
        self.nbooks = nbooks
        self.n_e = n_e

        self.encoder = Encoder(traj_dim=traj_dim, e_dim=e_dim, hidden=enc_hidden)
        self.decoder = Decoder(traj_dim=traj_dim, e_dim=e_dim, hidden=dec_hidden)
        self.quantizer = VectorQuantizer(n_e=n_e, e_dim=e_dim, beta=beta, nbooks=nbooks)

    def _mask_input(
        self,
        x: torch.Tensor,
        traj_mask: Optional[torch.Tensor],
        instance_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        w = torch.ones_like(x)
        if traj_mask is not None:
            w = w * traj_mask.to(dtype=x.dtype)
        if instance_mask is not None:
            w = w * instance_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x * w

    def encode(
        self,
        x: torch.Tensor,
        traj_mask: Optional[torch.Tensor] = None,
        instance_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_in = self._mask_input(x, traj_mask, instance_mask)
        return self.encoder(x_in)

    @staticmethod
    def _apply_instance_to_vq(
        z_q: torch.Tensor,
        vq_loss: torch.Tensor,
        indices: torch.Tensor,
        instance_mask: Optional[torch.Tensor],
    ):
        if instance_mask is None:
            return z_q, vq_loss.mean(), indices
        valid = instance_mask.to(dtype=vq_loss.dtype)
        denom = valid.sum().clamp(min=1.0)
        vq_reduced = (vq_loss * valid).sum() / denom
        vb = instance_mask.bool().unsqueeze(-1).expand_as(indices)
        indices_out = torch.where(vb, indices, torch.full_like(indices, -1))
        z_q_out = z_q * instance_mask.unsqueeze(-1).to(dtype=z_q.dtype)
        return z_q_out, vq_reduced, indices_out

    def forward(
        self,
        x: torch.Tensor,
        traj_mask: torch.Tensor,
        instance_mask: torch.Tensor,
        quant_prop: float = 1.0,
    ):
        """
        Returns dict with x_hat, recon_loss, vq_loss (scalar), indices (B,N,nbooks),
        z (encoder out), z_q (after VQ + instance mask).
        """
        z = self.encode(x, traj_mask, instance_mask)
        z = z * instance_mask.unsqueeze(-1).to(dtype=z.dtype)
        z_q, vq_loss, indices = self.quantizer(z, p=quant_prop)
        z_q, vq_reduced, indices = self._apply_instance_to_vq(z_q, vq_loss, indices, instance_mask)
        x_hat = self.decoder(z_q)

        recon_w = instance_mask.unsqueeze(-1).to(dtype=x.dtype) * traj_mask.to(dtype=x.dtype)
        diff = (x_hat - x) ** 2
        recon_loss = (diff * recon_w).sum() / recon_w.sum().clamp(min=1.0)

        return {
            'x_hat': x_hat,
            'recon_loss': recon_loss,
            'vq_loss': vq_reduced,
            'indices': indices,
            'z': z,
            'z_q': z_q,
        }

    def decode_from_indices(self, indices: torch.Tensor, instance_mask: Optional[torch.Tensor] = None):
        """
        indices: (B, N, nbooks) long; use -1 only if you will mask via instance_mask.
        Embedding does not accept -1; invalid slots are looked up with clamp then zeroed.
        """
        idx = indices.clamp(min=0)
        z_q = self.quantizer.get_codebook_entry(idx)
        if instance_mask is not None:
            z_q = z_q * instance_mask.unsqueeze(-1).to(dtype=z_q.dtype)
        return self.decoder(z_q)