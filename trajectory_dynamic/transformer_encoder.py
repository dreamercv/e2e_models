# Transformer that predicts VQ code indices per instance from Sparse4D-style features.
# Pair with TrajectoryVQVAE.decode_from_indices(pred_indices, instance_mask).
# Encoder stacks CausalSelfAttention (non-causal) + FFN from blocks/attention.py style.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention

__all__ = [
    'TrajectoryEncoder',
    'ce_loss',
    'EncoderBlock',
]


class _EncoderConfig:
    """Config bundle for CausalSelfAttention (causal=False → full bidirectional attention)."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        attn_pdrop: float,
        resid_pdrop: float,
        block_size: int,
    ):
        self.n_embd = n_embd
        self.n_head = n_head
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.block_size = block_size
        self.causal = False


class EncoderBlock(nn.Module):
    """
    Pre-LayerNorm block: x += Attn(LN(x)), x += FFN(LN(x)).
    Padding: pass valid_mask (B, T) True = real slot; keys at False are masked (see CausalSelfAttention).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        max_seq: int,
    ):
        super().__init__()
        cfg = _EncoderConfig(
            n_embd=d_model,
            n_head=nhead,
            attn_pdrop=dropout,
            resid_pdrop=dropout,
            block_size=max_seq,
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(cfg,d_model,nhead,False,dropout, dropout,max_seq)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h, _ = self.attn(self.ln1(x), valid_mask=valid_mask)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x


class TrajectoryEncoder(nn.Module):
    """
    Inputs:
        instance_feature: (B, N, instance_feat_dim)
        anchor: (B, N, anchor_dim)
    Output:
        logits: (B, N, nbooks, n_e_i)  — n_e_i = n_e // nbooks (matches VectorQuantizer per-book size)

    Use instance_mask=None or all-True when every slot is valid; else padding slots are excluded from attention.
    """

    def __init__(
        self,
        instance_num = 50,
        instance_feat_dim: int = 256,
        anchor_dim: int = 11,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % nhead == 0, 'd_model must be divisible by nhead'
        
        self.d_model = d_model

        self.instance_num = instance_num
        

        self.input_proj = nn.Linear(instance_feat_dim + anchor_dim, d_model)
        

        # --- Custom encoder (CausalSelfAttention, blocks/attention.py) ---
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    max_seq=instance_num,
                )
                for _ in range(num_layers)
            ]
        )
        

        self.out_norm = nn.LayerNorm(d_model)
        # self.code_heads = nn.ModuleList(
        #     [nn.Linear(d_model, self.n_e_i) for _ in range(nbooks)]
        # )

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        instance_mask: Optional[torch.Tensor] = None,
        slot_embed = None,
    ) -> torch.Tensor:
    

        x = torch.cat([instance_feature, anchor], dim=-1)
        x = self.input_proj(x)
        if slot_embed is not None:
            x = x + slot_embed

        # Custom encoder path: valid_mask True = real slot (key not masked)
        valid_mask: Optional[torch.Tensor] = None
        if instance_mask is not None:
            valid_mask = instance_mask.bool()

        for layer in self.layers:
            x = layer(x, valid_mask=valid_mask)
            

        x = self.out_norm(x)

        
        return x

    @torch.inference_mode()
    def predict_indices(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        instance_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self(instance_feature, anchor, instance_mask)
        pred = logits.argmax(dim=-1).long()
        if instance_mask is not None:
            m = instance_mask.bool().unsqueeze(-1).expand_as(pred)
            pred = pred.masked_fill(~m, -1)
        return pred


def ce_loss(
    logits: torch.Tensor,
    target_indices: torch.Tensor,
    instance_mask: torch.Tensor,
) -> torch.Tensor:
    """
    logits: (B, N, nbooks, n_e_i)
    target_indices: (B, N, nbooks) long; use -1 for positions to ignore (padding / no teacher)
    instance_mask: (B, N) bool or 0/1
    """
    if logits.shape[:3] != target_indices.shape:
        raise ValueError(f'logits {logits.shape} vs targets {target_indices.shape}')
    b, n, k, c = logits.shape
    inst = instance_mask.bool()
    total = logits.new_tensor(0.0)
    n_terms = 0
    for i in range(k):
        li = logits[:, :, i, :]
        ti = target_indices[:, :, i]
        valid = inst & (ti >= 0)
        if valid.any():
            total = total + F.cross_entropy(li[valid], ti[valid])
            n_terms += 1
    if n_terms == 0:
        return total
    return total / n_terms