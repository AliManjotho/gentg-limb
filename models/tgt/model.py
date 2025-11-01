"""Top-level TGT model assembling embeddings, encoder, and heads."""
from __future__ import annotations
from typing import Dict, List, Tuple
import torch
import torch.nn as nn

from ..common.embeddings import JointEmbedding, LimbEmbedding
from ..common.graph_utils import build_limb_index, joints_to_limbs
from .blocks import TGTEncoder
from .head import PoseHead, UncertaintyHead

class TGT(nn.Module):
    def __init__(self, joints: int = 17, limbs: List[Tuple[int,int]] | None = None,
                 d_model: int = 256, n_heads: int = 8, n_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        if limbs is None:
            # simple chain if none provided
            limbs = [(i, i+1) for i in range(joints-1)]
        self.limbs = limbs
        self.limb_index = None  # lazily built on first forward

        self.joint_embed = JointEmbedding(in_dim=2, d_model=d_model)  # 2D input embedding
        self.limb_embed  = LimbEmbedding(d_model=d_model)

        self.encoder = TGTEncoder(d_model=d_model, n_heads=n_heads, n_layers=n_layers, p_drop=dropout)
        self.pose_head = PoseHead(d_model=d_model, joints=joints)
        self.uncert_head = UncertaintyHead(d_model=d_model, joints=joints)

    def forward(self, k2d: torch.Tensor, coarse_xyz: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Inputs:
            k2d:         [B,T,J,2]     2D detections
            coarse_xyz:  [B,T,J,3]     (optionally zeros at start) for limb tokenization
        Returns:
            dict with:
              xyz:   [B,T,J,3]
              sigma: [B,T,J]
        """
        B, T, J, _ = k2d.shape
        device = k2d.device
        if self.limb_index is None:
            self.limb_index = build_limb_index(self.limbs, device=device)  # [2,L]

        # Embeddings
        jtoks = self.joint_embed(k2d)                            # [B,T,J,D]
        limb_vecs = joints_to_limbs(coarse_xyz, self.limb_index) # [B,T,L,3]
        ltoks = self.limb_embed(limb_vecs, self.limbs)           # [B,T,L,D]

        # Encode
        ctx = self.encoder(jtoks, ltoks)                         # [B,T,J+L,D]
        # Split joint tokens (first J positions)
        joint_ctx = ctx[:, :, :J, :]

        xyz = self.pose_head(joint_ctx)
        sigma = self.uncert_head(joint_ctx)
        return {"xyz": xyz, "sigma": sigma}
