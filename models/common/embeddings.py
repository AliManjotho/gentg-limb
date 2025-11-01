"""Token embeddings for joints and limbs (edge vectors)."""
from __future__ import annotations
import torch
import torch.nn as nn

class JointEmbedding(nn.Module):
    """
    Projects 2D keypoints (and optionally confidences) into d_model tokens.
    Input shape: [B, T, J, C] where C=2 or 3 (x,y[,conf])
    Output: [B, T, J, d_model]
    """
    def __init__(self, in_dim: int = 2, d_model: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

    def forward(self, k2d: torch.Tensor) -> torch.Tensor:
        return self.proj(k2d)


class LimbEmbedding(nn.Module):
    """
    Encodes limb vectors as (unit direction, length) then projects to d_model.
    Inputs:
      - xyz: [B, T, J, 3]  (coarse 3D or lifted features)
      - limbs: List of (parent, child) indices
    Output:
      - tokens: [B, T, L, d_model]
    """
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.dir_proj = nn.Linear(3, d_model // 2)
        self.len_proj = nn.Linear(1, d_model - d_model // 2)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, xyz: torch.Tensor, limbs: list[tuple[int, int]]) -> torch.Tensor:
        B, T, J, _ = xyz.shape
        device = xyz.device
        L = len(limbs)
        parent = torch.tensor([a for a,b in limbs], device=device, dtype=torch.long)
        child  = torch.tensor([b for a,b in limbs], device=device, dtype=torch.long)
        v = xyz[..., child, :] - xyz[..., parent, :]   # [B,T,L,3]
        length = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-6)
        unit = v / length
        tok = torch.cat([self.dir_proj(unit), self.len_proj(length)], dim=-1)
        return self.out_norm(tok)
