"""Conditioning utilities for GPC (TGT outputs + 2D keypoints + masks)."""
from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class Conditioning:
    """
    Container for conditioning tensors.
    """
    k2d: torch.Tensor          # [B,T,J,2]
    tgt_xyz: torch.Tensor      # [B,T,J,3] coarse 3D from TGT
    sigma: torch.Tensor        # [B,T,J]   uncertainty from TGT
    conf: torch.Tensor | None  # [B,T,J]   optional 2D confidences

def build_conditioning(k2d: torch.Tensor, tgt_xyz: torch.Tensor, sigma: torch.Tensor, conf: torch.Tensor | None = None) -> Conditioning:
    return Conditioning(k2d=k2d, tgt_xyz=tgt_xyz, sigma=sigma, conf=conf)

def make_resample_mask(sigma: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Return boolean mask of shape [B,T,J] where True => resample via diffusion.
    """
    return (sigma >= threshold)
