"""Limb-length temporal smoothness loss."""
from __future__ import annotations
from typing import List, Tuple
import torch

def limb_length(xyz: torch.Tensor, limbs: List[Tuple[int,int]]) -> torch.Tensor:
    """
    xyz: [B,T,J,3]
    return: [B,T,L]
    """
    B,T,J,_ = xyz.shape
    device = xyz.device
    parents = torch.tensor([a for a,b in limbs], device=device, dtype=torch.long)
    children = torch.tensor([b for a,b in limbs], device=device, dtype=torch.long)
    v = xyz[..., children, :] - xyz[..., parents, :]
    return torch.linalg.norm(v, dim=-1)

def temporal_smooth_l2(xyz: torch.Tensor, limbs: List[Tuple[int,int]]) -> torch.Tensor:
    """Penalize frame-to-frame change in limb lengths."""
    L_t = limb_length(xyz, limbs)  # [B,T,L]
    dL = L_t[:, 1:, :] - L_t[:, :-1, :]
    return (dL ** 2).mean()
