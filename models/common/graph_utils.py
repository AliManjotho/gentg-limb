"""Hybrid graph utilities: joint and limb edge construction."""
from __future__ import annotations
from typing import List, Tuple
import torch

def build_limb_index(limbs: List[Tuple[int,int]], device=None) -> torch.Tensor:
    """
    Returns a [2, L] tensor of (parent_indices; child_indices) for gathering.
    """
    parents = torch.tensor([a for a,b in limbs], dtype=torch.long, device=device)
    children = torch.tensor([b for a,b in limbs], dtype=torch.long, device=device)
    return torch.stack([parents, children], dim=0)

def joints_to_limbs(xyz: torch.Tensor, limb_index: torch.Tensor) -> torch.Tensor:
    """
    xyz: [B,T,J,3], limb_index: [2,L]
    return limb vectors: [B,T,L,3] = child - parent
    """
    parents, children = limb_index[0], limb_index[1]
    v = xyz[..., children, :] - xyz[..., parents, :]
    return v
