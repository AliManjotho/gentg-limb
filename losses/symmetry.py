"""Left–right symmetry loss for paired limbs."""
from __future__ import annotations
from typing import List, Tuple
import torch

def limb_length(xyz: torch.Tensor, limb: Tuple[int,int]) -> torch.Tensor:
    a,b = limb
    v = xyz[..., b, :] - xyz[..., a, :]
    return torch.linalg.norm(v, dim=-1)  # [...]
    
def symmetry_loss(xyz: torch.Tensor, limb_pairs: List[Tuple[Tuple[int,int], Tuple[int,int]]]) -> torch.Tensor:
    """
    Args:
      xyz: [B,T,J,3]
      limb_pairs: list of ((a1,b1),(a2,b2)) tuples corresponding to left/right limbs
    """
    losses = []
    for (l1, l2) in limb_pairs:
        L1 = limb_length(xyz, l1)
        L2 = limb_length(xyz, l2)
        losses.append((L1 - L2) ** 2)
    return torch.mean(torch.stack(losses, dim=0))
