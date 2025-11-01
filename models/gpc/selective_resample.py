"""Selective resampling utilities based on TGT uncertainty."""
from __future__ import annotations
import torch

def apply_selective_mask(x_current: torch.Tensor, x_new: torch.Tensor, mask_btj: torch.Tensor) -> torch.Tensor:
    """
    Keep confident joints from x_current; replace only masked joints with x_new.
    Args:
      x_current: [B,T,J,3] current pose
      x_new:     [B,T,J,3] new pose proposal
      mask_btj:  [B,T,J]   True => take from x_new
    """
    mask = mask_btj.unsqueeze(-1).type_as(x_current)
    return x_current * (1 - mask) + x_new * mask
