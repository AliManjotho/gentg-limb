"""Temporal batching utilities: windows, padding, masks, and a collate_fn.

These helpers make it easy to:
- Slice sliding windows from long sequences
- Pad variable-length windows to a common size for batching
- Build padding masks for attention-based models
- Use a DataLoader collate_fn that stacks dict samples safely
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import math
import torch

TensorDict = Dict[str, torch.Tensor]

def sliding_window_indices(T: int, window: int, stride: int) -> List[Tuple[int, int]]:
    """Return list of (start, end) indices for sliding windows within [0, T)."""
    if T <= 0 or window <= 0:
        return []
    if T <= window:
        return [(0, T)]
    starts = list(range(0, T - window + 1, max(1, stride)))
    return [(s, s + window) for s in starts]

def pad_to_length(x: torch.Tensor, target_len: int, dim: int = 0, pad_value: float = 0.0) -> torch.Tensor:
    """Pad tensor along dimension dim to target_len with pad_value (right padding)."""
    diff = target_len - x.shape[dim]
    if diff <= 0:
        return x
    pad_shape = list(x.shape)
    pad_shape[dim] = diff
    pad = torch.full(pad_shape, pad_value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)

def make_padding_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Create boolean mask [B, T] where True marks padded (invalid) positions."""
    B = lengths.shape[0]
    T = int(max_len or int(lengths.max().item()))
    rng = torch.arange(T, device=lengths.device).unsqueeze(0).expand(B, T)
    return rng >= lengths.unsqueeze(1)

def collate_temporal(batch: List[TensorDict], pad_value: float = 0.0) -> TensorDict:
    """Collate a list of samples with shapes:
         k2d:  [T_i, J, 2]
         k3d:  [T_i, J, 3] (optional)
         conf: [T_i, J]    (optional)
       Returns a batch dict with padded tensors:
         k2d:  [B, T_max, J, 2]
         k3d:  [B, T_max, J, 3] (if present)
         conf: [B, T_max, J]    (if present)
         pad_mask: [B, T_max]   (True at padded positions)
    """
    keys = set().union(*[d.keys() for d in batch])
    B = len(batch)
    lengths = torch.tensor([next(iter(d.values())).shape[0] for d in batch], dtype=torch.long)
    T_max = int(lengths.max().item())

    out: TensorDict = {}
    for k in keys:
        tensors = [d[k] for d in batch if k in d]
        # infer trailing shape
        trailing = tensors[0].shape[1:]
        stacked = []
        for t in tensors:
            t_pad = pad_to_length(t, T_max, dim=0, pad_value=pad_value)
            stacked.append(t_pad)
        out[k] = torch.stack(stacked, dim=0)  # [B, T_max, ...]
    out["pad_mask"] = make_padding_mask(lengths, max_len=T_max)
    return out
