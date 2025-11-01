"""Data transforms and temporal windowing."""
from __future__ import annotations
from typing import Dict, Tuple
import torch

def normalize_2d(k2d: torch.Tensor, img_wh: Tuple[int,int] = (1000, 1000)) -> torch.Tensor:
    """Normalize pixel 2D coords to roughly [-1, 1] range (simple heuristic)."""
    W, H = img_wh
    k2d_norm = (k2d - torch.tensor([W/2, H/2], device=k2d.device)) / torch.tensor([W/2, H/2], device=k2d.device)
    return k2d_norm

def center_root(k3d: torch.Tensor, root_index: int = 0) -> torch.Tensor:
    """Root-center 3D joints by subtracting the pelvis (or specified root)."""
    root = k3d[..., root_index:root_index+1, :]
    return k3d - root

def temporal_slice(x: torch.Tensor, start: int, window: int) -> torch.Tensor:
    """Slice temporal window [start, start+window)."""
    return x[:, start:start+window, ...]

def make_batch(sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Placeholder for collate logic when not using a custom collate_fn."""
    return sample
