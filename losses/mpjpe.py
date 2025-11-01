"""MPJPE and Procrustes-aligned MPJPE losses."""
from __future__ import annotations
import torch

def mpjpe(pred: torch.Tensor, gt: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    Mean Per Joint Position Error in millimeters (expects same units).
    pred, gt: [..., J, 3]
    """
    err = torch.linalg.norm(pred - gt, dim=-1)  # [..., J]
    if reduction == "none":
        return err
    if reduction == "sum":
        return err.sum()
    return err.mean()

def p_mpjpe(pred: torch.Tensor, gt: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    Procrustes-aligned MPJPE (rigid alignment). Minimal implementation.
    """
    # Center
    pred_c = pred - pred.mean(dim=-2, keepdim=True)
    gt_c = gt - gt.mean(dim=-2, keepdim=True)
    # Compute rotation via SVD
    H = pred_c.transpose(-1, -2) @ gt_c
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.transpose(-1, -2) @ U.transpose(-1, -2)
    # Ensure right-handed
    det = torch.linalg.det(R)
    mask = det < 0
    if mask.any():
        Vt[mask, :, -1] *= -1
        R = Vt.transpose(-1, -2) @ U.transpose(-1, -2)
    pred_aligned = (pred_c @ R)
    return mpjpe(pred_aligned, gt_c, reduction=reduction)
