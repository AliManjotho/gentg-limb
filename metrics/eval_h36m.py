"""Protocol-specific evaluation helpers for Human3.6M (minimal)."""
from __future__ import annotations
from typing import Dict
import torch
from .pck_auc import pck_3d, auc_3d
from losses.mpjpe import mpjpe, p_mpjpe

def evaluate_h36m(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    """Return a dict of common metrics on Human3.6M-style data."""
    return {
        'mpjpe': float(mpjpe(pred, gt).detach().cpu()),
        'p_mpjpe': float(p_mpjpe(pred, gt).detach().cpu()),
        'pck150': float(pck_3d(pred, gt, 150.0).detach().cpu()),
        'auc': float(auc_3d(pred, gt, 150.0).detach().cpu()),
    }
