"""3DPCK and AUC metrics in millimeters."""
from __future__ import annotations
import torch
import torch.nn.functional as F

def pck_3d(pred: torch.Tensor, gt: torch.Tensor, thresh_mm: float = 150.0) -> torch.Tensor:
    """
    pred, gt: [..., J, 3]
    Returns: average percentage of joints within thresh_mm.
    """
    dist = torch.linalg.norm(pred - gt, dim=-1)  # [..., J]
    correct = (dist <= thresh_mm).float()
    return correct.mean()

def auc_3d(pred: torch.Tensor, gt: torch.Tensor, max_mm: float = 150.0, steps: int = 31) -> torch.Tensor:
    """
    Area under the PCK curve from 0..max_mm.
    """
    thresholds = torch.linspace(0, max_mm, steps=steps, device=pred.device)
    pcks = []
    for t in thresholds:
        pcks.append(pck_3d(pred, gt, t))
    # Trapezoidal rule
    auc = torch.trapz(torch.stack(pcks), thresholds) / max_mm
    return auc
