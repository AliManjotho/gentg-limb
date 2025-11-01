"""Diffusion training objective wrapper (MSE between eps_hat and eps)."""
from __future__ import annotations
import torch

def diffusion_mse(eps_hat: torch.Tensor, eps: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    loss = (eps_hat - eps) ** 2
    if reduction == 'sum':
        return loss.sum()
    if reduction == 'none':
        return loss
    return loss.mean()
