"""Optimizers and LR schedulers helpers."""
from __future__ import annotations
from typing import Iterable, Tuple
import torch

def build_optimizer(params: Iterable, name: str = "adamw", lr: float = 1e-3, weight_decay: float = 0.0, betas: Tuple[float,float] = (0.9,0.999)):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas)
    raise ValueError(f"Unknown optimizer: {name}")

def build_scheduler(optimizer, name: str = "cosine", warmup_steps: int = 0, min_lr: float = 1e-6, max_steps: int = 100000):
    name = name.lower()
    if name == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            # cosine decay from 1.0 to min_lr / base_lr
            progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            import math
            return max(min_lr / optimizer.param_groups[0]["lr"], 0.5 * (1.0 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if name == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    raise ValueError(f"Unknown scheduler: {name}")
