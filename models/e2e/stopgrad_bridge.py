"""Stop-gradient utilities and feature bridges for end-to-end (TGT→GPC)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn

@dataclass
class BridgeConfig:
    stop_gradient: bool = True          # if True, block grads from GPC into TGT
    adapt_cond: bool = True             # if True, learn an adapter over conditioning channels
    cond_dim_in: int = 0                # expected input cond dimension per time step
    cond_dim_out: int = 0               # projected cond dimension for GPC

class StopGradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        return x.detach()
    @staticmethod
    def backward(ctx, g):
        return torch.zeros_like(g)

def stopgrad(x: torch.Tensor) -> torch.Tensor:
    """Detach with explicit autograd barrier."""
    return StopGradFn.apply(x)

class FeatureBridge(nn.Module):
    """Optional adapter for conditioning from TGT outputs to GPC input."""
    def __init__(self, cfg: BridgeConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.adapt_cond and cfg.cond_dim_in and cfg.cond_dim_out and cfg.cond_dim_out != cfg.cond_dim_in:
            self.adapter = nn.Sequential(
                nn.LayerNorm(cfg.cond_dim_in),
                nn.Linear(cfg.cond_dim_in, cfg.cond_dim_out),
                nn.GELU(),
                nn.Linear(cfg.cond_dim_out, cfg.cond_dim_out),
            )
        else:
            self.adapter = nn.Identity()

    def forward(self, cond: torch.Tensor, stop: bool) -> torch.Tensor:
        # cond: [B,T,C]
        if stop:
            cond = stopgrad(cond)
        return self.adapter(cond)
