"""End-to-end wrapper composing TGT and GPC with optional stop-gradient bridge."""
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn

from ..tgt.model import TGT
from ..gpc.model import GPC
from .stopgrad_bridge import BridgeConfig, FeatureBridge, stopgrad

class E2EModel(nn.Module):
    def __init__(self, joints: int = 17, d_model: int = 256, n_heads: int = 8, n_layers: int = 6,
                 gpc_steps: int = 1000, gpc_sampler_steps: int = 30, gpc_schedule: str = "cosine",
                 stop_gradient: bool = True, cond_dim: Optional[int] = None, proj_cond_dim: Optional[int] = None):
        super().__init__()
        self.joints = joints
        self.tgt = TGT(joints=joints, d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=0.1)
        # conditioning: k2d (J*2) + tgt xyz.xy (J*2) by default
        c_in = cond_dim if cond_dim is not None else joints * 4
        self.bridge = FeatureBridge(BridgeConfig(stop_gradient=stop_gradient,
                                                adapt_cond=True,
                                                cond_dim_in=c_in,
                                                cond_dim_out=(proj_cond_dim or c_in)))
        self.gpc = GPC(joints=joints, steps=gpc_steps, sampler_steps=gpc_sampler_steps,
                       schedule=gpc_schedule, cond_dim=(proj_cond_dim or c_in))
        self.stop_gradient = stop_gradient

    def forward(self, k2d: torch.Tensor, coarse0: torch.Tensor, gpc_train: bool = True) -> Dict[str, torch.Tensor]:
        """Run TGT → build conditioning → (optionally) run GPC.
        Args:
          k2d:      [B,T,J,2]
          coarse0:  [B,T,J,3] initial coarse (zeros or prior)
          gpc_train: if True, compute diffusion training objective; else do sampling
        Returns dict with keys: 'tgt_xyz','sigma','gpc_loss' (if train) and/or 'xyz_final'
        """
        out_tgt = self.tgt(k2d, coarse0)
        xyz, sigma = out_tgt["xyz"], out_tgt["sigma"]
        # Build conditioning (concat k2d + xy from xyz) → flatten joints
        cond = torch.cat([k2d, xyz[...,:2]], dim=-1).reshape(k2d.shape[0], k2d.shape[1], -1)
        cond = self.bridge(cond, stop=self.stop_gradient)

        out: Dict[str, torch.Tensor] = {"tgt_xyz": xyz, "sigma": sigma}

        if self.training and gpc_train:
            # Train diffusion to denoise TGT outputs (teacher-forced with GT externally)
            out_gpc = self.gpc(xyz, cond=cond)
            out["gpc_loss"] = out_gpc["loss"]
            out["eps_hat"] = out_gpc["eps_hat"]
        else:
            # Inference-time selective refinement using TGT uncertainty
            mask = (sigma >= 0.35)  # default threshold; override at call site if needed
            x_refined = self.gpc.sample(xyz, mask, cond=cond)
            out["xyz_final"] = x_refined
        return out
