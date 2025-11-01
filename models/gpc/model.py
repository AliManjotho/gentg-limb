"""GPC wrapper: diffusion process + denoiser + selective resampling and guidance losses."""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
from .diffusion import DiffusionProcess
from .denoiser_unet import TemporalUNet
from .conditioner import build_conditioning, make_resample_mask
from .selective_resample import apply_selective_mask

class GPC(nn.Module):
    def __init__(self, joints: int = 17, steps: int = 1000, sampler_steps: int = 50, schedule: str = "cosine", cond_dim: int = 0):
        super().__init__()
        self.joints = joints
        self.jcoords = joints * 3
        self.diff = DiffusionProcess(steps=steps, schedule=schedule)
        self.sampler_steps = sampler_steps
        self.denoiser = TemporalUNet(jcoords=self.jcoords, hidden=256, cond_dim=cond_dim)

    def forward(self, x0: torch.Tensor, cond: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        """
        Training-time: predict eps on random diffusion step.
        x0:   [B,T,J,3]
        cond: [B,T,C] optional conditioning
        """
        B,T,J,C = x0.shape
        x0_flat = x0.reshape(B, T, -1)
        t = torch.randint(0, self.diff.steps, (B,), device=x0.device)
        x_t, eps = self.diff.add_noise(x0_flat, t)
        eps_hat = self.denoiser(x_t, cond=cond)
        loss = torch.mean((eps_hat - eps) ** 2)
        return {"loss": loss, "eps_hat": eps_hat}

    @torch.no_grad()
    def sample(self, x_start: torch.Tensor, mask_btj: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """
        Inference-time selective sampling: only masked joints are refined.
        x_start: [B,T,J,3] initial pose (from TGT)
        mask_btj: [B,T,J] True => refine
        """
        B,T,J,_ = x_start.shape
        x = x_start.reshape(B, T, -1)
        for i, t in enumerate(torch.linspace(self.diff.steps-1, 0, steps=self.sampler_steps, device=x.device).long()):
            eps_hat = self.denoiser(x, cond=cond)
            t_prev = int((i+1 < self.sampler_steps) and torch.linspace(self.diff.steps-1, 0, steps=self.sampler_steps, device=x.device).long()[i+1].item() or -1)
            x = self.diff.ddim_step(x, int(t.item()), t_prev, eps_hat)
            # Reshape & selectively replace
            x_pose = x.reshape(B, T, J, 3)
            x = apply_selective_mask(x_start, x_pose, mask_btj).reshape(B, T, -1)
        return x.reshape(B, T, J, 3)
