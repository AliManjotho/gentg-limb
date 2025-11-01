"""Temporal denoiser (U-Net style) for diffusion on pose sequences."""
from __future__ import annotations
import torch
import torch.nn as nn
from einops import rearrange

class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
        )
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x):
        return self.norm(self.block(x) + x)

class TemporalUNet(nn.Module):
    """
    Operates on sequences flattened over joints*coords.
    Input:  x_t      [B, T, J*3]
            t_embed  [B, D] (optional simple Fourier/time embedding)
            cond     [B, T, C] conditioning channels (e.g., k2d, tgt_xyz)
    Output: eps_hat  [B, T, J*3]
    """
    def __init__(self, jcoords: int, hidden: int = 256, cond_dim: int = 0):
        super().__init__()
        d_in = jcoords + cond_dim
        self.in_proj  = nn.Linear(d_in, hidden)
        self.encoder1 = ResidualBlock(hidden)
        self.down     = nn.AvgPool1d(kernel_size=2, stride=2)
        self.encoder2 = ResidualBlock(hidden)
        self.up       = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder1 = ResidualBlock(hidden)
        self.out_proj = nn.Linear(hidden, jcoords)

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        # x_t: [B,T,J*3]; cond: [B,T,C] or None
        if cond is None:
            z = x_t
        else:
            z = torch.cat([x_t, cond], dim=-1)  # [B,T,d_in]
        z = self.in_proj(z)                      # [B,T,H]
        z = rearrange(z, 'b t h -> b h t')

        e1 = self.encoder1(z)                    # [B,H,T]
        z  = self.down(e1)                       # [B,H,T/2]
        z  = self.encoder2(z)                    # [B,H,T/2]
        z  = self.up(z)                          # [B,H,T]
        z  = self.decoder1(z + e1)               # [B,H,T]

        z  = rearrange(z, 'b h t -> b t h')
        eps = self.out_proj(z)                   # [B,T,J*3]
        return eps
