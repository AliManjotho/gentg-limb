"""Camera models and projection utilities."""
from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class PerspectiveCamera:
    fx: float
    fy: float
    cx: float
    cy: float

    def project(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        xyz: [..., 3]
        return: [..., 2]
        """
        x, y, z = xyz.unbind(-1)
        z = torch.clamp(z, min=1e-6)
        u = self.fx * (x / z) + self.cx
        v = self.fy * (y / z) + self.cy
        return torch.stack([u, v], dim=-1)

    def unproject(self, uvz: torch.Tensor) -> torch.Tensor:
        """
        uvz: [..., 3] where last dim is (u, v, z)
        return xyz: [..., 3]
        """
        u, v, z = uvz.unbind(-1)
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return torch.stack([x, y, z], dim=-1)


@dataclass
class WeakPerspectiveCamera:
    s: float  # scale
    tx: float # translation x
    ty: float # translation y

    def project(self, xyz: torch.Tensor) -> torch.Tensor:
        x, y, _ = xyz.unbind(-1)
        u = self.s * x + self.tx
        v = self.s * y + self.ty
        return torch.stack([u, v], dim=-1)


def reprojection_loss(k2d_pred: torch.Tensor, k2d_obs: torch.Tensor, conf: torch.Tensor | None = None) -> torch.Tensor:
    """
    k2d_pred: [B, T, J, 2]
    k2d_obs:  [B, T, J, 2]
    conf:     [B, T, J]
    """
    if conf is None:
        return torch.mean((k2d_pred - k2d_obs) ** 2)
    return torch.mean(((k2d_pred - k2d_obs) ** 2) * conf.unsqueeze(-1))
