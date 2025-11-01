"""Reprojection consistency loss utilities."""
from __future__ import annotations
import torch

def reprojection_l2(xyz: torch.Tensor, k2d_obs: torch.Tensor, camera, conf: torch.Tensor | None = None) -> torch.Tensor:
    """
    Args:
      xyz:      [B,T,J,3] predicted 3D joints (camera space)
      k2d_obs:  [B,T,J,2] observed 2D keypoints (pixels or normalized)
      camera:   object with .project(xyz) -> [B,T,J,2]
      conf:     [B,T,J] optional confidence mask (0..1)
    Returns:
      scalar loss
    """
    B,T,J,_ = xyz.shape
    k2d_pred = camera.project(xyz.view(B*T, J, 3)).view(B, T, J, 2)
    diff2 = (k2d_pred - k2d_obs) ** 2
    if conf is not None:
        diff2 = diff2 * conf.unsqueeze(-1)
    return diff2.mean()
