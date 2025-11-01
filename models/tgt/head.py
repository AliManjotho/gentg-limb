"""Heads for 3D coordinates and uncertainty prediction from TGT context."""
from __future__ import annotations
import torch
import torch.nn as nn

class PoseHead(nn.Module):
    """Predict per-joint 3D coordinates from joint-context tokens."""
    def __init__(self, d_model: int = 256, joints: int = 17):
        super().__init__()
        self.proj = nn.Linear(d_model, 3)
        self.joints = joints

    def forward(self, joint_tokens: torch.Tensor) -> torch.Tensor:
        # joint_tokens: [B,T,J,D]
        xyz = self.proj(joint_tokens)  # [B,T,J,3]
        return xyz

class UncertaintyHead(nn.Module):
    """Predict per-joint uncertainty (sigma in [0,1]) from joint-context tokens."""
    def __init__(self, d_model: int = 256, joints: int = 17):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)
        self.act = nn.Sigmoid()
        self.joints = joints

    def forward(self, joint_tokens: torch.Tensor) -> torch.Tensor:
        sigma = self.act(self.proj(joint_tokens)).squeeze(-1)  # [B,T,J]
        return sigma
