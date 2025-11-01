"""Normalization and dropout helpers."""
from __future__ import annotations
import torch.nn as nn

def make_norm(dim: int, type_: str = "layernorm") -> nn.Module:
    if type_.lower() in ("layernorm", "ln"):
        return nn.LayerNorm(dim)
    raise ValueError(f"Unknown norm type: {type_}")

def make_dropout(p: float = 0.0) -> nn.Module:
    return nn.Dropout(p)
