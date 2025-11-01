"""Temporal positional encodings: absolute and relative."""
from __future__ import annotations
import math
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor, t0: int = 0) -> torch.Tensor:
        """
        x: [B, T, N, D] or [B, T, D]
        returns x + pe
        """
        T = x.shape[1]
        pe = self.pe[t0:t0+T].unsqueeze(1)  # [T,1,D]
        if x.dim() == 4:
            pe = pe.unsqueeze(2)            # [T,1,1,D]
        return x + pe

class RelativePositionBias(nn.Module):
    """
    Simple T5-style relative position bias for attention logits along time.
    """
    def __init__(self, num_buckets: int = 32, max_distance: int = 128, n_heads: int = 8):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.bias = nn.Embedding(num_buckets, n_heads)

    def _relative_bucket(self, rel_pos: torch.Tensor) -> torch.Tensor:
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        ret = 0
        n = -rel_pos
        # Positive and negative are handled symmetrically
        is_small = n < num_buckets // 2
        val_if_large = (num_buckets // 2) + (
            (torch.log(n.float() / (num_buckets // 2)) / math.log(max_distance / (num_buckets // 2)))
            * (num_buckets - num_buckets // 2)
        ).long().clamp(max=num_buckets-1)
        val_if_large = torch.where(n <= 0, torch.zeros_like(val_if_large), val_if_large)
        ret = torch.where(is_small, n.clamp(min=0), val_if_large)
        return ret

    def forward(self, T: int, device=None) -> torch.Tensor:
        ctx = torch.arange(T, device=device)
        rel = ctx[:, None] - ctx[None, :]  # [T,T]
        buckets = self._relative_bucket(rel)
        bias = self.bias(buckets)          # [T,T,H]
        return bias.permute(2,0,1)         # [H,T,T]
