"""Temporal Graph Transformer blocks (joint+limb tokens)."""
from __future__ import annotations
import torch
import torch.nn as nn
from ..common.norm_dropout import make_norm, make_dropout

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, p_drop: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=p_drop, batch_first=True)
        self.drop = make_dropout(p_drop)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, N, D]; attn_bias (optional) added to attn weights [H, N, N] via key_padding_mask not supported;
        # we approximate by ignoring bias here for portability.
        out, _ = self.attn(x, x, x, need_weights=False)
        return self.drop(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, p_drop: float = 0.0):
        super().__init__()
        self.sa = MultiheadSelfAttention(d_model, n_heads, p_drop)
        self.ln1 = make_norm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.ln2 = make_norm(d_model)
        self.drop = make_dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm
        x = x + self.sa(self.ln1(x))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x

class TGTEncoder(nn.Module):
    """
    Processes concatenated joint and limb tokens per time step, then aggregates temporally.
    Input:
        joint_tokens: [B, T, J, D]
        limb_tokens:  [B, T, L, D]
    Output:
        context: [B, T, (J+L), D]
    """
    def __init__(self, d_model: int = 256, n_heads: int = 8, n_layers: int = 6, p_drop: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, ff_mult=4, p_drop=p_drop) for _ in range(n_layers)])

    def forward(self, joint_tokens: torch.Tensor, limb_tokens: torch.Tensor) -> torch.Tensor:
        B, T, J, D = joint_tokens.shape
        L = limb_tokens.shape[2]
        # Flatten time & tokens for self-attention over the token axis per frame, then over time by reshaping
        x = torch.cat([joint_tokens, limb_tokens], dim=2)  # [B,T,J+L,D]
        # Merge T and tokens → simple attention over tokens per frame
        x = x.reshape(B*T, J+L, D)
        for blk in self.layers:
            x = blk(x)
        x = x.reshape(B, T, J+L, D)
        return x
