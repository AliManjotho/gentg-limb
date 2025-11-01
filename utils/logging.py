"""Lightweight logging utilities (tqdm + TensorBoard optional)."""
from __future__ import annotations
from pathlib import Path
from typing import Optional

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore

class TBLogger:
    def __init__(self, logdir: str | Path, enabled: bool = True):
        self.enabled = enabled and (SummaryWriter is not None)
        self.writer = SummaryWriter(str(logdir)) if self.enabled else None
        Path(logdir).mkdir(parents=True, exist_ok=True)

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        if self.writer:
            self.writer.add_scalar(tag, scalar_value, global_step)

    def flush(self) -> None:
        if self.writer:
            self.writer.flush()

    def close(self) -> None:
        if self.writer:
            self.writer.close()
