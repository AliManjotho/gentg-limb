"""Timing utilities for quick performance measurements."""
from __future__ import annotations
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict

try:
    import torch
    _TORCH = True
except Exception:  # pragma: no cover
    _TORCH = False

def _sync():
    if _TORCH and hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

@dataclass
class Timer:
    name: str = "timer"
    start_time: float = field(default=0.0, init=False)
    total: float = field(default=0.0, init=False)
    iters: int = field(default=0, init=False)

    def __enter__(self):
        _sync()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        _sync()
        dt = time.perf_counter() - self.start_time
        self.total += dt
        self.iters += 1

    @property
    def avg(self) -> float:
        return self.total / max(1, self.iters)

    def reset(self):
        self.total = 0.0
        self.iters = 0

    def __repr__(self) -> str:
        return f"Timer(name={self.name}, total={self.total:.4f}s, iters={self.iters}, avg={self.avg:.6f}s)"


class MultiTimer:
    def __init__(self):
        self.timers: Dict[str, Timer] = {}

    @contextmanager
    def time(self, key: str):
        t = self.timers.setdefault(key, Timer(name=key))
        with t:
            yield
    def summary(self) -> str:
        return " | ".join([f"{k}: avg {t.avg*1000:.2f} ms ({t.iters}x)" for k,t in self.timers.items()])
