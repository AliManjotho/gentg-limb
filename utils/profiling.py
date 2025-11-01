"""Profiling helpers for GenTG-Limb.

Includes:
- `torch_profiler` context manager wrapping `torch.profiler.profile` (optional)
- `profiled` decorator using `cProfile` (pure-Python)
- Lightweight CUDA/CPU sync helpers
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import io
import os
import pstats
import cProfile
from typing import Optional

try:
    import torch
    _TORCH = True
except Exception:  # pragma: no cover
    _TORCH = False


@contextmanager
def torch_profiler(enabled: bool = True,
                   activities: Optional[list] = None,
                   schedule_wait: int = 1,
                   schedule_warmup: int = 1,
                   schedule_active: int = 3,
                   record_shapes: bool = False,
                   profile_memory: bool = True,
                   with_stack: bool = False,
                   out_dir: Optional[str] = None):
    """Context manager for torch profiler with sensible defaults.

    Example:
        with torch_profiler(out_dir="outputs/prof") as prof:
            loss = model(x).sum(); loss.backward()
    """
    if not enabled or not _TORCH:
        yield None
        return

    if activities is None:
        activities = []
        if hasattr(torch.profiler, "ProfilerActivity"):
            A = torch.profiler.ProfilerActivity
            activities = [A.CPU]
            if torch.cuda.is_available():
                activities.append(A.CUDA)

    schedule = torch.profiler.schedule(wait=schedule_wait, warmup=schedule_warmup, active=schedule_active, repeat=1)
    tensorboard_trace_handler = None
    if out_dir is not None:
        tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(out_dir)

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=tensorboard_trace_handler,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        with_flops=True,
    ) as prof:
        yield prof


def cuda_sync():
    if _TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class ProfileResult:
    text: str
    stats: pstats.Stats


def profiled(sort_by: str = "cumtime"):
    """Decorator that profiles a function with cProfile and returns its result.
    Prints a summary sorted by `sort_by`.

    Usage:
        @profiled("tottime")
        def train_epoch(...):
            ...
    """
    def deco(fn):
        def wrapped(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            try:
                return fn(*args, **kwargs)
            finally:
                pr.disable()
                s = io.StringIO()
                ps = pstats.Stats(pr, stream=s).sort_stats(sort_by)
                ps.print_stats(30)
                print(s.getvalue())  # noqa: T201
        return wrapped
    return deco
