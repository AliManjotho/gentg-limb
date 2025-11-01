"""Metric aggregation helpers: running averages and pretty dicts."""
from __future__ import annotations

class RunningMean:
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def update(self, value: float, count: int = 1):
        self.sum += float(value) * count
        self.n += int(count)

    @property
    def mean(self) -> float:
        return self.sum / max(1, self.n)

    def __repr__(self) -> str:
        return f"RunningMean(mean={self.mean:.4f}, n={self.n})"
