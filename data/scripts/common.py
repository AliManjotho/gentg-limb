"""Common I/O utilities for dataset preparation scripts."""
from __future__ import annotations
import json, os, random
from dataclasses import dataclass
from pathlib import Path
from typing import List
import numpy as np

def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def write_json(obj, path: str | os.PathLike) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, indent=2)

def save_npz(path: str | os.PathLike, **arrays) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, **arrays)

def discover_videos(root: str | os.PathLike, exts=(".mp4", ".avi", ".mov")) -> List[Path]:
    root = Path(root); return [p for p in root.rglob("*") if p.suffix.lower() in exts]

def set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed)

@dataclass
class KPTSpec:
    joints: int = 17
    order: str = "h36m-17"
