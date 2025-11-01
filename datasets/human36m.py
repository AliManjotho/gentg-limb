"""PyTorch Dataset for Human3.6M (processed NPZ)."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

class Human36M(Dataset):
    def __init__(self, root: str | Path, split: str = "train", window: int = 155, stride: int = 1):
        self.root = Path(root)
        self.ann_dir = self.root / "annotations"
        self.splits_path = self.root / "splits" / "default.json"
        self.split = split
        self.window = window
        self.stride = stride

        with open(self.splits_path, "r", encoding="utf-8") as f:
            files = json.load(f)[split]
        self.files = [self.ann_dir / fn for fn in files]

        # build index of (file, start_frame)
        self.index: List[tuple[int, int]] = []
        for fi, fpath in enumerate(self.files):
            with np.load(fpath, allow_pickle=True) as npz:
                T = npz["k2d"].shape[0]
            starts = list(range(0, max(1, T - self.window + 1), self.stride))
            self.index.extend([(fi, s) for s in starts])

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fi, s = self.index[idx]
        fpath = self.files[fi]
        with np.load(fpath, allow_pickle=True) as npz:
            k2d = npz["k2d"][s:s+self.window]  # [T, J, 2]
            k3d = npz.get("k3d", np.zeros((*k2d.shape[:-1], 3), dtype=np.float32))[s:s+self.window]
            conf = npz.get("conf", np.ones(k2d.shape[:2], dtype=np.float32))[s:s+self.window]

        k2d = torch.from_numpy(k2d).float()
        k3d = torch.from_numpy(k3d).float()
        conf = torch.from_numpy(conf).float()

        sample = {
            "k2d": k2d,       # [T, J, 2]
            "k3d": k3d,       # [T, J, 3]
            "conf": conf,     # [T, J]
        }
        return sample
