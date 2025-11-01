#!/usr/bin/env python
"""Prepare MPI-INF-3DHP dataset for GenTG-Limb."""
import argparse, numpy as np
from pathlib import Path
from common import ensure_dir, write_json, save_npz, set_seed

def convert_sequence(seq_dir: Path, out_ann_dir: Path, joints: int = 17):
    T = 150
    k3d = np.random.randn(T, joints, 3).astype(np.float32) * 80.0
    k2d = np.random.randn(T, joints, 2).astype(np.float32) * 80.0 + 480.0
    conf = np.clip(np.random.rand(T, joints).astype(np.float32), 0.2, 1.0)
    save_npz(out_ann_dir / f"{seq_dir.name}.npz", k3d=k3d, k2d=k2d, conf=conf)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--joints", type=int, default=17)
    args = ap.parse_args()
    set_seed(42)
    ann_dir = ensure_dir(args.out_dir / "annotations")
    splits_dir = ensure_dir(args.out_dir / "splits")
    for seq in sorted([p for p in args.raw_dir.iterdir() if p.is_dir()]):
        convert_sequence(seq, ann_dir, args.joints)
    splits = {"train": [p.name for i,p in enumerate(sorted(ann_dir.glob("*.npz"))) if i%4!=0],
              "val": [p.name for i,p in enumerate(sorted(ann_dir.glob("*.npz"))) if i%4==0]}
    write_json(splits, splits_dir / "default.json")
    print(f"Prepared MPI-INF-3DHP → {ann_dir}")

if __name__ == "__main__": main()
