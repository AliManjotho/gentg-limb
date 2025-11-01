#!/usr/bin/env python
"""Prepare Human3.6M annotations for GenTG-Limb."""
import argparse, numpy as np
from pathlib import Path
from common import ensure_dir, write_json, save_npz, set_seed

def convert_subject(subject_dir: Path, out_ann_dir: Path, joints: int = 17) -> int:
    seq_count = 0
    for i in range(2):  # mock 2 sequences
        T = 120
        k3d = np.random.randn(T, joints, 3).astype(np.float32) * 100.0
        k2d = np.random.randn(T, joints, 2).astype(np.float32) * 100.0 + 500.0
        conf = np.clip(np.random.rand(T, joints).astype(np.float32), 0.3, 1.0)
        save_npz(out_ann_dir / f"{subject_dir.name}_seq{i}.npz", k3d=k3d, k2d=k2d, conf=conf)
        seq_count += 1
    return seq_count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--joints", type=int, default=17)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    ann_dir = ensure_dir(args.out_dir / "annotations")
    splits_dir = ensure_dir(args.out_dir / "splits")
    total = 0
    for s in sorted([p for p in args.raw_dir.iterdir() if p.is_dir()]):
        total += convert_subject(s, ann_dir, args.joints)
    splits = {"train": [p.name for i,p in enumerate(sorted(ann_dir.glob("*.npz"))) if i%5!=0],
              "val": [p.name for i,p in enumerate(sorted(ann_dir.glob("*.npz"))) if i%5==0]}
    write_json(splits, splits_dir / "default.json")
    print(f"Converted {total} sequences → {ann_dir}")

if __name__ == "__main__": main()
