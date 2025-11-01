#!/usr/bin/env python
from __future__ import annotations
import argparse, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def plot_2d_skeleton(k2d: np.ndarray, limbs, out_path: Path, every: int = 5):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    T, J, _ = k2d.shape
    for t in range(0, T, every):
        plt.figure()
        x, y = k2d[t, :, 0], k2d[t, :, 1]
        plt.scatter(x, y, s=10)
        for a,b in limbs:
            plt.plot([x[a], x[b]], [y[a], y[b]])
        plt.gca().invert_yaxis()
        plt.title(f"Frame {t}")
        plt.savefig(out_path.parent / f"frame_{t:05d}.png")
        plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True, help="NPZ with k2d[TxJx2] or xyz[TxJx3]")
    ap.add_argument("--out", type=Path, required=True, help="Output folder or image file base path")
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    if "k2d" in data:
        k2d = data["k2d"]
        # simple limb chain for generic visualization
        J = k2d.shape[1]
        limbs = [(i, i+1) for i in range(J-1)]
        plot_2d_skeleton(k2d, limbs, args.out if args.out.suffix else (args.out / "frame_00000.png"))
        print(f"Saved 2D skeleton frames to {args.out}")
    elif "xyz" in data:
        print("3D visualization not implemented in this minimal script. Export xyz and visualize in your tool of choice.")
    else:
        raise ValueError("NPZ must contain 'k2d' or 'xyz'")

if __name__ == "__main__":
    main()
