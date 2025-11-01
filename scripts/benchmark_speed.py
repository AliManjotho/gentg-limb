#!/usr/bin/env python
from __future__ import annotations
import argparse, time, torch
from pathlib import Path
from datasets.kinematics import DEFAULT_LIMBS
from models.tgt.model import TGT

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=int, default=155)
    ap.add_argument('--joints', type=int, default=17)
    ap.add_argument('--iters', type=int, default=50)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TGT(joints=args.joints, limbs=DEFAULT_LIMBS, d_model=256, n_heads=8, n_layers=6, dropout=0.1).to(device).eval()
    B,T,J = 2, args.window, args.joints
    k2d = torch.randn(B,T,J,2, device=device)
    coarse = torch.zeros(B,T,J,3, device=device)

    # warmup
    for _ in range(10):
        with torch.no_grad():
            model(k2d, coarse)

    torch.cuda.synchronize() if device=='cuda' else None
    t0 = time.time()
    with torch.no_grad():
        for _ in range(args.iters):
            out = model(k2d, coarse)
    torch.cuda.synchronize() if device=='cuda' else None
    dt = (time.time() - t0) / args.iters
    fps = (B*T) / dt
    print(f'Avg latency: {dt*1000:.2f} ms/iter | Effective tokens/frame: {B*T} | Throughput ~= {fps:.1f} frames/s')

if __name__ == '__main__':
    main()
