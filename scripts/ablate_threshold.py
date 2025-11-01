#!/usr/bin/env python
from __future__ import annotations
import argparse, numpy as np
from pathlib import Path
import torch
from datasets.kinematics import DEFAULT_LIMBS
from models.tgt.model import TGT
from models.gpc.model import GPC
from utils.checkpoint import load_checkpoint

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tgt_ckpt', type=Path, required=True)
    ap.add_argument('--gpc_ckpt', type=Path, required=True)
    ap.add_argument('--input', type=Path, required=True, help='NPZ with k2d[TxJx2]')
    ap.add_argument('--joints', type=int, default=17)
    ap.add_argument('--thresholds', type=float, nargs='+', default=[0.25, 0.35, 0.45])
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = np.load(args.input, allow_pickle=True)
    k2d = torch.from_numpy(data['k2d']).float().unsqueeze(0).to(device)
    J = args.joints; T = k2d.shape[1]

    tgt = TGT(joints=J, limbs=DEFAULT_LIMBS, d_model=256, n_heads=8, n_layers=4, dropout=0.1).to(device)
    tgt.load_state_dict(load_checkpoint(args.tgt_ckpt, map_location=device).get('model', {}), strict=False)
    tgt.eval()

    gpc = GPC(joints=J, steps=1000, sampler_steps=30, schedule='cosine', cond_dim=J*4).to(device)
    gpc.load_state_dict(load_checkpoint(args.gpc_ckpt, map_location=device).get('model', {}), strict=False)
    gpc.eval()

    with torch.no_grad():
        coarse0 = torch.zeros((1,T,J,3), device=device)
        out_tgt = tgt(k2d, coarse0)
        xyz, sigma = out_tgt['xyz'], out_tgt['sigma']
        cond = torch.cat([k2d, xyz[...,:2]], dim=-1).reshape(1,T,-1)

        for thr in args.thresholds:
            mask = (sigma >= thr)
            x_out = gpc.sample(xyz, mask, cond=cond)
            mpjpe_est = torch.linalg.norm((x_out - xyz), dim=-1).mean().item()
            print(f'tau={thr:.2f} | mean delta (proxy for correction magnitude): {mpjpe_est:.3f}')
