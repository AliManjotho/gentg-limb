#!/usr/bin/env python
from __future__ import annotations
import argparse, torch
from pathlib import Path
from datasets.kinematics import DEFAULT_LIMBS
from models.tgt.model import TGT
from utils.checkpoint import load_checkpoint

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True, help='TGT checkpoint (.pth)')
    ap.add_argument('--joints', type=int, default=17)
    ap.add_argument('--out_onnx', type=Path, default=Path('outputs/tgt.onnx'))
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TGT(joints=args.joints, limbs=DEFAULT_LIMBS, d_model=256, n_heads=8, n_layers=4, dropout=0.1).to(device)
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get('model', ckpt), strict=False)
    model.eval()

    # Dummy inputs: k2d [1,T,J,2], coarse [1,T,J,3] (T=27)
    T = 27; J = args.joints
    k2d = torch.randn(1, T, J, 2, device=device)
    coarse = torch.zeros(1, T, J, 3, device=device)

    out_path = args.out_onnx
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, (k2d, coarse), str(out_path),
                      input_names=['k2d', 'coarse_xyz'],
                      output_names=['xyz', 'sigma'],
                      opset_version=17, dynamic_axes={'k2d': {1: 'T'}, 'coarse_xyz': {1: 'T'}})
    print(f'Exported ONNX to {out_path}')

if __name__ == '__main__':
    main()
