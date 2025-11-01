#!/usr/bin/env python
"""Build train/val/test splits for processed datasets."""
import argparse
from pathlib import Path
from common import ensure_dir, write_json

def build_default(ann_dir:Path):
    files=sorted([p.name for p in ann_dir.glob('*.npz')])
    return{"train":[f for i,f in enumerate(files) if i%10 not in(0,1)],
           "val":[f for i,f in enumerate(files) if i%10==1],
           "test":[f for i,f in enumerate(files) if i%10==0]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    args=ap.parse_args()
    ann_dir=args.root/"annotations"
    splits_dir=ensure_dir(args.root/"splits")
    splits=build_default(ann_dir)
    write_json(splits,splits_dir/"default.json")
    print(f"Saved splits to {splits_dir/'default.json'}")    

if __name__=="__main__": main()
