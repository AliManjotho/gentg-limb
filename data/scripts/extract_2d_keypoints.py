#!/usr/bin/env python
"""Extract 2D keypoints from videos (dummy placeholder)."""
import argparse, numpy as np
from pathlib import Path
from common import ensure_dir, save_npz, discover_videos

def synthesize_keypoints(T:int, J:int=17):
    k2d = np.random.randn(T,J,2).astype(np.float32)*40.0+480.0
    conf = np.clip(np.random.rand(T,J).astype(np.float32),0.2,1.0)
    return k2d,conf

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--videos",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    vids=[args.videos] if args.videos.is_file() else discover_videos(args.videos)
    out=ensure_dir(args.out)
    for v in vids:
        T=100; k2d,conf=synthesize_keypoints(T)
        save_npz(out/(v.stem+".npz"),k2d=k2d,conf=conf)
        print(f"{v.name}: saved {out/(v.stem+'.npz')}")

if __name__=="__main__": main()
