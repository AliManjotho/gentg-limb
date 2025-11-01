# Inference

After training TGT and GPC, you can run inference on a sequence of 2D keypoints.

## Offline sequence
```bash
python scripts/infer.py --config configs/infer/offline_seq.yaml --input data/h36m/sample_seq.npz
```

This will:
1. Load the 2D keypoints (`k2d`).
2. Run TGT to generate coarse 3D joints and per-joint uncertainty (`sigma`).
3. Optionally refine uncertain joints with GPC.
4. Save results to `outputs/infer_result.npz`.

The output file contains:
- `xyz`: final 3D coordinates [T, J, 3]

## Real-time mode
For webcam or streaming use:
```bash
python scripts/infer.py --config configs/infer/realtime_shortwin.yaml
```

To speed up inference:
- Reduce window size (e.g., 81 → 27).
- Lower `sampler_steps` in the GPC config.
- Disable GPC entirely for fastest deterministic output.
