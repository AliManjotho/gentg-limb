# Training

## Stage 1 — TGT
```bash
python scripts/train_tgt.py --config configs/train/tgt_base.yaml
```

## Stage 2 — GPC
```bash
python scripts/train_gpc.py --config configs/train/gpc_base.yaml
```

## Optional — End-to-End
```bash
python scripts/finetune_e2e.py --config configs/train/e2e_finetune.yaml
```

### Tips
- Start with small windows on limited GPUs (see `examples/sample_config_overrides/small_gpu.yaml`).
- Monitor MPJPE and reprojection loss. If reprojection dominates, re-check camera parameters.
- For diffusion speed, reduce `sampler_steps` and consider subnets with smaller hidden sizes.
