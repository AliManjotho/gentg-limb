# FAQ

### 1. My training diverges early — what should I check?
- Ensure 2D keypoints are normalized and in consistent pixel coordinates.
- Try smaller learning rates (e.g., `lr=5e-4`).
- Disable GPC and train only TGT first.

### 2. My 3D outputs look mirrored.
- Check camera coordinate conventions.
- Verify left/right limb index mapping in `datasets/kinematics.py`.

### 3. Diffusion correction is too slow.
- Reduce `sampler_steps` from 50 → 20.
- Increase `sigma_threshold` to limit resampling frequency.

### 4. CUDA out of memory.
- Reduce batch size and temporal window.
- Use mixed precision (`precision: amp_bf16`).

### 5. How to visualize?
```bash
python scripts/visualize_seq.py --npz outputs/infer_result.npz --out viz/
```

### 6. How to export ONNX model?
```bash
python scripts/export.py --checkpoint checkpoints/tgt_epoch120.pth
```
