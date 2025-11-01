# GenTG-Limb Documentation

Welcome to **GenTG-Limb**, a prior-free monocular 3D human pose estimation framework.

---

## Overview
GenTG-Limb unifies a **Temporal Graph Transformer (TGT)** for deterministic 3D lifting with a **Generative Pose Corrector (GPC)** based on diffusion for uncertainty-aware refinement.

### Pipeline
```
Video → 2D keypoints → TGT → coarse 3D pose + uncertainty → GPC → refined 3D sequence
```

### Key Components
- **TGT:** Joint + limb tokenization, temporal attention, multi-loss training.
- **GPC:** Diffusion-based selective correction of uncertain joints.
- **Losses:** MPJPE, reprojection, limb-length smoothness, left/right symmetry.

---

## Features
- End-to-end trainable 3D pose estimation from monocular 2D keypoints.
- Uncertainty-driven generative refinement.
- Modular, configurable, and PyTorch-based.
- Includes full configs, datasets, metrics, and tests.

For installation instructions, see [Installation](installation.md).
