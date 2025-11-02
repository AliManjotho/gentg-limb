# GenTG-Limb: Generative Temporal Graph-Limb Transformer for 3D Human Pose Estimation

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---
## Overview

**GenTG-Limb** is a unified framework for *monocular 3D human pose estimation* that combines:
- a **Temporal Graph Transformer (TGT)** for deterministic 3D lifting, and  
- a **Generative Pose Corrector (GPC)** based on *diffusion models* for selective uncertainty-aware refinement.

<p align="center">
  <img src="assets/fig-model.png" width="600" alt="Pipeline Overview">
</p>

### Pipeline
```
Video → 2D keypoints → TGT → coarse 3D pose + uncertainty → GPC → refined 3D sequence
```

---

## Features

- Transformer-based 3D lifting using joint + limb tokenization  
- Diffusion-based refinement guided by uncertainty  
- Modular YAML configs for datasets, models, and training  
- Selective resampling of uncertain joints only  
- Evaluation + Visualization + Profiling utilities  
- Full test suite and CI support

---

## Installation

### 1️. Clone and setup environment
```bash
git clone https://github.com/AliManjotho/gentg-limb.git
cd gentg-limb
pip install -r requirements.txt
```

### 2️. Optional dependencies
```bash
pip install matplotlib torch-geometric tensorboard
```

### 3️. Verify installation
```bash
pytest -q
```

---

## Repository Structure
```
gentg-limb/
├─ configs/
├─ data/
├─ datasets/
├─ models/
├─ losses/
├─ utils/
├─ scripts/
├─ notebooks/
├─ tests/
└─ docs/
```

---

## Training
```bash
python scripts/train_tgt.py --config configs/train/tgt_base.yaml
python scripts/train_gpc.py --config configs/train/gpc_base.yaml
python scripts/finetune_e2e.py --config configs/train/e2e_finetune.yaml
```

---

## Inference
```bash
python scripts/infer.py --config configs/infer/offline_seq.yaml --input data/h36m/sample_seq.npz
```

Results are saved to `outputs/infer_result.npz`.

Visualize:
```bash
python scripts/visualize_seq.py --npz outputs/infer_result.npz --out viz/
```

---

![teaser](assets/qualitative.png)


## Evaluation
```bash
python -m metrics.eval_h36m
```

Implements MPJPE, P-MPJPE, 3DPCK, and AUC metrics.

---

## Configuration
All configs are YAML-based.

### Dataset
| Key | Description | Example |
|-----|--------------|----------|
| `name` | Dataset name | `human36m` |
| `window` | Temporal window length | 155 |
| `stride` | Frame stride | 1 |
| `camera_model` | Camera type | `perspective` |

### Model
| Key | Description |
|-----|--------------|
| `d_model` | Embedding dimension |
| `n_heads` | Attention heads |
| `n_layers` | Transformer layers |
| `dropout` | Dropout rate |

### Training
| Key | Description |
|-----|--------------|
| `epochs` | Total training epochs |
| `grad_clip` | Gradient clipping threshold |
| `ema` | Exponential moving average settings |

### Optimizer
| Key | Description |
|-----|--------------|
| `lr` | Learning rate |
| `weight_decay` | L2 regularization |
| `betas` | Adam beta parameters |

### Diffusion
| Key | Description |
|-----|--------------|
| `steps` | Number of diffusion steps |
| `sampler_steps` | Sampling steps at inference |
| `beta_schedule` | Schedule type (`linear`, `cosine`) |

### Logging
| Key | Description |
|-----|--------------|
| `interval` | Iterations between logs |
| `eval_interval` | Evaluation frequency |
| `checkpoint_interval` | Checkpoint frequency |

---

## Testing
```bash
pytest -v
```

Covers models, losses, metrics, and data pipelines.

---

## Notebooks
| Notebook | Purpose |
|-----------|----------|
| 00_quickstart.ipynb | Run a demo |
| 01_data_checks.ipynb | Inspect datasets |
| 02_train_tgt_logs.ipynb | Visualize TGT logs |
| 03_train_gpc_logs.ipynb | GPC training logs |
| 04_eval_plots.ipynb | Evaluation plots |

---

## Citation
```bibtex
@inproceedings{memon2025gentglimb,
author = {Anam Memon and Qasim Ali Arain and Nasrullah Pirzada and Muhammad Akram Shaikh and Ali Asghar Manjotho},
title = {GenTG-Limb: Generative Temporal Graph Transformers for Prior-Free 3D Human Pose},
booktitle = {In Proceedings of the First International Conference on Innovations in Information and Communication Technologies (IICT'26), Jan 15-17, 2026},
address = {Jamshoro, Pakistan},
publisher = {IEEE},
year = {2026}
}
```

---

## License
This code is distributed under an [MIT LICENSE](LICENSE).
