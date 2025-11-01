# 🧠 GenTG-Limb: Generative Temporal Graph-Limb Transformer for 3D Human Pose Estimation

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://github.com/yourname/gentg-limb/actions/workflows/ci.yml/badge.svg)

---

## 🌐 Overview

**GenTG-Limb** is a unified framework for *monocular 3D human pose estimation* that combines:
- a **Temporal Graph Transformer (TGT)** for deterministic 3D lifting, and  
- a **Generative Pose Corrector (GPC)** based on *diffusion models* for selective uncertainty-aware refinement.

<p align="center">
  <img src="docs/images/pipeline_overview.png" width="600" alt="Pipeline Overview">
</p>

### Pipeline
```
Video → 2D keypoints → TGT → coarse 3D pose + uncertainty → GPC → refined 3D sequence
```

---

## ✨ Features

- Transformer-based 3D lifting using joint + limb tokenization  
- Diffusion-based refinement guided by uncertainty  
- Modular YAML configs for datasets, models, and training  
- Selective resampling of uncertain joints only  
- Evaluation + Visualization + Profiling utilities  
- Full test suite and CI support

---

## 📦 Installation

### 1️⃣ Clone and setup environment
```bash
git clone https://github.com/yourname/gentg-limb.git
cd gentg-limb
pip install -r requirements.txt
```

### 2️⃣ Optional dependencies
```bash
pip install matplotlib torch-geometric tensorboard
```

### 3️⃣ Verify installation
```bash
pytest -q
```

---

## 📂 Repository Structure
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

## 🏋️ Training
```bash
python scripts/train_tgt.py --config configs/train/tgt_base.yaml
python scripts/train_gpc.py --config configs/train/gpc_base.yaml
python scripts/finetune_e2e.py --config configs/train/e2e_finetune.yaml
```

---

## 🎥 Inference
```bash
python scripts/infer.py --config configs/infer/offline_seq.yaml --input data/h36m/sample_seq.npz
```

Results are saved to `outputs/infer_result.npz`.

Visualize:
```bash
python scripts/visualize_seq.py --npz outputs/infer_result.npz --out viz/
```

---

## 📈 Evaluation
```bash
python -m metrics.eval_h36m
```

Implements MPJPE, P-MPJPE, 3DPCK, and AUC metrics.

---

## ⚙️ Configuration
All configs are YAML-based. See [`docs/configuration_reference.md`](docs/configuration_reference.md).

---

## 🧪 Testing
```bash
pytest -v
```

Covers models, losses, metrics, and data pipelines.

---

## 📘 Notebooks
| Notebook | Purpose |
|-----------|----------|
| 00_quickstart.ipynb | Run a demo |
| 01_data_checks.ipynb | Inspect datasets |
| 02_train_tgt_logs.ipynb | Visualize TGT logs |
| 03_train_gpc_logs.ipynb | GPC training logs |
| 04_eval_plots.ipynb | Evaluation plots |

---

## 🧮 Citation
```bibtex
@article{gentg2025,
  title={GenTG-Limb: Generative Temporal Graph-Limb Transformer for 3D Human Pose Estimation},
  author={Your Name et al.},
  journal={arXiv preprint arXiv:2501.xxxxx},
  year={2025}
}
```

---

## 🛠️ License
MIT License. See [LICENSE](LICENSE).
