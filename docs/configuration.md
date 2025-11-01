# Configuration Guide

GenTG-Limb uses YAML configs for modular training and inference setups.

## Basic structure
Example: `configs/train/tgt_base.yaml`
```yaml
dataset:
  name: human36m
  joints: 17
  window: 155

model:
  type: TGT
  d_model: 256
  n_heads: 8
  n_layers: 6

optimizer:
  name: adamw
  lr: 1e-3

train:
  epochs: 120
```

### Overriding from CLI
You can override any parameter using dotlist syntax:
```bash
python scripts/train_tgt.py --config configs/train/tgt_base.yaml   optimizer.lr=5e-4 train.epochs=50
```

### Logging configuration
Defined in `configs/logging.yaml` — controls checkpoints and loggers.

### Dataset configuration
`configs/dataset/` holds camera models and paths. You can add your own datasets here.

### Inference configuration
`configs/infer/` specifies paths to pretrained checkpoints and runtime behavior.
