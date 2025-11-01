# Configuration Reference

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
