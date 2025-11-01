# Design Notes

## Temporal Graph Transformer (TGT)

TGT processes both **joint tokens** and **limb tokens** in a hybrid graph structure.

### Input representation
Each frame’s 2D keypoints $k_{t,j} = (x_{t,j}, y_{t,j})$ is embedded as:
$$
h^J_{t,j} = \text{MLP}_J(k_{t,j})
$$

Limb vectors between joint pairs $(a,b)$ are represented as:
$$
h^L_{t,l} = [\text{dir}_{t,l}, \text{len}_{t,l}] = \Big[\frac{p_b - p_a}{\|p_b - p_a\|}, \|p_b - p_a\|\Big]
$$

### Encoder
Multi-head temporal attention layers with residual connections and feedforward networks.

### Loss functions
- **MPJPE:** $L_{MPJPE}$  
- **Reprojection:** $L_{reproj} = \| \Pi(\hat{P}) - k_{2D} \|^2$  
- **Limb Smoothness:** $L_{len} = \sum_t (\ell_t - \ell_{t-1})^2$  
- **Symmetry:** $L_{sym} = \sum_i (\ell_i^{left} - \ell_i^{right})^2$

The total TGT loss is:
$$
L_{TGT} = L_{MPJPE} + \lambda_1 L_{reproj} + \lambda_2 L_{len} + \lambda_3 L_{sym}
$$

---

## Generative Pose Corrector (GPC)

### Diffusion Objective
Forward process:
$$
q(x_t | x_0) = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

Training objective:
$$
L_{GPC} = \mathbb{E}_{t,\epsilon} [\| \epsilon - \epsilon_\theta(x_t, t, c) \|^2]
$$

During inference, selective resampling is applied where uncertainty $\sigma_{t,j} > \tau$.

---

## Implementation Notes
- Training uses AdamW with cosine schedule and EMA.
- TGT window length: 155; GPC diffusion window: 27.
- Built with PyTorch 2.3, PyG 2.5.
