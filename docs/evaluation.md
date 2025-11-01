# Evaluation

GenTG-Limb supports standard evaluation metrics for 3D human pose estimation.

## Metrics

### Mean Per Joint Position Error (MPJPE)
$$
L_{MPJPE} = \frac{1}{J} \sum_{j=1}^{J} \| \hat{p}_j - p_j \|_2
$$

### Procrustes-Aligned MPJPE (P-MPJPE)
Rigid alignment of predicted and ground truth poses before computing MPJPE.

### 3D Percentage of Correct Keypoints (3DPCK)
$$
3DPCK = \frac{1}{J} \sum_{j=1}^{J} \mathbb{1}[\| \hat{p}_j - p_j \|_2 < \tau]
$$

### AUC (Area Under Curve)
Integration of 3DPCK across thresholds up to 150 mm.

---

## Running Evaluation
```bash
pytest -q tests/test_tgt_forward.py
python -m metrics.eval_h36m
```

To compute evaluation metrics manually:
```python
from metrics.eval_h36m import evaluate_h36m
results = evaluate_h36m(pred_xyz, gt_xyz)
print(results)
```

---

## Protocols
- **Human3.6M:** uses standard 17-joint subset with S9/S11 for testing.
- **MPI-INF-3DHP:** uses ground-truth camera projection for normalization.
