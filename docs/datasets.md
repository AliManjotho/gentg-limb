# Datasets

GenTG-Limb supports:
- **Human3.6M**
- **MPI-INF-3DHP**

## Preprocessing
Use the provided scripts in `data/scripts/`:

### Human3.6M
```bash
python data/scripts/prepare_h36m.py --raw_dir data/h36m_raw --out_dir data/h36m
```

### MPI-INF-3DHP
```bash
python data/scripts/prepare_mpi_inf_3dhp.py --raw_dir data/mpi_raw --out_dir data/mpi_inf_3dhp
```

### 2D keypoints extraction
```bash
python data/scripts/extract_2d_keypoints.py --videos data/videos --out data/keypoints
```

### Splits
```bash
python data/scripts/build_splits.py --dataset human36m --root data/h36m
```

## Expected structure
```
data/
├─ h36m/
│  ├─ annotations/
│  ├─ splits/
│  └─ videos/
├─ mpi_inf_3dhp/
│  ├─ annotations/
│  ├─ splits/
│  └─ videos/
└─ cache/
```

> Note: raw videos are optional for training but required for visualization.
