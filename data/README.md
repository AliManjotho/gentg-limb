# Datasets

This project supports **Human3.6M** and **MPI-INF-3DHP**.

## 1) Human3.6M
- Obtain access from the official website.
- Extract the dataset to a folder, e.g. `data/h36m/`.
- Preprocess with:
  ```bash
  python data/scripts/prepare_h36m.py --raw_dir data/h36m_raw --out_dir data/h36m
  ```
- Expected artifacts:
  - `data/h36m/annotations/` (2D/3D joint arrays, camera params)
  - `data/h36m/videos/` (optional for visualization)
  - splits JSON under `data/h36m/splits/`

## 2) MPI-INF-3DHP
- Download from the official website.
- Extract to `data/mpi_inf_3dhp/`.
- Preprocess with:
  ```bash
  python data/scripts/prepare_mpi_inf_3dhp.py --raw_dir data/mpi_raw --out_dir data/mpi_inf_3dhp
  ```

## 3) 2D Keypoints
We use an external 2D HPE to generate detections for training/inference:
```bash
python data/scripts/extract_2d_keypoints.py --videos data/... --out data/.../keypoints
```
This produces `.npz` with arrays:
- `k2d`: shape `[T, J, 2]`
- `conf`: shape `[T, J]` (optional confidences)

## 4) Integrity checks & splits
```bash
python data/scripts/build_splits.py --dataset human36m --out data/h36m/splits
```

## Directory structure
```
data/
├─ h36m/
│  ├─ annotations/
│  ├─ videos/ (optional)
│  └─ splits/
├─ mpi_inf_3dhp/
│  ├─ annotations/
│  ├─ videos/ (optional)
│  └─ splits/
└─ cache/
```

> Tip: large binary artifacts and raw videos should not be committed. `.gitignore` ignores typical paths.
