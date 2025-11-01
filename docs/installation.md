# Installation

## Requirements
- Python 3.9–3.11
- PyTorch 2.3+ (CUDA build recommended)
- (Optional) torch-geometric for graph ops

## Setup
```bash
git clone https://github.com/yourname/gentg-limb.git
cd gentg-limb
pip install -r requirements.txt
```

### Torch Geometric (optional)
Follow the official instructions for your CUDA and PyTorch version.

## Verify
```bash
pytest -q
```
