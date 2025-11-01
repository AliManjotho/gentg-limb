"""Reproducible seeding across numpy, random, torch."""
import os, random
import numpy as np

def set_seed(seed: int = 42, deterministic_torch: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True  # type: ignore
            torch.backends.cudnn.benchmark = False     # type: ignore
    except Exception:
        pass
