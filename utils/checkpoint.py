"""Checkpoint save/load helpers for GenTG-Limb."""
from pathlib import Path
from typing import Any, Dict, Optional
import torch

def save_checkpoint(path: str | Path, state: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path: str | Path, map_location: Optional[str] = None) -> Dict[str, Any]:
    return torch.load(str(path), map_location=map_location or "cpu")
