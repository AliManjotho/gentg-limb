"""Config loader/merger utilities using OmegaConf."""
from __future__ import annotations
from typing import Any, Dict, Sequence
from pathlib import Path
from omegaconf import OmegaConf

def load_config(*paths: str | Path) -> Dict[str, Any]:
    cfg = OmegaConf.create()
    for p in paths:
        if p is None: continue
        cfg = OmegaConf.merge(cfg, OmegaConf.load(str(p)))
    return OmegaConf.to_container(cfg, resolve=True)

def merge_cli_overrides(cfg: Dict[str, Any], overrides: Sequence[str] | None) -> Dict[str, Any]:
    if not overrides: return cfg
    cli = OmegaConf.from_dotlist(list(overrides))
    merged = OmegaConf.merge(OmegaConf.create(cfg), cli)
    return OmegaConf.to_container(merged, resolve=True)
