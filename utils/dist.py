"""Distributed helpers (minimal DDP launcher utilities)."""
from __future__ import annotations
import os, torch, torch.distributed as dist

def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def setup_distributed(backend: str = 'nccl', init_method: str = 'env://') -> None:
    if is_dist_avail_and_initialized():
        return
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend=backend, init_method=init_method)
    else:
        pass

def cleanup_distributed() -> None:
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()
