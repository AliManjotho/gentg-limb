"""
Dataset package for GenTG-Limb.
Exposes factory `build_dataset` for named datasets.
"""
from .human36m import Human36M
from .mpi_inf_3dhp import MPIInf3DHP

__all__ = ["Human36M", "MPIInf3DHP"]
