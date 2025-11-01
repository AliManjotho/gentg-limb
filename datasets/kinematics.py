"""Skeleton and limb definitions for GenTG-Limb."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

# Default 17-joint skeleton (H36M-style index mapping illustrative; adapt to your exact order)
JOINT_NAMES = [
    "Hip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "Spine", "Thorax", "Neck", "Head", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist"
]

# Limb edges as pairs of joint indices (parent -> child)
DEFAULT_LIMBS: List[Tuple[int, int]] = [
    (0,1),(1,2),(2,3),
    (0,4),(4,5),(5,6),
    (0,7),(7,8),(8,9),(9,10),
    (8,11),(11,12),(12,13),
    (8,14),(14,15),(15,16)
]

# Left/right symmetric limb pairs (by limb index; keep lengths similar via soft constraint)
SYMMETRY_PAIRS: List[Tuple[int, int]] = [
    (0,3), (1,4), (2,5),   # legs
    (10,13), (11,14), (12,15)  # arms
]
