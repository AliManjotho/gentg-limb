"""Minimal visualization helpers for 2D/3D skeletons with matplotlib."""
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np

def _import_mpl():
    import matplotlib.pyplot as plt  # local import to avoid hard dependency
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    return plt

def plot_2d_skeleton(k2d: np.ndarray, limbs: List[Tuple[int,int]], title: Optional[str] = None, out_path: Optional[str] = None):
    """Plot a single 2D pose (J,2)."""
    plt = _import_mpl()
    x, y = k2d[:,0], k2d[:,1]
    plt.figure()
    plt.scatter(x, y, s=10)
    for a,b in limbs:
        plt.plot([x[a], x[b]], [y[a], y[b]])
    plt.gca().invert_yaxis()
    if title: plt.title(title)
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_3d_skeleton(xyz: np.ndarray, limbs: List[Tuple[int,int]], title: Optional[str] = None, out_path: Optional[str] = None):
    """Plot a single 3D pose (J,3)."""
    plt = _import_mpl()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs = xyz[:,0], xyz[:,1], xyz[:,2]
    ax.scatter(xs, ys, zs, s=10)
    for a,b in limbs:
        ax.plot([xs[a], xs[b]], [ys[a], ys[b]], [zs[a], zs[b]])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    if title: ax.set_title(title)
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
