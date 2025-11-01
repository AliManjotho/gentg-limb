import torch
from models.common.graph_utils import build_limb_index, joints_to_limbs

def test_build_and_compute_limb_vectors():
    limbs = [(0,1),(1,2),(2,3)]
    idx = build_limb_index(limbs, device="cpu")
    assert idx.shape == (2, 3)
    xyz = torch.zeros(2, 5, 4, 3)  # B=2, T=5, J=4
    xyz[...,1,:] = torch.tensor([1.0,0.0,0.0])
    xyz[...,2,:] = torch.tensor([1.0,2.0,0.0])
    xyz[...,3,:] = torch.tensor([1.0,2.0,3.0])
    v = joints_to_limbs(xyz, idx)
    assert v.shape == (2,5,3,3)
    # vector 0: 1-0 = [1,0,0]
    assert torch.allclose(v[...,0,:], torch.tensor([1.0,0.0,0.0]))
