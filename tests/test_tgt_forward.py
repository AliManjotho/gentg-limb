import torch
from models.tgt.model import TGT
from datasets.kinematics import DEFAULT_LIMBS

def test_tgt_forward_shapes():
    B,T,J = 2, 9, 17
    k2d = torch.randn(B,T,J,2)
    coarse = torch.zeros(B,T,J,3)
    model = TGT(joints=J, limbs=DEFAULT_LIMBS, d_model=64, n_heads=4, n_layers=2, dropout=0.0)
    out = model(k2d, coarse)
    assert set(out.keys()) == {"xyz","sigma"}
    assert out["xyz"].shape == (B,T,J,3)
    assert out["sigma"].shape == (B,T,J)
