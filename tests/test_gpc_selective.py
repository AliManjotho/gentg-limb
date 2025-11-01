import torch
from models.gpc.selective_resample import apply_selective_mask
from models.gpc.model import GPC

def test_selective_resample_masking():
    B,T,J = 1, 4, 5
    x_cur = torch.zeros(B,T,J,3)
    x_new = torch.ones(B,T,J,3)
    mask = torch.zeros(B,T,J, dtype=torch.bool)
    mask[:,:,0] = True  # only joint 0 replaced
    out = apply_selective_mask(x_cur, x_new, mask)
    assert float(out[:,:,0,:].mean()) == 1.0
    assert float(out[:,:,1:,:].mean()) == 0.0

def test_gpc_training_step_shapes():
    B,T,J = 2, 8, 6
    x0 = torch.randn(B,T,J,3)
    cond = torch.randn(B,T,J*2)  # arbitrary cond channels
    gpc = GPC(joints=J, steps=50, sampler_steps=5, schedule="cosine", cond_dim=J*2)
    out = gpc(x0, cond=cond)
    assert "loss" in out and "eps_hat" in out
    assert out["eps_hat"].shape == (B,T,J*3)
