import torch
from losses.mpjpe import mpjpe, p_mpjpe
from losses.limb_smooth import temporal_smooth_l2
from losses.symmetry import symmetry_loss

def test_mpjpe_and_p_mpjpe():
    pred = torch.zeros(4, 10, 17, 3)
    gt   = torch.zeros(4, 10, 17, 3)
    assert float(mpjpe(pred, gt)) == 0.0
    assert float(p_mpjpe(pred, gt)) == 0.0

def test_temporal_smooth_l2():
    xyz = torch.zeros(2, 5, 4, 3)  # constant pose => smoothness == 0
    limbs = [(0,1),(1,2),(2,3)]
    val = float(temporal_smooth_l2(xyz, limbs))
    assert abs(val) < 1e-8

def test_symmetry_loss():
    xyz = torch.zeros(2, 5, 4, 3)
    pairs = [((0,1),(2,3))]
    val = float(symmetry_loss(xyz, pairs))
    assert abs(val) < 1e-8
