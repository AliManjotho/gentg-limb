import torch
from losses.reprojection import reprojection_l2
from datasets.cameras import PerspectiveCamera

def test_reprojection_loss_zero_on_perfect_match():
    cam = PerspectiveCamera(fx=1000.0, fy=1000.0, cx=500.0, cy=500.0)
    B,T,J = 1, 5, 3
    xyz = torch.tensor([[[[0.0,0.0,1.0],[1.0,0.0,1.0],[0.0,1.0,1.0]]]]).expand(B,T,J,3)
    k2d = cam.project(xyz.view(-1,J,3)).view(B,T,J,2)
    loss = reprojection_l2(xyz, k2d, cam)
    assert abs(float(loss)) < 1e-6

def test_reprojection_loss_increases_with_error():
    cam = PerspectiveCamera(fx=1000.0, fy=1000.0, cx=500.0, cy=500.0)
    B,T,J = 1, 5, 3
    xyz = torch.tensor([[[[0.0,0.0,1.0],[1.0,0.0,1.0],[0.0,1.0,1.0]]]]).expand(B,T,J,3)
    k2d_true = cam.project(xyz.view(-1,J,3)).view(B,T,J,2)
    k2d_err = k2d_true + 10.0
    loss_true = reprojection_l2(xyz, k2d_true, cam)
    loss_err = reprojection_l2(xyz, k2d_err, cam)
    assert loss_err > loss_true
