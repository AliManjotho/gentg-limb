import torch
from metrics.pck_auc import pck_3d, auc_3d
from metrics.eval_h36m import evaluate_h36m

def test_pck_and_auc_shapes():
    pred = torch.zeros(2, 10, 17, 3)
    gt = torch.zeros_like(pred)
    pck = pck_3d(pred, gt)
    auc = auc_3d(pred, gt)
    assert isinstance(pck, torch.Tensor)
    assert isinstance(auc, torch.Tensor)
    assert 0.0 <= float(pck) <= 1.0
    assert 0.0 <= float(auc) <= 1.0

def test_evaluate_h36m_returns_dict():
    pred = torch.randn(1, 5, 17, 3)
    gt = pred.clone()
    metrics = evaluate_h36m(pred, gt)
    assert isinstance(metrics, dict)
    for key in ("mpjpe", "p_mpjpe", "pck150", "auc"):
        assert key in metrics
