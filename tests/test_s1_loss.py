"""train/s1_loss.py -- per-term relative-MSE (plan risk R-A3)."""
import torch

from train.s1_loss import group_loss, relative_mse


def test_relative_mse_zero_when_pred_equals_target():
    t = torch.randn(2, 4, 8)
    assert relative_mse(t, t).item() == 0.0


def test_relative_mse_scale_invariant_when_both_scaled_together():
    pred = torch.randn(2, 4, 8)
    target = torch.randn(2, 4, 8)
    base = relative_mse(pred, target).item()
    scaled = relative_mse(pred * 100.0, target * 100.0).item()
    assert abs(base - scaled) < 1e-4


def test_relative_mse_normalizes_across_target_scales():
    """plan risk R-A3: a target with 10x the norm should NOT automatically
    produce a 100x larger loss for the same relative error -- that's
    exactly what plain (non-relative) MSE would do, and exactly what this
    function exists to prevent."""
    torch.manual_seed(0)
    small_target = torch.randn(2, 4, 8)
    big_target = small_target * 10.0
    same_relative_error_pred_small = small_target * 1.1  # 10% relative error
    same_relative_error_pred_big = big_target * 1.1       # same 10% relative error

    loss_small = relative_mse(same_relative_error_pred_small, small_target).item()
    loss_big = relative_mse(same_relative_error_pred_big, big_target).item()
    assert abs(loss_small - loss_big) < 1e-4


def test_group_loss_applies_weight():
    pred = torch.zeros(1, 2, 4)
    target = torch.ones(1, 2, 4)
    base = group_loss(pred, target, weight=1.0)
    weighted = group_loss(pred, target, weight=0.5)
    assert torch.allclose(weighted, 0.5 * base)
