"""Per-term relative-MSE S1 loss ("Loss (per-term scale-normalized)" in
the plan).

Plan risk R-A3: depth-scale imbalance across the 4 groups (‖H28‖ >> ‖H7‖ --
activation norms grow with depth) means a plain sum-of-squared-error would
let the deepest/largest-norm target dominate the gradient regardless of
how "wrong" the shallower groups are. Relative MSE
(`‖pred - target‖² / ‖target‖²`) normalizes each term to its own target's
scale before summing/weighting, so the 7/7/7/7 depth split isn't silently
re-weighted by activation-norm growth with depth -- monitor per-group grad
norms during training; if the decoder terms still dominate after this
normalization, that's the plan's signal to re-split (e.g. 6/6/8/8) before
S2, not a loss-function problem.
"""
import torch

DEFAULT_WEIGHTS = {"enc1": 1.0, "enc2": 1.0, "dec2": 1.0, "dec1": 1.0}


def relative_mse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """‖pred - target‖² / ‖target‖², computed in fp32 regardless of input
    dtype (bf16 accumulation over a full [B,T,d] tensor is numerically
    dicey for a loss that then goes straight into a gradient)."""
    num = (pred.float() - target.float()).pow(2).sum()
    den = target.float().pow(2).sum().clamp_min(eps)
    return num / den


def group_loss(pred: torch.Tensor, target: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    return weight * relative_mse(pred, target)
