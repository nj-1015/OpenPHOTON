"""S1 exit metrics (plan slide 37 G1 + the plan's "G1 exit metric + L_rec
reconciliation" section: G1 is a structural-capability gate, not a
convergence gate -- separate the TRAINED objective from the MEASURED
diagnostic).

G1-a (primary, TRAINED): did each of the 4 per-block relative-MSE curves
fall (EMA-smoothed slope), and is no term dominating (plan risk R-A3 --
if the decoder terms are >> the encoder terms even after per-term
relative-MSE normalization, the 7/7/7/7 split or R_l is wrong; re-split
before S2, don't just reweight the loss)?

G1-b (diagnostic, MEASURED not trained -- plan risk R-A2: decoupled
teacher forcing means the composed end-to-end error is unbounded by the
per-block losses alone, by design; S2 closes that gap): the
student-RUNNING probe -- run the student's OWN chained forward (teacher-
forcing OFF) and check its dec2 reconstruction (h1_hat) tracks its own
enc1/enc2 outputs (h1, h2). `photon.model.forward(..., return_intermediates=True)`
already returns exactly h1/h2/h1_hat -- no new model code needed, this is
pure measurement on top of the existing S0 return contract.
"""
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def ema(values: Sequence[float], alpha: float = 0.1) -> list:
    out = []
    m = None
    for v in values:
        m = float(v) if m is None else alpha * float(v) + (1 - alpha) * m
        out.append(m)
    return out


def fit_slope(values: Sequence[float]) -> float:
    """Least-squares slope of `values` vs. step index -- negative = falling."""
    y = np.asarray(values, dtype=np.float64)
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=np.float64)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def loss_fell(values: Sequence[float], frac: float = 0.2) -> bool:
    """Robust to single-step noise: mean of the last `frac` of the run is
    below the mean of the first `frac`, rather than comparing two single
    (possibly noisy) endpoints."""
    n = len(values)
    k = max(1, int(n * frac))
    return float(np.mean(values[-k:])) < float(np.mean(values[:k]))


def g1a_report(loss_history: dict) -> dict:
    """loss_history: {"enc1": [...], "enc2": [...], "dec2": [...], "dec1": [...]}
    (train.train_s1.TrainerState.loss_history's shape).

    Returns a per-group dict of {initial, final, initial_raw, final_raw,
    slope, fell} plus a top-level `dec_dominates_enc` flag (plan R-A3's
    re-split trigger -- decoder finals more than 5x the encoder finals,
    both already on the same relative-MSE scale).

    `initial`/`final` are windowed means (first/last `frac`=20% of the
    run, matching `loss_fell`'s own window) rather than two raw single
    steps -- the S1 per-block loss is measured per-microbatch (no
    averaging), so a single endpoint is noisy; "final<initial" per the
    plan is a statement about the trend over the run ("slope over 50
    steps, final<initial" -- the plan's own phrasing groups these under
    one trend judgment), not a bet on any one step's sample. `initial_raw`/
    `final_raw` (the literal first/last values) are kept alongside for
    transparency/debugging.
    """
    frac = 0.2
    report = {}
    finals = {}
    for name, values in loss_history.items():
        n = len(values)
        k = max(1, int(n * frac))
        report[name] = dict(
            initial=float(np.mean(values[:k])),
            final=float(np.mean(values[-k:])),
            initial_raw=float(values[0]),
            final_raw=float(values[-1]),
            slope=fit_slope(values),
            fell=loss_fell(values, frac=frac),
        )
        finals[name] = report[name]["final"]

    dec_max = max((finals[n] for n in ("dec1", "dec2") if n in finals), default=0.0)
    enc_max = max((finals[n] for n in ("enc1", "enc2") if n in finals), default=1e-12)
    report["dec_dominates_enc"] = dec_max > 5 * enc_max
    return report


@torch.no_grad()
def g1b_student_running_probe(photon, input_ids: torch.LongTensor) -> dict:
    """cos(h1_hat, h1) and cos(chunk2(h1_hat), h2) -- the plan's exact G1-b
    formula, run on the student's own chained (non-teacher-forced) forward.
    MEASURED, not trained (driving this as a trained L_rec loss is S2
    scope, per the plan)."""
    was_training = photon.training
    photon.eval()
    try:
        _, mid = photon(input_ids, return_intermediates=True)
    finally:
        photon.train(was_training)

    h1, h2, h1_hat = mid["h1"], mid["h2"], mid["h1_hat"]
    cos_h1 = F.cosine_similarity(h1_hat.float(), h1.float(), dim=-1).mean().item()
    chunk2_h1_hat = photon.chunk2(h1_hat)
    cos_h2 = F.cosine_similarity(chunk2_h1_hat.float(), h2.float(), dim=-1).mean().item()
    return dict(cos_h1_hat_h1=cos_h1, cos_chunk2_h1_hat_h2=cos_h2)
