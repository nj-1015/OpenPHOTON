"""M5 (fp32/CPU leg) -- THE S0 gate.

bypass mode (C1=C2=R1=R2=1, identity chunkers/converters, full causal mask
over T, absolute positions, shift/pack/take_last_C as pass-throughs) must
reproduce stock Qwen3-0.6B logits to floating-point tolerance. If this test
fails, the fault is plumbing (weight loading, norm placement, tying, mask
construction), not architecture (slide 23) -- localise via the per-stage
hidden comparison printed by harness/golden_gate.run_case (reused here, not
duplicated).

Three cases (reviewer finding on T-0243's first review pass -- a single
short B=1 prompt under-tests the full-causal path against a length- or
batch-dependent mask bug): p0 (short, B=1, the original case), p1 (longer,
B=1), batch (p0+p1 truncated to a common length, stacked to B=2). See
harness/capture_reference.py's docstring for the exact construction.

Requires the reference .npz from harness/capture_reference.py; this test
generates it into a temp dir if missing rather than depending on a
checked-in fixture (per .gitignore: `*.npz` reference dumps are not
committed).
"""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
REF_PATH = os.path.join(REPO_ROOT, "_ref", "qwen3_0.6b_ref.npz")

sys.path.insert(0, REPO_ROOT)


@pytest.fixture(scope="module")
def ref():
    if not os.path.exists(REF_PATH):
        subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "harness", "capture_reference.py"), REF_PATH],
            check=True,
        )
    return np.load(REF_PATH)


@pytest.fixture(scope="module")
def bypass_photon():
    from photon.config import PhotonConfig
    from photon.surgery import load_photon_qwen3

    cfg = PhotonConfig(C1=1, C2=1, R1=1, R2=1, bypass=True)
    photon, stock = load_photon_qwen3(cfg)
    del stock
    return photon


@pytest.mark.parametrize("case", ["p0", "p1", "p2", "batch"])
def test_golden_bypass_matches_stock_logits(ref, bypass_photon, case):
    from harness.golden_gate import run_case

    result = run_case(bypass_photon, ref, case)

    max_abs = result["logit_report"]["max_abs"]
    agree = result["agree_frac"]
    assert max_abs < 1e-4, f"golden test RED ({case}): logits max_abs={max_abs:.3e} >= 1e-4"
    assert agree == 1.0, f"golden test RED ({case}): argmax agreement {agree * 100:.1f}% < 100%"
