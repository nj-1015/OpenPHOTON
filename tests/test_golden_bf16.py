"""M5b -- bf16/GPU golden leg (S0 exit criterion: "bf16 100% last-token
argmax agreement").

The CPU/fp32 golden leg (test_golden.py) already proves the orchestration
is bit-exact in fp32. bf16 won't be bit-exact -- bf16 has ~3 decimal digits
of mantissa precision, and Qwen3-0.6B's logits over a 151,936-wide vocab
easily accumulate multi-unit absolute differences at that precision -- so
the gate here is different from the fp32 leg: **100% LAST-token argmax
agreement** (what actually matters for greedy/sampled generation) plus a
loose near-100% floor across all positions, not a numeric max_abs
tolerance. Reference is the fp32/CPU logits captured by
harness/capture_reference.py (test_golden.py's `ref` fixture).

GPU+bf16-guarded: skips cleanly (not xfail, not error) on any machine
where CUDA compute isn't actually *usable* -- checked via a live smoke-test
matmul, not just `torch.cuda.is_available()`, because a CUDA device can be
*visible* to torch while being architecturally unsupported by the
installed build. That is exactly this repo's native-Windows dev
environment: an RTX 5070 Ti Laptop GPU (compute capability 12.0, Blackwell)
is visible to the native Windows torch install (2.10.0+cu126), which
reports `cuda.is_available() == True`, but that build only ships kernels
for sm_50..sm_90 -- any real kernel launch raises "no kernel image is
available for execution on the device". This test was actually exercised
via a WSL Ubuntu torch 2.10.0+cu128 build that DOES ship sm_120 kernels for
this same physical GPU (confirmed with a smoke-test matmul before writing
this file) -- see the task report for the full hardware-detect trace and
the actual bf16 numbers from that run.
"""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
REF_PATH = os.path.join(REPO_ROOT, "_ref", "qwen3_0.6b_ref.npz")


def _cuda_bf16_usable():
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False"
    try:
        cc = torch.cuda.get_device_capability(0)
    except Exception as e:
        return False, f"device capability query failed: {e}"
    if cc[0] < 8:
        return False, f"compute capability {cc} < (8, 0), bf16 tensor cores unavailable"
    try:
        x = torch.randn(4, 4, device="cuda", dtype=torch.bfloat16)
        (x @ x).sum().item()
        torch.cuda.synchronize()
    except Exception as e:
        return False, (f"CUDA smoke-test kernel launch failed ({type(e).__name__}): {e} "
                        f"-- device is visible but not usable by this torch build")
    return True, f"cc={cc}, device={torch.cuda.get_device_name(0)}"


USABLE, REASON = _cuda_bf16_usable()
pytestmark = pytest.mark.skipif(not USABLE, reason=f"bf16 CUDA not usable on this machine: {REASON}")


@pytest.fixture(scope="module")
def ref():
    if not os.path.exists(REF_PATH):
        subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "harness", "capture_reference.py"), REF_PATH],
            check=True,
        )
    return np.load(REF_PATH)


@pytest.mark.parametrize("case", ["p0", "p1", "p2", "batch"])
def test_golden_bypass_bf16_last_token_argmax(ref, case):
    from photon.config import PhotonConfig
    from photon.surgery import load_photon_qwen3

    cfg = PhotonConfig(C1=1, C2=1, R1=1, R2=1, bypass=True, dtype="bfloat16")
    photon, stock = load_photon_qwen3(cfg)
    del stock
    photon.to("cuda")
    photon.eval()
    # the tie must survive the device move (nn.Module.to() walks the shared
    # Parameter object once; it's the same Python object under both
    # attribute names, so this must still hold)
    assert photon.lm_head.weight is photon.embed_tokens.weight

    input_ids = torch.tensor(ref[f"{case}_input_ids"], dtype=torch.long, device="cuda")
    with torch.no_grad():
        logits, _ = photon(input_ids, return_intermediates=True)
    logits = logits.float().cpu().numpy()
    ref_logits = ref[f"{case}_logits"]  # fp32/CPU reference

    argmax = logits.argmax(-1)
    ref_argmax = ref_logits.argmax(-1)

    last_token_match = bool((argmax[:, -1] == ref_argmax[:, -1]).all())
    agree_frac = float((argmax == ref_argmax).mean())
    max_abs = float(np.abs(logits - ref_logits).max())

    print(f"\n[bf16 golden:{case}] hardware={REASON}")
    print(f"[bf16 golden:{case}] last_token_argmax_match={last_token_match}  "
          f"position_agree_frac={agree_frac * 100:.1f}%  max_abs={max_abs:.3e}")

    assert last_token_match, (
        f"bf16 golden RED ({case}): last-token argmax mismatch "
        f"(got {argmax[:, -1].tolist()}, expected {ref_argmax[:, -1].tolist()})"
    )
    # loose floor, not a numeric tolerance -- bf16 is expected to drift on
    # some interior positions, just not catastrophically
    assert agree_frac >= 0.90, (
        f"bf16 golden RED ({case}): position-level argmax agreement "
        f"{agree_frac * 100:.1f}% < 90% floor"
    )
