"""tests/test_s3_smoke.py -- the S3 KD end-to-end smoke as a pytest
(mirrors tests/test_s2_smoke.py). bf16-gated: SKIPS on a machine with no
usable bf16 CUDA device (e.g. the native-Windows CPU torch), RUNS on the
pod / any cc>=8.0 GPU. The human-readable equivalent is scripts/smoke_s3.py.

Checks: (1) the KD loss falls over ~40 steps (real distillation gradient);
(2) checkpoint@20 -> simulated kill -> resume reproduces weights AND the
next dataloader batch bit-identically.
"""
import os

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE_SHARD_DIR = os.path.join(REPO_ROOT, "data", "_smoke_shards")


def _bf16_cuda_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        if torch.cuda.get_device_capability(0)[0] < 8:
            return False
        x = torch.randn(4, 4, device="cuda", dtype=torch.bfloat16)
        (x @ x).sum().item()
        torch.cuda.synchronize()
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _bf16_cuda_usable(),
    reason="no usable bf16 CUDA device (S3 smoke needs bf16; runs on the pod / cc>=8.0 GPU)")


def _ensure_smoke_shard():
    import subprocess
    import sys
    if not os.path.exists(os.path.join(SMOKE_SHARD_DIR, "index.json")):
        subprocess.run([sys.executable, os.path.join(REPO_ROOT, "scripts", "build_smoke_shard.py")], check=True)


def _cfg():
    from configs.s3 import SmokeS3Config
    cfg = SmokeS3Config()
    cfg.shard_dir = SMOKE_SHARD_DIR
    return cfg


def test_kd_loss_falls():
    from harness.g1_metrics import fit_slope, loss_fell
    from train.s3_step import build_state, train_step
    _ensure_smoke_shard()

    torch.manual_seed(0)
    state, _ = build_state(_cfg(), device="cuda")
    for _ in range(40):
        train_step(state)
    kd = state.loss_history["kd"]
    assert fit_slope(kd) < 0 and loss_fell(kd, frac=0.2), f"KD did not fall: {kd[:3]} -> {kd[-3:]}"


def test_resume_bit_identical():
    from train.s3_step import build_state, checkpoint, resume, train_step
    _ensure_smoke_shard()
    ckpt_path = os.path.join(REPO_ROOT, "checkpoints", "_smoke_s3", "ckpt.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    cfg = _cfg()

    torch.manual_seed(0)
    ref, _ = build_state(cfg, device="cuda")
    for _ in range(25):
        train_step(ref)
    ref_weights = {k: v.clone() for k, v in ref.photon.state_dict().items()}
    ref_next = ref.loader.next_batch()

    torch.manual_seed(0)
    killed, _ = build_state(cfg, device="cuda")
    for _ in range(20):
        train_step(killed)
    checkpoint(killed, ckpt_path)
    del killed
    torch.cuda.empty_cache()

    resumed, _ = resume(cfg, ckpt_path, device="cuda")
    for _ in range(5):
        train_step(resumed)
    got = resumed.photon.state_dict()
    assert all(torch.equal(v, got[k]) for k, v in ref_weights.items()), "weights not bit-identical after resume"
    assert torch.equal(ref_next, resumed.loader.next_batch()), "dataloader position not bit-identical after resume"
