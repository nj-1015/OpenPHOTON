"""Standalone driver for the local S3 smoke test -- same checks as
tests/test_s3_smoke.py, runnable directly (mirrors scripts/smoke_s2.py).

WSL cu128 Blackwell (or any usable bf16 CUDA device), 0 Colab CU:

    python scripts/smoke_s3.py

Checks: (1) the KD loss FALLS over 50 steps (real distillation gradient),
rec_B stays bounded, ce is reported (0 when ce_weight=0); (2) resume is
bit-identical (checkpoint@25 -> simulated kill -> reconstruct -> match).
"""
import os
import subprocess
import sys

import torch

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO_ROOT)

SMOKE_SHARD_DIR = os.path.join(REPO_ROOT, "data", "_smoke_shards")
N_STEPS = 50
CKPT_AT_STEP = 25  # 1-indexed: checkpoint after the 25th train_step() call


def ensure_smoke_shard():
    if not os.path.exists(os.path.join(SMOKE_SHARD_DIR, "index.json")):
        print("smoke shard missing -- building it now (scripts/build_smoke_shard.py)...")
        subprocess.run([sys.executable, os.path.join(REPO_ROOT, "scripts", "build_smoke_shard.py")], check=True)


def check_cuda_bf16():
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False"
    try:
        cc = torch.cuda.get_device_capability(0)
        if cc[0] < 8:
            return False, f"compute capability {cc} < (8,0)"
        x = torch.randn(4, 4, device="cuda", dtype=torch.bfloat16)
        (x @ x).sum().item()
        torch.cuda.synchronize()
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    return True, f"cc={cc}, device={torch.cuda.get_device_name(0)}"


def main():
    usable, reason = check_cuda_bf16()
    print(f"bf16 CUDA usable: {usable} ({reason})")
    if not usable:
        print("SKIP: no usable bf16 CUDA device on this machine.")
        return

    ensure_smoke_shard()

    from configs.s3 import SmokeS3Config
    from harness.g1_metrics import fit_slope, loss_fell
    from train.s3_step import S3_LOSS_KEYS, build_state, checkpoint, resume, train_step

    cfg = SmokeS3Config()
    cfg.shard_dir = SMOKE_SHARD_DIR
    ckpt_path = os.path.join(REPO_ROOT, "checkpoints", "_smoke_s3", "ckpt.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    print("\n=== part 1: L_S3 per-term loss curves over", N_STEPS, "steps ===")
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    state, opt_name = build_state(cfg, device="cuda")
    print("optimizer:", opt_name, " init_from:", cfg.init_from,
          " ce_weight:", cfg.ce_weight)
    for _ in range(N_STEPS):
        train_step(state)
    # KD is the primary signal and MUST fall. rec_B should stay bounded (it
    # starts near its floor from the warm init; on a cold smoke it may wander
    # a little -- reported, not asserted). ce is a constant 0 when
    # ce_weight=0, so its "fell" check is not meaningful -- reported only.
    kd_values = state.loss_history["kd"]
    n = len(kd_values)
    k = max(1, int(n * 0.2))
    kd_initial = sum(kd_values[:k]) / k
    kd_final = sum(kd_values[-k:]) / k
    kd_slope = fit_slope(kd_values)
    kd_fell = kd_final < kd_initial and kd_slope < 0 and loss_fell(kd_values, frac=0.2)
    for name in S3_LOSS_KEYS:
        values = state.loss_history[name]
        initial = sum(values[:k]) / k
        final = sum(values[-k:]) / k
        slope = fit_slope(values)
        tag = ("FELL" if kd_fell else "DID NOT FALL") if name == "kd" else "(report-only)"
        print(f"  {name:6s} initial={initial:.4e} final={final:.4e} slope={slope:.4e}  {tag}")
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"  peak CUDA memory: {peak_gb:.2f} GB")
    print("PART 1:", "PASS" if kd_fell else "FAIL")

    print("\n=== part 2: resume bit-identical (checkpoint@step", CKPT_AT_STEP, ") ===")
    torch.manual_seed(0)
    ref_state, _ = build_state(cfg, device="cuda")
    for step in range(30):
        train_step(ref_state)
    ref_final = {k2: v.clone() for k2, v in ref_state.photon.state_dict().items()}
    ref_next_batch = ref_state.loader.next_batch()

    torch.manual_seed(0)
    killed_state, _ = build_state(cfg, device="cuda")
    for step in range(CKPT_AT_STEP):
        train_step(killed_state)
    checkpoint(killed_state, ckpt_path)
    del killed_state
    torch.cuda.empty_cache()

    resumed_state, resumed_opt_name = resume(cfg, ckpt_path, device="cuda")
    print("resumed optimizer:", resumed_opt_name)
    for step in range(30 - CKPT_AT_STEP):
        train_step(resumed_state)

    resumed_final = resumed_state.photon.state_dict()
    weights_match = all(torch.equal(v, resumed_final[k2]) for k2, v in ref_final.items())
    batch_match = torch.equal(ref_next_batch, resumed_state.loader.next_batch())
    print("weights bit-identical:", weights_match)
    print("next-batch bit-identical:", batch_match)
    print("PART 2:", "PASS" if (weights_match and batch_match) else "FAIL")


if __name__ == "__main__":
    main()
