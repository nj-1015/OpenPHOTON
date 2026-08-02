"""P4 -- the local S1 smoke test (plan's "the milestone to reach"):
T=128, B=2, ~50 steps, WSL cu128 Blackwell, 0 Colab CU.

Asserts:
  1. the loop actually runs on the GPU in bf16 (gated on a live bf16
     matmul, skips cleanly -- not an error -- on CPU-only or an
     architecturally-unsupported build, same discipline as
     tests/test_golden_bf16.py).
  2. each of the 4 per-block losses falls: slope<0 AND
     mean(last 20%) < mean(first 20%) (robust to single-step noise) --
     proves the crux wiring produces real, useful gradient, not just that
     the loop runs without crashing.
  3. resume is bit-identical: checkpoint@step25, drop every in-memory
     object, reconstruct a genuinely fresh trainer state, resume, and
     match an uninterrupted 30-step run on both final weights and the
     next batch drawn -- tests/test_checkpoint_roundtrip.py's "simulated
     kill" discipline, extended from a toy model to the real S1 trainer.
     Nothing in this training loop is stochastic (no dropout, no shuffle,
     deterministic teacher forward, deterministic shard order) other than
     the RNG state ckpt.DataloaderState carries along for API-contract
     reasons (see data/stream_dataset.py) -- so bit-identical is the
     correct bar here, not just "close."
  4. memory sanity (peak allocated stays within a generous local-GPU
     budget -- a no-OOM smoke gate, not a benchmark).

8-bit Adam on Blackwell/WSL cu128 is a plan-flagged known unknown --
train.train_s1.build_optimizer() tries bitsandbytes first and falls back
to AdamW with a loud log line if it's unusable; whichever ran is printed
here too.
"""
import os
import subprocess
import sys

import pytest
import torch

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
SMOKE_SHARD_DIR = os.path.join(REPO_ROOT, "data", "_smoke_shards")


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
def smoke_shard():
    if not os.path.exists(os.path.join(SMOKE_SHARD_DIR, "index.json")):
        subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "scripts", "build_smoke_shard.py")],
            check=True,
        )
    return SMOKE_SHARD_DIR


def _smoke_cfg(shard_dir):
    from configs.s1 import SmokeS1Config
    cfg = SmokeS1Config()
    cfg.shard_dir = shard_dir
    return cfg


def test_smoke_loop_runs_and_per_block_losses_fall(smoke_shard):
    from harness.g1_metrics import g1a_report
    from train.train_s1 import build_state, train_step

    cfg = _smoke_cfg(smoke_shard)
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    state, opt_name = build_state(cfg, device="cuda")
    print(f"\n[smoke] optimizer: {opt_name}  hardware={REASON}")

    n_steps = 50
    for _ in range(n_steps):
        train_step(state)
        # (1) the loop actually runs on the Blackwell card in bf16 -- not
        # silently on CPU / silently upcast
        p = next(state.photon.parameters())
        assert p.is_cuda
        assert p.dtype == torch.bfloat16

    report = g1a_report(state.loss_history)
    for name in ("enc1", "enc2", "dec2", "dec1"):
        r = report[name]
        print(f"[smoke] {name}: initial={r['initial']:.4e} final={r['final']:.4e} slope={r['slope']:.4e}")
        assert r["final"] < r["initial"], f"{name} loss did not fall: {r}"
        assert r["slope"] < 0, f"{name} loss slope not negative: {r}"
        assert r["fell"], f"{name} loss did not fall robustly (mean last20% vs first20%): {r}"
    assert not report["dec_dominates_enc"], (
        "decoder-group losses dominate encoder-group losses >5x even after "
        "relative-MSE normalization -- plan risk R-A3's re-split trigger"
    )

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"[smoke] peak CUDA memory allocated: {peak_mem_gb:.2f} GB")
    assert peak_mem_gb < 20.0, f"peak memory {peak_mem_gb:.2f}GB exceeds the local-GPU smoke budget"


def test_smoke_resume_bit_identical(smoke_shard, tmp_path):
    from train.train_s1 import build_state, checkpoint, resume, train_step

    cfg = _smoke_cfg(smoke_shard)
    ckpt_path = str(tmp_path / "smoke_ckpt.pt")

    # --- uninterrupted reference run: 30 steps straight through ---
    torch.manual_seed(0)
    ref_state, _ = build_state(cfg, device="cuda")
    for _ in range(30):
        train_step(ref_state)
    ref_final_state_dict = {k: v.clone() for k, v in ref_state.photon.state_dict().items()}
    ref_next_batch = ref_state.loader.next_batch()  # the batch step 31 WOULD consume

    # --- interrupted run: 25 steps, checkpoint, simulated kill ---
    torch.manual_seed(0)
    killed_state, _ = build_state(cfg, device="cuda")
    for _ in range(25):
        train_step(killed_state)
    checkpoint(killed_state, ckpt_path)
    del killed_state
    torch.cuda.empty_cache()

    # --- genuinely fresh reconstruction + resume ---
    resumed_state, opt_name = resume(cfg, ckpt_path, device="cuda")
    print(f"\n[smoke] resumed optimizer: {opt_name}")
    for _ in range(5):  # steps 26..30
        train_step(resumed_state)

    resumed_state_dict = resumed_state.photon.state_dict()
    for k, v in ref_final_state_dict.items():
        assert torch.equal(v, resumed_state_dict[k]), f"weight {k!r} diverged after resume"

    resumed_next_batch = resumed_state.loader.next_batch()
    assert torch.equal(ref_next_batch, resumed_next_batch), \
        "next batch after resume does not match the uninterrupted run"
