"""P2 -- the local S2 smoke test (build plan): T=128, B=2, ~50 steps,
WSL cu128 Blackwell, 0 Colab CU. Mirrors tests/test_s1_smoke.py's
discipline exactly, but exercises the S2 crux (composed forward + trained
L_rec, one backward, single global grad-norm clip) instead of S1's 4
decoupled sub-forward/backward groups.

Asserts:
  1. the loop actually runs on the GPU in bf16 (gated on a live bf16
     matmul, skips cleanly on CPU-only or unsupported builds).
  2. each of the 3 L_S2 terms (final/rec_A/rec_B) falls: slope<0 AND
     mean(last 20%) < mean(first 20%) -- proves the composed crux (incl.
     the trained L_rec) produces real, useful gradient on ALL THREE terms,
     not just that the loop runs.
  3. resume is bit-identical: checkpoint@step25, drop every in-memory
     object, reconstruct a genuinely fresh trainer state, resume, and
     match an uninterrupted 30-step run on both final weights and the
     next batch drawn -- same "simulated kill" discipline as
     tests/test_checkpoint_roundtrip.py / tests/test_s1_smoke.py.
  4. memory sanity (peak allocated stays within a generous local-GPU
     budget -- a no-OOM smoke gate, not a benchmark).
  5. `train.s2_step.build_state`'s `init_from` weight-seeding path
     (`checkpoint.ckpt.load(init_from, photon, optimizer=None)`) actually
     loads weights when the file exists -- SmokeS2Config itself uses
     `init_from=None` (fresh surgery; doesn't need the real 3GB S1
     checkpoint on disk), so this is exercised separately against a tiny
     throwaway checkpoint.

8-bit Adam / bitsandbytes availability follows the same pattern as S1
(train_s1.build_optimizer, reused verbatim by train.s2_step.build_state).
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
    from configs.s2 import SmokeS2Config
    cfg = SmokeS2Config()
    cfg.shard_dir = shard_dir
    return cfg


def test_smoke_loop_runs_and_all_three_terms_fall(smoke_shard):
    from harness.g1_metrics import fit_slope, loss_fell
    from train.s2_step import S2_LOSS_KEYS, build_state, train_step

    cfg = _smoke_cfg(smoke_shard)
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    state, opt_name = build_state(cfg, device="cuda")
    print(f"\n[smoke] optimizer: {opt_name}  hardware={REASON}  init_from={cfg.init_from}")

    n_steps = 50
    for _ in range(n_steps):
        train_step(state)
        # the loop actually runs on the Blackwell card in bf16 -- not
        # silently on CPU / silently upcast
        p = next(state.photon.parameters())
        assert p.is_cuda
        assert p.dtype == torch.bfloat16

    all_fell = True
    for name in S2_LOSS_KEYS:
        values = state.loss_history[name]
        n = len(values)
        k = max(1, int(n * 0.2))
        initial = sum(values[:k]) / k
        final = sum(values[-k:]) / k
        slope = fit_slope(values)
        fell = loss_fell(values, frac=0.2)
        print(f"[smoke] {name}: initial={initial:.4e} final={final:.4e} slope={slope:.4e}")
        assert final < initial, f"{name} loss did not fall: initial={initial} final={final}"
        assert slope < 0, f"{name} loss slope not negative: {slope}"
        assert fell, f"{name} loss did not fall robustly (mean last20% vs first20%)"
        all_fell = all_fell and (final < initial and slope < 0 and fell)

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"[smoke] peak CUDA memory allocated: {peak_mem_gb:.2f} GB")
    assert peak_mem_gb < 20.0, f"peak memory {peak_mem_gb:.2f}GB exceeds the local-GPU smoke budget"


def test_smoke_resume_bit_identical(smoke_shard, tmp_path):
    from train.s2_step import build_state, checkpoint, resume, train_step

    cfg = _smoke_cfg(smoke_shard)
    ckpt_path = str(tmp_path / "smoke_s2_ckpt.pt")

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


def test_build_state_seeds_weights_from_init_from_when_file_exists(smoke_shard, tmp_path):
    """SmokeS2Config uses init_from=None (fresh surgery -- doesn't need the
    real 3GB S1 checkpoint on disk). This test exercises the OTHER branch
    of train.s2_step.build_state directly: with init_from pointed at a
    tiny throwaway checkpoint (NOT the real S1 one), build_state must load
    those exact weights via checkpoint.ckpt.load(init_from, photon,
    optimizer=None) -- proving the "seed weights from cfg.init_from" path
    itself works, independent of the real S1 checkpoint's availability."""
    from train.s2_step import build_state, checkpoint, train_step

    cfg = _smoke_cfg(smoke_shard)
    cfg.init_from = None
    seed_state, _ = build_state(cfg, device="cuda")

    # take a real step first, so the checkpointed optimizer state is
    # actually POPULATED (else "loaded optimizer has empty state" would
    # pass vacuously -- there'd be nothing for ckpt.load(..., optimizer=None)
    # to have (correctly) failed to restore).
    train_step(seed_state)
    assert len(seed_state.optimizer.state) > 0, "sanity: optimizer state should be populated after a step"

    # mutate one real parameter to a distinctive, checkable value, then
    # checkpoint it -- this is now the "S1-like" warm-start checkpoint.
    with torch.no_grad():
        seed_state.photon.chunk1.proj.weight.fill_(0.12345)
    tiny_ckpt_path = str(tmp_path / "tiny_init_from.pt")
    checkpoint(seed_state, tiny_ckpt_path)
    mutated_value = seed_state.photon.chunk1.proj.weight.detach().clone()
    del seed_state
    torch.cuda.empty_cache()

    # a genuinely fresh build_state, seeded from that tiny checkpoint
    cfg2 = _smoke_cfg(smoke_shard)
    cfg2.init_from = tiny_ckpt_path
    loaded_state, _ = build_state(cfg2, device="cuda")

    assert torch.equal(loaded_state.photon.chunk1.proj.weight.detach(), mutated_value), (
        "build_state's init_from seeding did not load the checkpointed weights"
    )
    # optimizer must be FRESH at cfg.lr, NOT restored from the tiny ckpt's
    # (non-empty, just verified above) optimizer state -- build_state calls
    # ckpt.load(..., optimizer=None), which must skip the optimizer entirely.
    assert len(loaded_state.optimizer.state) == 0
    assert loaded_state.optimizer.param_groups[0]["lr"] == cfg2.lr
