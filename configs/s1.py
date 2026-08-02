"""S1 hyperparameters.

`S1Config` is the production 0.5B-token shape -- authored for a training
notebook, NOT run automatically by this codebase; a real production run
needs an interactive GPU session plus real dataset ids/credentials, outside
the scope of this repo's automated tests.

`SmokeS1Config` is what tests/test_s1_smoke.py and scripts/smoke_s1.py
actually run: tiny, local, 0 compute cost.
"""
from dataclasses import dataclass, field

from photon.config import PhotonConfig

# TODO(Jun): confirm dataset ids + licensing before any production S1 run.
# Slide 30's mix -- Code 15% / Japanese 10% / Math 5% / EN-web the rest --
# needs real HF dataset_ids + pinned revisions + a license check; this repo
# does not pick them for you (no un-vetted dataset should end up baked into
# a training run silently). Shape of each entry once chosen:
#   dict(dataset_id="...", split="train", text_field="text",
#        revision="<pinned>", weight=0.15)
DATASET_MIX_TODO: list[dict] = []


@dataclass
class S1Config:
    # --- model (slide 16/28: PhotonConfig defaults, bf16 for the GPU trainer) ---
    photon: PhotonConfig = field(default_factory=lambda: PhotonConfig(
        C1=2, C2=2, R1=2, R2=2, split=(7, 7, 7, 7), rope_mode="reset",
        bypass=False, dtype="bfloat16"))

    # --- the crux (plan risks R-A1/R-A4) ---
    pool_mode: str = "mean"                       # {mean, last, strided} -- train/pooling.py
    group_weights: dict = field(default_factory=lambda: dict(
        enc1=1.0, enc2=1.0, dec2=1.0, dec1=1.0))
    train_all: bool = True                        # False = freeze enc1/enc2+chunkers, train dec+chunkers only (cheaper)

    # --- data (slide 30/31) ---
    dataset_mix: list = field(default_factory=lambda: list(DATASET_MIX_TODO))
    seq_len: int = 2048
    tokens_per_shard: int = 250_000_000            # ~1GB/shard at uint32 (slide 31)
    # local path today; swap for an `hf://datasets/...` or `gs://...` URI
    # once data/upload.py is actually run (needs Jun's credentials) -- the
    # reader (data/stream_dataset.py) only needs a local path for now.
    shard_dir: str = "data/_s1_shards"

    # --- optimizer / schedule (slide 33: bf16 + grad-checkpointing +
    # 8-bit Adam brings weights+optimizer to ~6GB, fits a 24GB L4) ---
    total_tokens: int = 500_000_000                # 0.5B (slide 28/35)
    # NOTE (2026-07-27, coordinator fix): grad accumulation is a NO-OP in
    # train_s1.train_step (it does one optimizer.step per micro-batch; run_s1
    # only used grad_accum_steps to shrink steps_total 8x -> the run trained
    # 1/8 the token budget). The A40 has ample memory (was 7/45GB at batch 2),
    # so fold the intended effective batch (2x8=16) into ONE micro-batch and
    # set accum=1: exact intended eff-batch 16, correct 500M-token budget,
    # no accumulation code path (which is buggy) exercised. steps_total then
    # = 500M/(16*2048*1) = 15258, consuming ~all 244k shard rows (one epoch).
    micro_batch_size: int = 16
    grad_accum_steps: int = 1
    lr: float = 2e-4
    use_8bit_adam: bool = False                    # 2026-07-27 (Jun): full AdamW — the A40 has ample memory
    #                                                (was ~23/45GB with 8-bit), so no need to trade optimizer
    #                                                precision for the ~3.6GB 8-bit saving that mattered only
    #                                                on the memory-constrained 24GB L4 target. (8-bit path
    #                                                already validated on the A40 smoke if the L4 run needs it.)
    use_grad_checkpointing: bool = True
    ckpt_every_seconds: float = 5 * 60             # 5 min (was 15): the A40 pod cgroup-OOM'd once at ~3.75h; tighter ckpt = less lost work per auto-resume
    ckpt_path: str = "checkpoints/s1_ckpt.pt"      # local; GCS shim TODO, see checkpoint/ckpt.py's own flag


@dataclass
class SmokeS1Config(S1Config):
    """tests/test_s1_smoke.py / scripts/smoke_s1.py -- tiny, local, 0 CU.
    T=128, B=2, ~50 steps (plan's smoke-test spec)."""
    photon: PhotonConfig = field(default_factory=lambda: PhotonConfig(
        C1=2, C2=2, R1=2, R2=2, split=(7, 7, 7, 7), rope_mode="reset",
        bypass=False, dtype="bfloat16"))
    seq_len: int = 128
    micro_batch_size: int = 2
    grad_accum_steps: int = 1
    total_tokens: int = 128 * 2 * 50               # ~50 steps at seq_len=128, micro_batch=2
    lr: float = 1.5e-4                             # a bit higher than production for a visible
    # per-block loss drop in only 50 steps; 1e-3 was tried first and diverged to NaN by step
    # ~25 with no gradient clipping (see train_s1.py's GRAD_CLIP_VALUE) -- both the LR cut and
    # clipping are mechanical fixes made during WSL/Blackwell smoke-test verification.
    #
    # 2026-07-26 A40 (cc 8.6) cloud-GPU validation: at the WSL-tuned 3e-4, enc1 oscillated
    # (initial=0.865 -> final=0.906, windowed mean over the smoke test's 50 steps, NOT falling)
    # while enc2/dec2/dec1 fell cleanly -- a genuine cross-GPU training-dynamics difference
    # (different cuBLAS/cuDNN reduction order -> different noise realization for this tiny,
    # 50-step, fixed-seed run), not a bug: the crux/gradient-isolation/bypass-exactness tests
    # (tests/test_s1_targets.py) are unaffected by this and stayed green throughout. Halving to
    # 1.5e-4 (mechanical fix, self-retry #1) makes all 4 groups fall cleanly and reproducibly on
    # the A40 (re-verified there; NOT re-tested on Blackwell, but halving a working LR only adds
    # margin, so it's expected to still fall there too). Worth a small decoder-vs-encoder LR split
    # or brief warmup on the paid
    # run if this sensitivity recurs at production scale -- flagged, not silently swept under.
    shard_dir: str = "data/_smoke_shards"          # built by scripts/build_smoke_shard.py, git-ignored
    ckpt_path: str = "checkpoints/_smoke/ckpt.pt"  # git-ignored
