"""S2 hyperparameters (build plan P1).

`S2Config` is the production 0.5B-token shape (build plan's "Phased build"
P1 + "Compute (0.5B on A40)") -- authored for scripts/run_s2.py, NOT run by
this codebase (paid-compute boundary, same as configs/s1.py's S1Config).

`SmokeS2Config` is what tests/test_s2_smoke.py and scripts/smoke_s2.py
actually run: tiny, local, 0 CU, fresh surgery (init_from=None -- doesn't
need the 3GB S1 checkpoint on disk).
"""
from dataclasses import dataclass, field

from photon.config import PhotonConfig


@dataclass
class S2Config:
    # --- model (same PhotonConfig shape as S1 -- S2 forks the S1 model,
    # it does not change the architecture) ---
    photon: PhotonConfig = field(default_factory=lambda: PhotonConfig(
        C1=2, C2=2, R1=2, R2=2, split=(7, 7, 7, 7), rope_mode="reset",
        bypass=False, dtype="bfloat16"))

    # --- the crux (train/s2_loss.py) ---
    # alpha_A/alpha_B kept splittable (plan: "Make alpha per-term-splittable")
    # rather than a single `alpha`; alpha_A + alpha_B ~= 0.3 (plan's alpha).
    alpha_A: float = 0.15
    alpha_B: float = 0.15

    # --- data (plan P3: REUSE the S1 shards verbatim, no new tokenization) ---
    seq_len: int = 2048
    shard_dir: str = "/workspace/data/s1_shards"

    # --- optimizer / schedule ---
    total_tokens: int = 500_000_000                # 0.5B, same budget as S1
    micro_batch_size: int = 16                      # eff-batch 16 (S1's corrected shape)
    grad_accum_steps: int = 1                       # train_step does one step per micro-batch
    lr: float = 5e-5                                # refining warm S1 weights -- lower than
    # S1's 2e-4 (plan: "lower than S1's 2e-4, sweepable"); S2 starts from a
    # trained checkpoint, not a cold surgically-grafted one.
    use_8bit_adam: bool = False                     # A40 has ample memory (S1 finding)
    use_grad_checkpointing: bool = True
    ckpt_every_seconds: float = 5 * 60              # 5 min, matches S1's tightened cadence
    ckpt_path: str = "checkpoints/s2_ckpt.pt"        # local; distinct from S1's ckpt_path

    # --- S2-specific: warm-start from the S1 checkpoint (plan P1's
    # `init_from=checkpoints/s1_final_ckpt.pt`) ---
    # `train/s2_step.py::build_state` seeds STUDENT WEIGHTS ONLY from this
    # path (`checkpoint.ckpt.load(init_from, photon, optimizer=None)`) --
    # the optimizer is built fresh at S2's own `lr` afterward, it is not
    # restored from the S1 run. None -> skip seeding (fresh surgery init;
    # what SmokeS2Config uses, since the smoke test has no business
    # depending on the 3GB S1 checkpoint being present on disk).
    init_from: str | None = "checkpoints/s1_final_ckpt.pt"


@dataclass
class SmokeS2Config(S2Config):
    """tests/test_s2_smoke.py / scripts/smoke_s2.py -- tiny, local, 0 CU.
    seq_len=128, micro_batch=2, ~50 steps (mirrors configs/s1.py's
    SmokeS1Config shape). init_from=None: fresh surgery, not a warm S1
    start -- the smoke test doesn't need the real 3GB S1 checkpoint on
    disk (see tests/test_s2_smoke.py's
    test_build_state_seeds_weights_from_init_from_when_file_exists for the
    init_from load path itself, which uses a tiny throwaway checkpoint
    instead)."""
    photon: PhotonConfig = field(default_factory=lambda: PhotonConfig(
        C1=2, C2=2, R1=2, R2=2, split=(7, 7, 7, 7), rope_mode="reset",
        bypass=False, dtype="bfloat16"))
    seq_len: int = 128
    micro_batch_size: int = 2
    grad_accum_steps: int = 1
    total_tokens: int = 128 * 2 * 50                # ~50 steps at seq_len=128, micro_batch=2
    lr: float = 5e-5
    shard_dir: str = "data/_smoke_shards"           # built by scripts/build_smoke_shard.py, git-ignored
    ckpt_path: str = "checkpoints/_smoke_s2/ckpt.pt"  # git-ignored
    init_from: str | None = None                    # fresh surgery, no S1 ckpt dependency
