"""S3 hyperparameters -- logit / KL distillation stage.

THE OBJECTIVE SWITCH. S1 and S2 both trained the student to match the
teacher's HIDDEN STATES (per-block pooled targets in S1; the composed
final hidden H28 + reconstruction in S2). A probe after the S2 0.5B run
proved that path had saturated -- reconstruction cosine 0.986, yet
composed next-token PPL was still ~20x the teacher, and feeding the
decoder a LITERALLY PERFECT briefing did not help. The reason: the model
had never been trained to PREDICT TOKENS. S1/S2 optimized internal
activations against a full-attention teacher the compressed architecture
structurally cannot reproduce.

S3 switches the objective to matching the teacher's OUTPUT DISTRIBUTION --
forward KL D_KL(p_teacher || p_student) on the next-token logits. This is
the paper's actual training objective (they train the LM objective from
scratch; we distil Qwen3's logits into the same compressed hierarchy), and
it FREES the hidden states: h0_hat no longer has to equal H28, it only has
to yield the right token probabilities -- which the lm_head + softmax make
far more forgiving of internal representation differences.

`rec_weight` keeps a small RecGen fixed-point anchor (rec_B from S2,
verbatim) alive so inference-time reconstruction-substitution (the growing
KV reduction) does not drift once the hidden states are no longer pinned.

`S3Config` = production 0.5B-token trial shape (the same does-this-work
budget as the S2 trial); `SmokeS3Config` = tiny local / 0-CU smoke.
Warm-starts STUDENT WEIGHTS ONLY from the S2 final checkpoint.

MEMORY (why micro_batch is smaller than S2's 16): KD materializes TWO
full-vocab `[B, T, V=151936]` logit tensors -- the student's (grad-carrying,
from the tied lm_head) and the teacher's (transient, no_grad) -- plus fp32
log-softmax intermediates for the KL math. At seq_len=2048 that fp32
blow-up is the peak, so micro_batch=4 keeps the A40 comfortably under its
~45 GB cgroup ceiling. seq_len is PINNED at 2048: the s1_shards are
pre-chunked at 2048, so the loader yields 2048-token rows regardless of
this field, and steps_total = total_tokens // (micro_batch * seq_len) would
silently mis-budget if seq_len disagreed with the shard row length (an
S1-class accounting trap -- do NOT set this to anything but 2048 for these
shards). If a memory probe on the pod shows headroom, micro_batch can be
raised at launch (it does not affect the token budget, only steps_total).
"""
from dataclasses import dataclass, field

from photon.config import PhotonConfig


@dataclass
class S3Config:
    # --- model (unchanged PhotonConfig shape -- S3 forks the S2 model, it
    # does not alter the architecture) ---
    photon: PhotonConfig = field(default_factory=lambda: PhotonConfig(
        C1=2, C2=2, R1=2, R2=2, split=(7, 7, 7, 7), rope_mode="reset",
        bypass=False, dtype="bfloat16"))

    # --- KD objective (train/s3_loss.py) ---
    kd_temperature: float = 1.0     # forward KL softmax temperature (1.0 = plain KL)
    ce_weight: float = 0.0          # optional CE-on-gold-token anchor; 0.0 = pure distillation
    rec_weight: float = 0.1         # RecGen fixed-point anchor (rec_B from s2_loss) -- keeps
    # chunk2(h1_hat) ~= h2 so inference-time reconstruction-substitution stays
    # consistent once the hidden states are freed. Do NOT drop to 0 unless you
    # have decided to abandon RecGen KV reduction.

    # --- data (REUSE the S1/S2 shards -- the KD targets are the teacher's
    # logits ON these tokens, which the student has never had distilled into
    # it, so reuse is a valid fresh signal; no re-tokenization) ---
    seq_len: int = 2048             # PINNED to the shard row length (see module docstring)
    shard_dir: str = "/workspace/data/s1_shards"

    # --- optimizer / schedule ---
    total_tokens: int = 500_000_000                 # 0.5B trial (same budget as the S2 trial)
    micro_batch_size: int = 4                       # KD memory ceiling (see module docstring)
    grad_accum_steps: int = 1                       # one optimizer.step per micro-batch; NO
    # token-budget division (steps_total = total_tokens // (micro_batch * seq_len)).
    lr: float = 3e-5                                # refining a warm S2 init -- lower than S2's 5e-5
    use_8bit_adam: bool = False
    use_grad_checkpointing: bool = True
    ckpt_every_seconds: float = 5 * 60
    ckpt_path: str = "checkpoints/s3_ckpt.pt"

    # --- warm-start: STUDENT WEIGHTS ONLY from the S2 final checkpoint
    # (train/s3_step.build_state seeds via ckpt.load(init_from, photon,
    # optimizer=None); the optimizer is built fresh at S3's lr). On the pod
    # the run script passes --init-from /workspace/checkpoints/s2/ckpt.pt. ---
    init_from: str | None = "checkpoints/s2_final_ckpt.pt"


@dataclass
class SmokeS3Config(S3Config):
    """scripts/smoke_s3.py / tests/test_s3_smoke.py -- tiny, local, 0 CU.
    seq_len=128, micro_batch=2, ~50 steps. init_from=None: fresh surgery,
    no dependency on the 3GB S2 checkpoint (the init_from load path itself
    is covered by tests/test_s3_smoke.py with a tiny throwaway checkpoint)."""
    photon: PhotonConfig = field(default_factory=lambda: PhotonConfig(
        C1=2, C2=2, R1=2, R2=2, split=(7, 7, 7, 7), rope_mode="reset",
        bypass=False, dtype="bfloat16"))
    seq_len: int = 128
    micro_batch_size: int = 2
    grad_accum_steps: int = 1
    total_tokens: int = 128 * 2 * 50                # ~50 steps at seq_len=128, micro_batch=2
    lr: float = 3e-5
    shard_dir: str = "data/_smoke_shards"           # built by scripts/build_smoke_shard.py, git-ignored
    ckpt_path: str = "checkpoints/_smoke_s3/ckpt.pt"  # git-ignored
    init_from: str | None = None                    # fresh surgery, no S2 ckpt dependency
