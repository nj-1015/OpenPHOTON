"""S2 trainer (build plan P1): fork of train/train_s1.py's trainer shape,
but ONE composed forward (student's own chained forward, teacher-forcing
OFF) -> `train/s2_loss.l_s2` -> ONE `loss.backward()` -> a single global
`clip_grad_value_` (see GRAD_CLIP_VALUE's comment: the plan's default
`clip_grad_norm_` choice hit its own pre-authorized fallback trigger --
non-finite norms -- during the P2 cold-start smoke) -> one optimizer step.
S1 ran 4 decoupled sub-forward/backward passes per step; S2 collapses that
to end-to-end training of the full composed model, initialized from S1's
warm weights.

Teacher: same `harness.teacher.TeacherHiddenExtractor` as S1 (online, bf16,
eval/no_grad, frozen) -- only `H28` (pre-norm layers[27] output) is used
here; S2 doesn't need H7/H14/H21 (no decoupled per-group targets anymore).
Student: bf16, gradient-checkpointed per-layer, same
`train_s1.enable_gradient_checkpointing` reused verbatim.
Optimizer: `train_s1.build_optimizer` reused verbatim (AdamW by default,
8-bit Adam if `cfg.use_8bit_adam` and bitsandbytes is usable) -- built
FRESH at S2's own `cfg.lr`, never restored from the S1 checkpoint (only
STUDENT WEIGHTS are seeded from S1; see `build_state`'s `init_from` step).
Resume: checkpoint/ckpt.py, reused verbatim (model + optimizer state_dict +
DataloaderState) -- same as S1. There are two distinct load paths, do not
conflate them:
  - `build_state(cfg)` seeds STUDENT WEIGHTS ONLY from `cfg.init_from`
    (`ckpt.load(init_from, photon, optimizer=None)`) -- the S1->S2 warm
    start, once, at the beginning of a fresh S2 run.
  - `resume(cfg, path)` restores the FULL S2 trainer state (student
    weights + S2's OWN optimizer state + dataloader position) from an
    S2 checkpoint -- mid-run auto-resume after an interruption. It calls
    `build_state` first (which may redundantly apply `init_from` seeding,
    exactly as `train_s1.resume` does), then `ckpt.load` immediately
    overwrites the model/optimizer state with the S2 checkpoint's own --
    harmless, not a bug (same pattern S1's own `resume()` already uses).
"""
from dataclasses import dataclass, field

import torch

from checkpoint import ckpt
from data.stream_dataset import ShardStreamLoader
from harness.teacher import TeacherHiddenExtractor
from photon.surgery import load_photon_qwen3

from .s2_loss import l_s2
from .train_s1 import build_optimizer, enable_gradient_checkpointing

S2_LOSS_KEYS = ("final", "rec_A", "rec_B")

GRAD_CLIP_VALUE = 1.0
# Plan's DEFAULT choice was a single global `clip_grad_norm_(1.0)` ("S2 has
# one backward + warm S1 init -- no 1e19 cold-start spikes"), with an
# explicit pre-authorized fallback: "fall back to clip_grad_value_(1.0) if
# smoke shows non-finite norms". The P2 local smoke test (fresh-surgery
# init_from=None, cold-start, exactly the scenario the "warm S1 init"
# assumption above does NOT cover) hit precisely that: the pre-clip global
# grad norm measured `inf` (probed directly -- z0_dec1/z0_dec2's raw
# gradients overflow fp32 when SQUARED for the L2 norm, same root cause as
# train_s1.py's GRAD_CLIP_VALUE finding for S1's cold-start decoder groups).
# `clip_grad_norm_`'s single scale factor (`max_norm/total_norm`) then
# becomes `1.0/inf = 0`, silently zeroing EVERY parameter's gradient
# (including the perfectly-finite encoder/dec2 ones) -- the optimizer step
# becomes a complete no-op (confirmed: a 10x LR change produced
# bit-identical loss curves, and a direct weight snapshot showed zero
# change post-step). Switched to `clip_grad_value_` (elementwise clamp,
# never computes a norm, so one exploding group can't zero out every
# other group's real gradient) per the plan's own contingency -- this is
# strictly safer for BOTH the smoke's cold start AND a genuinely warm S1
# production start (a no-op if gradients are already small; S1's own
# smoke/production runs already validate this clipping strategy).


@dataclass
class TrainerState:
    photon: torch.nn.Module
    teacher: TeacherHiddenExtractor
    optimizer: torch.optim.Optimizer
    loader: ShardStreamLoader
    cfg: object
    step: int = 0
    loss_history: dict = field(default_factory=lambda: {k: [] for k in S2_LOSS_KEYS})


def build_state(cfg, device: str = "cuda") -> tuple[TrainerState, str]:
    photon, stock = load_photon_qwen3(cfg.photon)
    del stock  # photon holds shared references to all its parameters
    photon.to(device)
    photon.train()
    if cfg.use_grad_checkpointing:
        enable_gradient_checkpointing(photon)

    if cfg.init_from:
        # weights-only seed from the S1 checkpoint -- optimizer=None means
        # ckpt.load() skips restoring/optimizer-loading entirely (see
        # checkpoint/ckpt.py: "if optimizer is not None and not
        # isinstance(optimizer, dict)"), so this ONLY touches photon's
        # state_dict. The fresh optimizer below is then built at S2's own
        # (lower) lr, not S1's.
        ckpt.load(cfg.init_from, photon, optimizer=None)

    optimizer, opt_name = build_optimizer(photon, cfg)
    teacher = TeacherHiddenExtractor(dtype=cfg.photon.torch_dtype, device=device)
    loader = ShardStreamLoader(cfg.shard_dir, batch_size=cfg.micro_batch_size)

    state = TrainerState(photon=photon, teacher=teacher, optimizer=optimizer,
                          loader=loader, cfg=cfg)
    return state, opt_name


def train_step(state: TrainerState) -> dict:
    """One full optimizer step: ONE composed student forward (teacher-
    forcing OFF -- `photon(input_ids, return_intermediates=True)`) -> the
    frozen teacher's H28 on the same input_ids -> `l_s2` -> one
    `loss.backward()` -> one global `clip_grad_value_` (see
    GRAD_CLIP_VALUE's comment for why value-clip, not norm-clip) -> one
    `optimizer.step()`. Returns `{"final": loss, "rec_A": loss, "rec_B": loss}`
    (python floats, already `.item()`'d)."""
    photon, cfg = state.photon, state.cfg
    device = next(photon.parameters()).device

    input_ids = state.loader.next_batch().to(device)
    teacher_hs = state.teacher.capture(input_ids)
    H28 = teacher_hs["H28"]

    state.optimizer.zero_grad(set_to_none=True)
    _, mid = photon(input_ids, return_intermediates=True)
    total, parts = l_s2(mid, H28, photon.chunk2, alpha_A=cfg.alpha_A, alpha_B=cfg.alpha_B)
    total.backward()

    trainable_params = [p for p in photon.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_value_(trainable_params, clip_value=GRAD_CLIP_VALUE)
    state.optimizer.step()

    state.step += 1
    losses = {name: parts[name].item() for name in S2_LOSS_KEYS}
    for name in S2_LOSS_KEYS:
        state.loss_history[name].append(losses[name])
    return losses


def checkpoint(state: TrainerState, path: str) -> None:
    """Saves student model + optimizer + dataloader position + the training
    step counter (metadata dict, so resume restores `state.step` exactly
    rather than re-deriving it from the dataloader position). The teacher
    is deliberately NOT included (same rationale as train_s1.checkpoint --
    reload from the HF cache is deterministic)."""
    ckpt.save(path, state.photon, state.optimizer, state.loader.state(),
              metadata={"step": state.step, "stage": "S2"})


def resume(cfg, path: str, device: str = "cuda") -> tuple[TrainerState, str]:
    """Fresh reconstruction (new photon/teacher/optimizer/loader objects,
    per the S0 'simulated kill' discipline in
    tests/test_checkpoint_roundtrip.py) + ckpt.load() to restore student
    weights, optimizer state, dataloader position, and the step counter
    (from the checkpoint's metadata; falls back to 0 for pre-metadata
    checkpoints). NB: `build_state` may apply `cfg.init_from` seeding first
    (S1 warm start) -- `ckpt.load` below immediately overwrites that with
    the S2 checkpoint's own model/optimizer state, so this is correct
    (if slightly redundant), matching `train_s1.resume`'s own pattern."""
    state, opt_name = build_state(cfg, device=device)
    _, _, dl_state, metadata = ckpt.load(path, state.photon, state.optimizer)
    state.loader = ShardStreamLoader.from_state(cfg.shard_dir, cfg.micro_batch_size, dl_state)
    state.step = int(metadata.get("step", 0))
    return state, opt_name
