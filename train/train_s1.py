"""S1 trainer (plan P3): per batch, one frozen-teacher forward -> build the
4 decoupled (input, target) pairs (train/s1_targets.py) -> 4 independent
sub-forward/backward passes (one 7-layer subgraph live at a time) -> a
single optimizer step accumulating all 4 groups' gradients.

Teacher: online, bf16, eval/no_grad, frozen -- never checkpointed (reload
from the HF cache is deterministic; see checkpoint()/resume() below, which
only ever touch the STUDENT + optimizer + dataloader state).
Student: bf16, gradient-checkpointed per-layer within each 7-layer group
(functools.partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
assigned to each Qwen3DecoderLayer's `_gradient_checkpointing_func` --
transformers' own GradientCheckpointingLayer.__call__ already knows how to
use it, see modeling_layers.py; nothing in photon/ is modified to enable
this, it's configured on the live layer objects from here).
Optimizer: 8-bit Adam (bitsandbytes) by default, falling back to AdamW if
bitsandbytes/its CUDA kernels aren't usable on this machine (plan's flagged
known-unknown for Blackwell/WSL cu128 -- see the fallback's log line).
Resume: checkpoint/ckpt.py, reused verbatim (model + optimizer state_dict +
DataloaderState).
"""
import functools
from dataclasses import dataclass, field

import torch

from checkpoint import ckpt
from checkpoint.ckpt import DataloaderState
from data.stream_dataset import ShardStreamLoader
from harness.teacher import TeacherHiddenExtractor
from photon.surgery import load_photon_qwen3

from .s1_loss import group_loss
from .s1_targets import GROUP_ORDER, build_group_forwards


@dataclass
class TrainerState:
    photon: torch.nn.Module
    teacher: TeacherHiddenExtractor
    optimizer: torch.optim.Optimizer
    loader: ShardStreamLoader
    cfg: object
    step: int = 0
    loss_history: dict = field(default_factory=lambda: {k: [] for k in GROUP_ORDER})


def build_optimizer(photon, cfg):
    """Returns (optimizer, name). Tries bitsandbytes 8-bit Adam first
    (cfg.use_8bit_adam); falls back to AdamW and logs loudly if
    bitsandbytes is missing or its CUDA kernels don't actually run on this
    machine -- 8-bit Adam on Blackwell/WSL cu128 is a plan-flagged known
    unknown, not something to silently paper over."""
    params = [p for p in photon.parameters() if p.requires_grad]
    if cfg.use_8bit_adam:
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.Adam8bit(params, lr=cfg.lr)
            # cheap live kernel check, not just a successful import -- a
            # broken bnb CUDA build can import fine and only fail at
            # .step() time.
            probe = torch.nn.Linear(8, 8).to(next(photon.parameters()).device, dtype=torch.bfloat16)
            probe_opt = bnb.optim.Adam8bit(probe.parameters(), lr=1e-3)
            probe(torch.randn(2, 8, device=probe.weight.device, dtype=torch.bfloat16)).sum().backward()
            probe_opt.step()
            del probe, probe_opt
            return optimizer, "bitsandbytes.optim.Adam8bit"
        except Exception as e:  # pragma: no cover -- exercised only when bnb truly is broken
            print(f"[train_s1] 8-bit Adam unavailable ({type(e).__name__}: {e}) -- "
                  f"falling back to torch.optim.AdamW. 8-bit Adam is only exercised "
                  f"for real on the paid A100 run if this fallback ever fires locally.")
    return torch.optim.AdamW(params, lr=cfg.lr), "torch.optim.AdamW"


def enable_gradient_checkpointing(photon) -> None:
    ckpt_fn = functools.partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
    for stage in (photon.enc1, photon.enc2, photon.dec2, photon.dec1):
        for layer in stage:
            layer.gradient_checkpointing = True
            layer._gradient_checkpointing_func = ckpt_fn


def freeze_encoder_if_configured(photon, cfg) -> None:
    """plan risk R-A4: train_all=False freezes chunk1/enc1/chunk2/enc2,
    training only conv1/conv2/dec2/dec1/z0s -- the cheaper alternative."""
    if cfg.train_all:
        return
    for module in (photon.chunk1, photon.enc1, photon.chunk2, photon.enc2):
        for p in module.parameters():
            p.requires_grad_(False)


def build_state(cfg, device: str = "cuda") -> tuple[TrainerState, str]:
    photon, stock = load_photon_qwen3(cfg.photon)
    del stock  # photon holds shared references to all its parameters
    photon.to(device)
    photon.train()
    freeze_encoder_if_configured(photon, cfg)
    if cfg.use_grad_checkpointing:
        enable_gradient_checkpointing(photon)

    optimizer, opt_name = build_optimizer(photon, cfg)
    teacher = TeacherHiddenExtractor(dtype=cfg.photon.torch_dtype, device=device)
    loader = ShardStreamLoader(cfg.shard_dir, batch_size=cfg.micro_batch_size)

    state = TrainerState(photon=photon, teacher=teacher, optimizer=optimizer,
                          loader=loader, cfg=cfg)
    return state, opt_name


GRAD_CLIP_VALUE = 1.0
# Per-group, per-element gradient clipping (torch.nn.utils.clip_grad_value_,
# not clip_grad_norm_) -- a smoke-test-verification finding, not a
# crux/architecture change:
#
# dec1/dec2's cold-start local-decoder layers (real, pretrained Qwen3
# layers 14-27, fed a synthetic teacher-forced [briefing;children] window
# at reset positions -- genuinely out of the distribution those layers
# were pretrained on, exactly the "why fine-tuning fails" gap the plan's
# slide 27 names) produce astronomically large-but-FINITE gradients at
# init on z0_dec1/z0_dec2 (measured ~1e19, growing ~1e3-1e4x per layer
# across the 7-layer backward). That's still finite in fp32/bf16 alone,
# but squaring it to compute an L2 norm (clip_grad_norm_'s method)
# overflows fp32/bf16's ~3.4e38 max, turning a large-but-real gradient
# into a literal `inf`. Two prior GLOBAL clip_grad_norm_ calls (across all
# 4 groups' params at once) then propagated that single inf into every
# OTHER group's gradient too (norm-of-a-stack-containing-inf is inf, and
# max_norm/inf -> 0, 0*inf -> nan) -- which is what made enc1/enc2's loss
# curves erratic even though their own gradients were perfectly finite.
# clip_grad_value_ sidesteps the squaring-overflow entirely (elementwise
# clamp, no norm computed) and is applied separately after EACH group's
# own backward (while only that group's grad is populated -- zero_grad
# reset everything to None at the top of the step), so one group's
# instability can no longer contaminate another's update.
#
# This does not "solve" the underlying ill-conditioning (that's the S1
# distillation problem itself -- refinement, not a solved system, per the
# plan) -- it keeps the optimizer step numerically sane while that
# refinement happens. Flagged in the close-out report as a real training-
# dynamics finding worth Jun's attention (e.g. a decoder-group warmup or
# a smaller decoder LR may be worth sweeping on the paid run), not
# something papered over silently.


def train_step(state: TrainerState) -> dict:
    """One full optimizer step: teacher forward -> build the 4 decoupled
    groups -> forward+backward+clip each group in turn (accumulating
    gradient into the shared student parameters) -> a single
    optimizer.step(). Returns {"enc1": loss, "enc2": loss, "dec2": loss,
    "dec1": loss} (python floats, already .item()'d)."""
    photon, cfg = state.photon, state.cfg
    device = next(photon.parameters()).device

    input_ids = state.loader.next_batch().to(device)
    teacher_hs = state.teacher.capture(input_ids)
    groups = build_group_forwards(photon, cfg.photon, teacher_hs, pool_mode=cfg.pool_mode)
    trainable_params = [p for p in photon.parameters() if p.requires_grad]

    state.optimizer.zero_grad(set_to_none=True)
    losses = {}
    for name in GROUP_ORDER:
        out, target = groups[name]()
        loss = group_loss(out, target, weight=cfg.group_weights[name])
        loss.backward()
        # clip THIS group's gradient now, before the next group's backward
        # runs -- see GRAD_CLIP_VALUE's comment for why this must be
        # per-group and value-based, not one global norm-based call.
        torch.nn.utils.clip_grad_value_(trainable_params, clip_value=GRAD_CLIP_VALUE)
        losses[name] = loss.item()
    state.optimizer.step()

    state.step += 1
    for name in GROUP_ORDER:
        state.loss_history[name].append(losses[name])
    return losses


def checkpoint(state: TrainerState, path: str) -> None:
    """Saves student model + optimizer + dataloader position. The teacher
    is deliberately NOT included (plan: reload from the HF cache is
    deterministic -- no point paying checkpoint size/time for a frozen,
    reproducible-from-scratch model)."""
    ckpt.save(path, state.photon, state.optimizer, state.loader.state())


def resume(cfg, path: str, device: str = "cuda") -> tuple[TrainerState, str]:
    """Fresh reconstruction (new photon/teacher/optimizer/loader objects,
    per the S0 'simulated kill' discipline in
    tests/test_checkpoint_roundtrip.py -- not a save-then-immediately-load
    round trip) + ckpt.load() to restore student weights, optimizer state,
    and dataloader position."""
    state, opt_name = build_state(cfg, device=device)
    _, _, dl_state = ckpt.load(path, state.photon, state.optimizer)
    state.loader = ShardStreamLoader.from_state(cfg.shard_dir, cfg.micro_batch_size, dl_state)
    return state, opt_name
