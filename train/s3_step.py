"""S3 trainer -- logit/KL distillation. Fork of train/s2_step.py's shape,
but the loss is `train/s3_loss.l_s3` (forward KL to the teacher's output
distribution + optional CE + a small rec_B anchor) instead of the S2
hidden-matching composed loss.

Per step: ONE composed student forward (`photon(input_ids,
return_intermediates=True)` -- logits AND the intermediates rec_B needs) ->
the frozen teacher's next-token LOGITS on the same input_ids (NOT its
hidden states -- S3 doesn't use H28 at all) -> `l_s3` -> one
`loss.backward()` -> one global `clip_grad_value_` -> one optimizer step.

Two distinct checkpoint load paths, same as S2 (do not conflate):
  - `build_state(cfg)` seeds STUDENT WEIGHTS ONLY from `cfg.init_from` (the
    S2->S3 warm start), once, on a fresh run.
  - `resume(cfg, path)` restores the FULL S3 trainer state (weights +
    S3's own optimizer state + dataloader position) from an S3 checkpoint.

Grad clip: `clip_grad_value_(1.0)` (elementwise), same choice and rationale
as train/s2_step.py -- a single global norm-clip can be silently zeroed to
a no-op by one non-finite group, so value-clip is strictly safer for both a
cold smoke and a warm production start.
"""
from dataclasses import dataclass, field

import torch

from checkpoint import ckpt
from data.stream_dataset import ShardStreamLoader
from harness.teacher import TeacherHiddenExtractor
from photon.surgery import load_photon_qwen3

from .s3_loss import S3_LOSS_KEYS, l_s3
from .train_s1 import build_optimizer, enable_gradient_checkpointing

GRAD_CLIP_VALUE = 1.0
# See train/s2_step.py::GRAD_CLIP_VALUE for the full derivation: a single
# global clip_grad_norm_ becomes 1.0/inf = 0 if any group's grad is
# non-finite (z0_* cold-start overflow when squared for the L2 norm),
# silently zeroing EVERY parameter's gradient and no-op'ing the step.
# clip_grad_value_ (elementwise clamp, never computes a norm) can't do that,
# so it is used here too -- a no-op when gradients are already small (the
# warm S2-init production case), a safety net for the cold smoke.


@dataclass
class TrainerState:
    photon: torch.nn.Module
    teacher: TeacherHiddenExtractor
    optimizer: torch.optim.Optimizer
    loader: ShardStreamLoader
    cfg: object
    step: int = 0
    loss_history: dict = field(default_factory=lambda: {k: [] for k in S3_LOSS_KEYS})


def build_state(cfg, device: str = "cuda") -> tuple[TrainerState, str]:
    photon, stock = load_photon_qwen3(cfg.photon)
    del stock  # photon holds shared references to all its parameters
    photon.to(device)
    photon.train()
    if cfg.use_grad_checkpointing:
        enable_gradient_checkpointing(photon)

    if cfg.init_from:
        # weights-only seed from the S2 checkpoint (optimizer=None -> ckpt.load
        # skips optimizer restore; only photon's state_dict is touched). The
        # fresh optimizer below is built at S3's own (lower) lr.
        ckpt.load(cfg.init_from, photon, optimizer=None)

    optimizer, opt_name = build_optimizer(photon, cfg)
    teacher = TeacherHiddenExtractor(dtype=cfg.photon.torch_dtype, device=device)
    loader = ShardStreamLoader(cfg.shard_dir, batch_size=cfg.micro_batch_size)

    state = TrainerState(photon=photon, teacher=teacher, optimizer=optimizer,
                          loader=loader, cfg=cfg)
    return state, opt_name


def train_step(state: TrainerState) -> dict:
    """One full optimizer step (see module docstring). Returns
    `{"kd": ..., "ce": ..., "rec_B": ...}` (python floats, already .item()'d)."""
    photon, cfg = state.photon, state.cfg
    device = next(photon.parameters()).device

    input_ids = state.loader.next_batch().to(device)
    teacher_logits = state.teacher.logits(input_ids)     # [B, T, V], detached, no_grad

    state.optimizer.zero_grad(set_to_none=True)
    student_logits, mid = photon(input_ids, return_intermediates=True)
    total, parts = l_s3(student_logits, teacher_logits, mid, photon.chunk2,
                        gold_ids=input_ids, kd_temperature=cfg.kd_temperature,
                        ce_weight=cfg.ce_weight, rec_weight=cfg.rec_weight)
    total.backward()

    trainable_params = [p for p in photon.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_value_(trainable_params, clip_value=GRAD_CLIP_VALUE)
    state.optimizer.step()

    state.step += 1
    losses = {name: parts[name].item() for name in S3_LOSS_KEYS}
    for name in S3_LOSS_KEYS:
        state.loss_history[name].append(losses[name])
    return losses


def checkpoint(state: TrainerState, path: str) -> None:
    """Student model + optimizer + dataloader position (teacher excluded --
    deterministic reload from the HF cache, same as S1/S2)."""
    ckpt.save(path, state.photon, state.optimizer, state.loader.state())


def resume(cfg, path: str, device: str = "cuda") -> tuple[TrainerState, str]:
    """Fresh reconstruction + ckpt.load() to restore student weights,
    optimizer state, and dataloader position from an S3 checkpoint. NB:
    build_state may apply cfg.init_from (S2 warm start) first -- ckpt.load
    below immediately overwrites it with the S3 checkpoint's own state, so
    this is correct (if slightly redundant), matching train/s2_step.resume."""
    state, opt_name = build_state(cfg, device=device)
    _, _, dl_state = ckpt.load(path, state.photon, state.optimizer)
    state.loader = ShardStreamLoader.from_state(cfg.shard_dir, cfg.micro_batch_size, dl_state)
    return state, opt_name
