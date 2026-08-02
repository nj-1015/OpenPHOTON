"""THE S1 crux, in code: pooled-teacher targets + decoupled teacher forcing
(PHOTON_Qwen3_S1_plan.md's crux table, reproduced here):

| group | teacher-forced INPUT (detached, teacher-derived) | TARGET      | trains                        |
|-------|---------------------------------------------------|-------------|--------------------------------|
| enc1  | chunk1(H0)                                         | P_C1(H7)    | chunk1 + layers 0-6            |
| enc2  | chunk2(P_C1(H7))                                    | P_C1C2(H14) | chunk2 + layers 7-13           |
| dec2  | pack(conv2(shift(P_C1C2(H14))), P_C1(H7))           | P_C1(H21)   | conv2 + layers 14-20 + z0_dec2 |
| dec1  | pack(conv1(shift(P_C1(H21))), H0)                   | H28         | conv1 + layers 21-27 + z0_dec1 |

(`P_k` default mode "mean" -- train/pooling.py. `k` is derived from
cfg.C1/cfg.C2 rather than hardcoded 2/4, so sweeping C1/C2 doesn't require
touching this file. With cfg.bypass=True and C1=C2=R1=R2=1, `P_1 == id`,
chunk*/conv* are identity, and this degenerates to replaying the teacher's
own layers on its own hidden states -- see tests/test_s1_targets.py's
bypass-exactness test, the "sane at init" sanity check for this file.)

Every group is DECOUPLED: its input is built purely from the frozen
teacher's captured `{H0,H7,H14,H21,H28}` (harness/teacher.py) -- never from
another group's output or the student's own running activation. That's
what makes each of the 4 returned closures a fully independent computation
graph the trainer can forward + backward + free one at a time (the reason
S1's live activation footprint is one 7-layer subgraph, not the whole
model -- see the plan's Trainer section and "why defensible" point 1,
MOHAWK-stage-2-faithful).

Every group reuses model.py's own `_run_stage` (identical mask/position-id
construction to what model.py's forward already does for that stage) and
photon.reshape.shift/pack/take_last_C VERBATIM/unmodified -- nothing in
photon/ is touched or reimplemented here, only called from outside it.
"""
from typing import Callable

import torch

from photon import masks, reshape, rope
from photon.config import PhotonConfig
from photon.model import PhotonQwen3ForCausalLM

from .pooling import pool

GroupForward = Callable[[], tuple[torch.Tensor, torch.Tensor]]  # () -> (output, target)
GROUP_ORDER = ("enc1", "enc2", "dec2", "dec1")


def _stage_mask_and_positions(z: torch.Tensor, L: int, B: int, cfg: PhotonConfig, device):
    """The local-decoder mask/position construction model.py's forward
    uses for dec1/dec2 (bypass -> full causal/absolute; else -> local
    causal(R+C)/reset), reused verbatim here for the decoupled sub-forward."""
    if cfg.bypass:
        pos = rope.absolute_position_ids(L, B, device=device)
        mask = masks.full_causal_mask(L, dtype=z.dtype, device=device)
    else:
        pos = rope.decoder_position_ids(cfg.rope_mode, L, z.shape[0], device=device)
        mask = masks.local_causal_mask(L, dtype=z.dtype, device=device)
    return mask, pos


def build_group_forwards(photon: PhotonQwen3ForCausalLM, cfg: PhotonConfig,
                          teacher_hs: dict, pool_mode: str = "mean") -> dict[str, GroupForward]:
    """teacher_hs: {"H0","H7","H14","H21","H28"} -> [B,T,d], detached
    (harness.teacher.TeacherHiddenExtractor.capture()'s return value).

    Returns 4 zero-arg closures keyed by GROUP_ORDER. Calling one runs
    THAT group's sub-forward immediately and returns `(output, target)` --
    nothing is forward()'d until called, so the trainer controls exactly
    one group's graph being live (and backward-able, then freeable) at a
    time.
    """
    H0, H7, H14, H21, H28 = (teacher_hs[k] for k in ("H0", "H7", "H14", "H21", "H28"))
    device = H0.device
    B = H0.shape[0]

    # shared pooled teacher tensors -- each is one group's TARGET and the
    # next group's INPUT-source (see the table above), computed once.
    P_C1_H7 = pool(H7, cfg.C1, pool_mode)                # -> h1 grid   [B, T/C1, d]
    P_C1C2_H14 = pool(H14, cfg.C1 * cfg.C2, pool_mode)   # -> h2 grid   [B, T/(C1*C2), d]
    P_C1_H21 = pool(H21, cfg.C1, pool_mode)              # -> h1_hat grid [B, T/C1, d]

    def run_enc1():
        x = photon.chunk1(H0)
        M = x.shape[1]
        pos = rope.absolute_position_ids(M, B, device=device)
        mask = masks.full_causal_mask(M, dtype=x.dtype, device=device)
        out = photon._run_stage(photon.enc1, x, mask, pos)
        return out, P_C1_H7

    def run_enc2():
        x = photon.chunk2(P_C1_H7)
        M = x.shape[1]
        pos = rope.absolute_position_ids(M, B, device=device)
        mask = masks.full_causal_mask(M, dtype=x.dtype, device=device)
        out = photon._run_stage(photon.enc2, x, mask, pos)
        return out, P_C1C2_H14

    def run_dec2():
        u = photon.conv2(reshape.shift(P_C1C2_H14, photon.z0_dec2, bypass=cfg.bypass))
        z = reshape.pack(u, P_C1_H7, bypass=cfg.bypass)
        L = z.shape[1] if cfg.bypass else cfg.R2 + cfg.C2
        mask, pos = _stage_mask_and_positions(z, L, B, cfg, device)
        z_out = photon._run_stage(photon.dec2, z, mask, pos)
        out = reshape.take_last_C(z_out, cfg.C2, B=B, bypass=cfg.bypass)
        return out, P_C1_H21

    def run_dec1():
        u = photon.conv1(reshape.shift(P_C1_H21, photon.z0_dec1, bypass=cfg.bypass))
        z = reshape.pack(u, H0, bypass=cfg.bypass)
        L = z.shape[1] if cfg.bypass else cfg.R1 + cfg.C1
        mask, pos = _stage_mask_and_positions(z, L, B, cfg, device)
        z_out = photon._run_stage(photon.dec1, z, mask, pos)
        out = reshape.take_last_C(z_out, cfg.C1, B=B, bypass=cfg.bypass)
        # H28 is PRE-norm (harness/teacher.py's module docstring) -- matches
        # `out`/h0_hat, which is also pre-norm (model.py applies norm once,
        # at the final lm_head step). Never swap this for capture_reference
        # .py's post-norm hs[28] -- S2 must keep this same convention.
        return out, H28

    return {"enc1": run_enc1, "enc2": run_enc2, "dec2": run_dec2, "dec1": run_dec1}
