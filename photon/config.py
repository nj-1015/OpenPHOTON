"""PhotonConfig — hyperparameters for the PHOTON-Qwen3 surgery/forward.

Kept as a plain dataclass (lift the *shape* of a typical HF hybrid-model
config -- dataclass fields, no framework base class -- not any particular
architecture's fields; PHOTON's own hyperparameters are defined fresh below).
"""
from dataclasses import dataclass

import torch


@dataclass
class PhotonConfig:
    # compression ratios: chunker C1 folds tokens -> level-1 latents,
    # chunker C2 folds level-1 latents -> level-2 (global) latents.
    C1: int = 2
    C2: int = 2
    # converter widths (briefing vectors emitted per parent latent). The
    # PHOTON paper never states R_l numerically (slide 15/16) -- sweep in S1.
    # R1 feeds dec1's local window, R2 feeds dec2's.
    R1: int = 2
    R2: int = 2

    # how the 28 stock Qwen3 layers are split across the four stages, in
    # order: enc1, enc2, dec2, dec1. Must sum to num_hidden_layers (28).
    split: tuple[int, int, int, int] = (7, 7, 7, 7)

    # RoPE position handling for the local decoders (dec1, dec2).
    #   "reset" — every chunk sees positions 0..R+C-1 (parallel-safe, the
    #             S0 default per slide 22).
    #   "carry" — chunk sees its true absolute offset. M6 scope; stubbed
    #             (raises NotImplementedError) until then.
    rope_mode: str = "reset"

    # bypass=True switches shift/pack/take_last_C to pass-through and the
    # decoder stages to a full-causal-over-T / absolute-position mask —
    # the M5 golden-test configuration. Requires C1=C2=R1=R2=1.
    bypass: bool = False

    # compute dtype -- string (not torch.dtype) so the config stays a plain,
    # trivially-serialisable dataclass. Resolved via `torch_dtype` below and
    # wired through as the default in surgery.load_photon_qwen3() (the
    # single source of truth for what precision a run loads at). S0 only
    # exercises "float32" (CPU); the M5 bf16/L4 leg will just set this to
    # "bfloat16" -- no other code changes needed.
    dtype: str = "float32"

    def __post_init__(self):
        if sum(self.split) != 28:
            raise ValueError(f"split {self.split} must sum to 28, got {sum(self.split)}")
        if self.rope_mode not in ("reset", "carry"):
            raise ValueError(f"rope_mode must be 'reset' or 'carry', got {self.rope_mode!r}")
        if self.bypass and not (self.C1 == self.C2 == self.R1 == self.R2 == 1):
            raise ValueError("bypass=True requires C1=C2=R1=R2=1")
        if not hasattr(torch, self.dtype):
            raise ValueError(f"dtype {self.dtype!r} is not a torch dtype name")

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype)
