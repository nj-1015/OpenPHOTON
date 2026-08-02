"""P_k: [B, T, d] -> [B, T//k, d] pooling for the S1 teacher targets.

NOTE on file location: PHOTON_Qwen3_S1_plan.md's file list names this
`photon/pooling.py`. It lives under `train/` instead because the task's
hard constraint ("do NOT modify any file under photon/ -- S1 is new code
AROUND the frozen student/teacher") reads as "the photon/ package is the
S0-frozen surface," and pooling is purely an S1 training-time concept (the
frozen model never calls it). No behavior differs from the plan's design.

Plan risk R-A1: "P_k is a surrogate the deck never specifies." Made a
swept flag, default `mean` (matches modules.Chunker's identity-mean init,
so the enc1/enc2 groups start near-zero loss -- see s1_targets.py and the
plan's "why defensible" point 3).

    mean     -- average over each causal chunk of k consecutive positions.
    last     -- last position of each chunk (indices k-1, 2k-1, ...) --
                "the last token of a causal chunk is the teacher's own
                causal chunk-summary" (plan's primary alternative to sweep).
    strided  -- uniform decimation, first position of each chunk (indices
                0, k, 2k, ...) -- the plain subsampling baseline.
"""
import torch

MODES = ("mean", "last", "strided")


def pool(h: torch.Tensor, k: int, mode: str = "mean") -> torch.Tensor:
    """h: [B, T, d] -> [B, T // k, d]."""
    if k == 1:
        return h
    B, T, d = h.shape
    assert T % k == 0, f"pool: T={T} not divisible by k={k}"
    if mode == "mean":
        return h.reshape(B, T // k, k, d).mean(dim=2)
    if mode == "last":
        return h.reshape(B, T // k, k, d)[:, :, -1, :]
    if mode == "strided":
        return h[:, ::k, :]
    raise ValueError(f"unknown pooling mode {mode!r}, expected one of {MODES}")
