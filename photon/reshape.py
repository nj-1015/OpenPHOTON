"""shift(), pack(), take_last_C() -- the batch-dim surgery (slide 21/22).

shift() is the #1 S0 risk (S0 plan, risk list item 1): chunk g must condition
on the *parent* latent at index g-1, never its own -- z0 is a learned start
latent standing in for g=0's (nonexistent) parent. Get this off by one and
the model trivially cheats (sees its own parent), loss collapses, and
nothing else measured means anything. tests/test_shift.py is the isolated
gate for this file.

bypass=True (the M5 golden-test configuration, C=1/R=1 everywhere) turns all
three functions into pass-throughs:
  - shift:       no offset -- return the parent latent unchanged.
  - pack:        no [briefing;children] concat / batch-dim move -- with
                 R=1 the single briefing vector *is* exactly the running
                 hidden state (shift+converter are both identity), so
                 dropping x and returning u squeezed of its R=1 axis
                 continues the stream at [B, T, d] instead of routing
                 through a [B*chunks, R+C, d] detour.
  - take_last_C: nothing was packed into chunk-batch rows, so nothing needs
                 to be un-packed -- return the tensor unchanged.
Together with model.py's bypass branch for mask/position_ids (full causal
over T, absolute positions, instead of local causal over R+C, reset), this
makes the general slide-21 forward reduce to running all 28 stock layers in
order on one [B, T, d] stream -- i.e. stock Qwen3 -- so the golden test
exercises the *same* orchestration code path used at real (C, R) values.
"""
import torch


def shift(h: torch.Tensor, z0: torch.Tensor, bypass: bool = False) -> torch.Tensor:
    """h: [B, M, d] parent latents, one per chunk. Returns the "briefing
    parent" seen by chunk g: z0 at g=0, h[g-1] for g>=1.
    """
    if bypass:
        return h
    B, M, d = h.shape
    z0_exp = z0.expand(B, 1, d).to(dtype=h.dtype, device=h.device)
    return torch.cat([z0_exp, h[:, :-1, :]], dim=1)


def pack(u: torch.Tensor, x: torch.Tensor, bypass: bool = False) -> torch.Tensor:
    """u: [B, M, R, d] briefing conditioning per chunk (already shift+conv'd).
    x: [B, M*C, d] children (the teacher-forced encoder-side skip input).
    Normal: interleave [briefing ; children] per chunk and move the chunk
    index into the batch dim -> [B*M, R+C, d].
    bypass: ignore x, drop the packing entirely -- return u with its R=1
    axis squeezed, i.e. [B, M, d] (M == T when C == 1).
    """
    if bypass:
        assert u.shape[2] == 1, "bypass pack() requires R == 1"
        return u.squeeze(2)
    B, M, R, d = u.shape
    _, MC, _ = x.shape
    C = MC // M
    assert M * C == MC, f"children length {MC} not divisible into {M} chunks"
    x_r = x.reshape(B, M, C, d)
    seq = torch.cat([u, x_r], dim=2)          # [B, M, R+C, d]
    return seq.reshape(B * M, R + C, d)


def take_last_C(z: torch.Tensor, C: int, B: int | None = None, bypass: bool = False) -> torch.Tensor:
    """z: [B*M, R+C, d] (normal) or [B, T, d] (bypass, already un-packed).
    Selects the last C rows per chunk-batch-row (the "children" outputs --
    the briefing rows were only computed to be attended to, then discarded)
    and reshapes chunks back out of the batch dim -> [B, M*C, d].
    """
    if bypass:
        return z
    if B is None:
        raise ValueError("take_last_C requires B (original batch size) when bypass=False")
    BM, RC, d = z.shape
    last = z[:, -C:, :]                        # [B*M, C, d]
    M = BM // B
    return last.reshape(B, M * C, d)
