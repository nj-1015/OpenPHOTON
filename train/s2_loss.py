"""S2 crux loss (build plan's "THE CRUX -- composed forward + L_rec"):

    L_S2 = relative_mse(h0_hat, H28)  +  alpha_A * rec_A  +  alpha_B * rec_B
    rec_A = 1 - cos(h1_hat,          h1.detach())   -- level-1 recon (droppable, partly redundant)
    rec_B = 1 - cos(chunk2(h1_hat),  h2.detach())   -- the RecGen fixed point (ESSENTIAL)

All four tensors (`h0_hat`, `h1`, `h2`, `h1_hat`) come from ONE composed
`photon(input_ids, return_intermediates=True)` forward (the student's own
chained forward -- teacher-forcing OFF, unlike S1's decoupled groups).
`H28` comes from `harness/teacher.py`'s frozen-teacher capture on the same
`input_ids` (already detached there, `@torch.no_grad()`).

**Final term** (load-bearing convention, do not change): `h0_hat` is dec1's
PRE-norm output (`photon/model.py` applies `self.norm` exactly once, at the
final `lm_head` step) and `H28` is `layers[27]`'s PRE-norm forward-hook
output (`harness/teacher.py`'s locked convention, `_verify_h28_is_pre_norm`)
-- the SAME (output, target) pair S1's dec1 group trained; only the INPUT
path changes (student's own chained `h1_hat`/`h0`, not teacher-forced).
Uses `s1_loss.relative_mse`, NOT raw MSE: `H28`'s norm is large (a real,
pretrained-model activation), so raw MSE would make `alpha * L_rec`
numerically invisible next to it and L_rec would never train.

**Detach table** (reviewer: scrutinize):

| term                                   | grad-carrying        | detached |
|-----------------------------------------|-----------------------|----------|
| `relative_mse(h0_hat, H28)` ("final")    | full composed graph (dec1<-dec2<-enc2<-enc1<-chunk1<-embed) | H28 (teacher, already detached) |
| rec_A `1 - cos(h1_hat, h1.detach())`     | `h1_hat`              | `h1` (student's own encoder output, used only as this term's REFERENCE) |
| rec_B `1 - cos(chunk2(h1_hat), h2.detach())` | `chunk2(h1_hat)`  | `h2` (student's own encoder output, used only as this term's REFERENCE) |

`h1`/`h2` are NOT detached globally -- they still carry gradient through
their OTHER use in the composed forward (`h1` feeds `chunk2`/`enc2` and
dec2's skip connection; `h2` feeds dec2's briefing) via the "final" term.
`.detach()` here only strips gradient from the copy used as this loss
term's reference target -- it stops the "target-side cheat" (the encoder
moving `h1`/`h2` to match `h1_hat`/`chunk2(h1_hat)` instead of the
reconstruction path matching the encoder), NOT the entire encoder-training
signal. The encoder is trained via the "final" term's full composed-graph
backward, AS INTENDED -- but see the empirically-verified finding below,
L_rec also (partially, secondarily) reaches the encoder.

**FINDING (tests/test_s2_loss.py, empirically probed against the real
model -- reviewer discussion point, not silently patched):** the plan's
prose describes L_rec as training "only the reconstruction/re-chunk path
(conv1/conv2/dec1/dec2/z0*/chunk2)". Verified against the real model:
this holds EXACTLY for dec1/conv1/z0_dec1 (strictly downstream consumers
of h1_hat -- L_rec, defined only on h1_hat/chunk2(h1_hat), structurally
cannot reach them). It does NOT hold for the encoder: `photon/model.py`'s
dec2 skip connection (`z = reshape.pack(u2, h1, ...)`) feeds the
student's own non-detached `h1` directly into `h1_hat`'s graph, so L_rec
alone (backpropagated without the "final" term) ALSO reaches
chunk1/enc1/chunk2/enc2/embed_tokens through that skip -- confirmed with
a throwaway gradient probe before writing tests/test_s2_loss.py's
assertions. `alpha_A`/`alpha_B` are therefore ALSO a secondary encoder
learning-rate lever, not a purely reconstruction-path-scoped one. Fixing
this would require detaching `h1`/`h0` at the skip connection INSIDE
`photon/model.py` itself -- out of scope (frozen for S2) -- so this is
surfaced for the reviewer/Jun to weigh in on, not silently absorbed.

The two cosines are the exact `harness/g1_metrics.g1b_student_running_probe`
formula (fp32, `dim=-1`, `.mean()`), promoted from a measured DIAGNOSTIC to
a TRAINED loss term (that promotion is S2's whole point per the plan).
"""
import torch
import torch.nn.functional as F

from .s1_loss import relative_mse

DEFAULT_ALPHA_A = 0.15
DEFAULT_ALPHA_B = 0.15


def l_s2(mid: dict, H28: torch.Tensor, chunk2, alpha_A: float = DEFAULT_ALPHA_A,
         alpha_B: float = DEFAULT_ALPHA_B) -> tuple[torch.Tensor, dict]:
    """mid: the dict returned by `photon(input_ids, return_intermediates=True)`
    (must contain h0_hat/h1/h2/h1_hat). H28: the frozen teacher's pre-norm
    layers[27] output on the SAME input_ids, `[B, T, d]`, detached.
    chunk2: `photon.chunk2` (the live module -- called here, not precomputed,
    so `chunk2(h1_hat)` stays in h1_hat's autograd graph).

    Returns `(total, {"final": ..., "rec_A": ..., "rec_B": ...})` -- all
    three still-differentiable scalars (not `.item()`'d; the caller does
    that after `.backward()`, same discipline as `train/train_s1.py`).
    """
    h0_hat = mid["h0_hat"]
    h1 = mid["h1"]
    h2 = mid["h2"]
    h1_hat = mid["h1_hat"]

    final = relative_mse(h0_hat, H28)

    cos_A = F.cosine_similarity(h1_hat.float(), h1.detach().float(), dim=-1)
    rec_A = (1.0 - cos_A).mean()

    chunk2_h1_hat = chunk2(h1_hat)
    cos_B = F.cosine_similarity(chunk2_h1_hat.float(), h2.detach().float(), dim=-1)
    rec_B = (1.0 - cos_B).mean()

    total = final + alpha_A * rec_A + alpha_B * rec_B
    return total, {"final": final, "rec_A": rec_A, "rec_B": rec_B}
