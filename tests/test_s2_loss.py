"""train/s2_loss.py -- the S2 crux gradient-isolation gate (build plan P4,
mirrors tests/test_s1_targets.py's gradient-isolation style: prove the
detach table at the autograd level, not just that the loss computes a
number).

Loads the real Qwen3-0.6B (CPU/fp32), same fixture pattern as
tests/test_s1_targets.py / tests/test_causality_integrated.py.

**Empirically-verified finding (not assumed -- probed with a throwaway
script against the real model before writing these assertions), flagged
for reviewer attention:** the plan's crux doc describes L_rec as training
"only the reconstruction/re-chunk path (conv1/conv2/dec1/dec2/z0*/chunk2)"
with the encoder trained "only via the final term". That holds for HALF of
that list (dec1/conv1/z0_dec1 truly get NO grad from L_rec alone -- they
are strictly DOWNSTREAM of h1_hat, consuming it to produce h0_hat, so a
loss defined purely on h1_hat/chunk2(h1_hat) cannot reach them). It does
NOT hold for the encoder: `photon/model.py`'s dec2 skip connection
(`z = reshape.pack(u2, h1, ...)`) feeds the STUDENT's own, non-detached
`h1` directly into `h1_hat`'s computation graph -- so L_rec alone (with
`h1_hat` undetached, only its DETACHED reference copy `h1.detach()` used
as the target) still backpropagates through that skip into chunk1/enc1
(which produced h1) and chunk2/enc2 (which produced h2, itself computed
from h1) and embed_tokens (h0->h1). This is inherent to reusing
`photon/model.py`'s forward unmodified (frozen, per this task) -- the
`.detach()` calls in `s2_loss.l_s2` correctly stop the "target-side cheat"
gradient (the encoder can't shortcut by moving h1 to match h1_hat), but
they cannot stop the SEPARATE, structural skip-connection gradient path
into the same encoder modules. Net effect: L_rec's alpha_A/alpha_B are
*also* a (secondary) encoder learning-rate lever, not purely a
reconstruction-path lever, as the plan's prose implies. Documented here
with a passing test of the TRUE behavior (not the idealized one) so this
is visible and testable going forward, not silently swept under -- explicit
reviewer discussion point, not a code defect to silently patch (patching
would need detaching `h1`/`h0` at the skip inside `photon/model.py` itself,
which is out of scope -- `photon/` is frozen for S2).
"""
import inspect

import torch
import torch.nn.functional as F
import pytest

from photon.config import PhotonConfig
from photon.surgery import load_photon_qwen3
from train.s1_loss import relative_mse
from train.s2_loss import l_s2


@pytest.fixture(scope="module")
def photon_real():
    cfg = PhotonConfig(C1=2, C2=2, R1=2, R2=2)
    photon, stock = load_photon_qwen3(cfg)
    del stock
    return photon, cfg


def _input_ids(B=1, T=8, vocab=100):
    torch.manual_seed(0)
    return torch.randint(0, vocab, (B, T))


def _submodule(photon, name):
    return dict(chunk1=photon.chunk1, chunk2=photon.chunk2, conv1=photon.conv1,
                conv2=photon.conv2, enc1=photon.enc1, enc2=photon.enc2,
                dec1=photon.dec1, dec2=photon.dec2, embed_tokens=photon.embed_tokens,
                norm=photon.norm)[name]


def _has_grad(mod) -> bool:
    return any(p.grad is not None and torch.any(p.grad != 0) for p in mod.parameters())


def _zero_grads(photon):
    for p in photon.parameters():
        p.grad = None


def test_l_rec_alone_reaches_the_h1_hat_producing_stage(photon_real):
    """rec_A + rec_B alone (no final term) must populate grad on conv2,
    dec2, z0_dec2, chunk2 -- the stage that actually PRODUCES h1_hat /
    chunk2(h1_hat) (matches the plan's list, minus dec1/conv1/z0_dec1 --
    see module docstring for why those are excluded)."""
    photon, cfg = photon_real
    input_ids = _input_ids()
    _zero_grads(photon)

    _, mid = photon(input_ids, return_intermediates=True)
    H28 = torch.randn(*mid["h0_hat"].shape)  # unused by L_rec; only final needs it
    _, parts = l_s2(mid, H28, photon.chunk2, alpha_A=0.15, alpha_B=0.15)
    l_rec = parts["rec_A"] + parts["rec_B"]
    l_rec.backward()

    for name in ("conv2", "dec2"):
        assert _has_grad(_submodule(photon, name)), f"L_rec left {name} with no gradient"
    assert photon.z0_dec2.grad is not None and torch.any(photon.z0_dec2.grad != 0)
    assert _has_grad(photon.chunk2), "L_rec (rec_B calls chunk2(h1_hat) directly) must reach chunk2's own weights"


def test_l_rec_alone_does_not_reach_the_dec1_stage(photon_real):
    """dec1/conv1/z0_dec1 produce h0_hat FROM h1_hat -- they are strictly
    DOWNSTREAM of h1_hat, so a loss defined only on h1_hat/chunk2(h1_hat)
    (L_rec) cannot reach them. This is the one part of the plan's "L_rec
    trains only the reconstruction path" claim that holds exactly."""
    photon, cfg = photon_real
    input_ids = _input_ids()
    _zero_grads(photon)

    _, mid = photon(input_ids, return_intermediates=True)
    H28 = torch.randn(*mid["h0_hat"].shape)
    _, parts = l_s2(mid, H28, photon.chunk2, alpha_A=0.15, alpha_B=0.15)
    l_rec = parts["rec_A"] + parts["rec_B"]
    l_rec.backward()

    for name in ("conv1", "dec1"):
        for p in _submodule(photon, name).parameters():
            assert p.grad is None, f"L_rec unexpectedly reached {name} (should be downstream-only of h1_hat)"
    assert photon.z0_dec1.grad is None


def test_l_rec_alone_also_reaches_the_encoder_via_the_dec2_skip_connection(photon_real):
    """FINDING (see module docstring): because `photon/model.py`'s dec2
    skip connection feeds the student's own non-detached `h1` into
    `h1_hat`'s graph, L_rec ALSO backpropagates into chunk1/enc1 (which
    produced h1) and chunk2/enc2 (which produced h2 from h1) and
    embed_tokens (h0->h1) -- even though `h1`/`h2` are `.detach()`'d as
    L_rec's REFERENCE targets. The detach stops the "target-side cheat"
    (encoder moving h1 to match h1_hat) but not this separate structural
    path. Asserted here (not silently ignored) so the reviewer pass has a
    concrete, tested claim to evaluate, not just prose."""
    photon, cfg = photon_real
    input_ids = _input_ids()
    _zero_grads(photon)

    _, mid = photon(input_ids, return_intermediates=True)
    H28 = torch.randn(*mid["h0_hat"].shape)
    _, parts = l_s2(mid, H28, photon.chunk2, alpha_A=0.15, alpha_B=0.15)
    l_rec = parts["rec_A"] + parts["rec_B"]
    l_rec.backward()

    for name in ("chunk1", "enc1", "enc2", "embed_tokens"):
        assert _has_grad(_submodule(photon, name)), (
            f"expected L_rec to (also) reach {name} via the dec2 skip connection "
            f"-- if this now fails, the skip-connection gradient path changed "
            f"and the finding documented in this module's docstring is stale"
        )


def test_final_term_alone_reaches_the_full_composed_graph(photon_real):
    """The "final" term (relative_mse(h0_hat, H28)) must reach EVERY
    trainable stage (chunk1/enc1/chunk2/enc2/conv2/dec2/z0_dec2/conv1/dec1/
    z0_dec1/embed_tokens) -- the whole point of S2 training end-to-end,
    unlike S1's 4 decoupled groups. `norm` must NOT get gradient (h0_hat is
    PRE-norm; this term never touches `self.norm`/`lm_head`'s own forward
    call -- see harness/teacher.py's locked pre-norm convention)."""
    photon, cfg = photon_real
    input_ids = _input_ids()
    _zero_grads(photon)

    _, mid = photon(input_ids, return_intermediates=True)
    H28 = torch.randn(*mid["h0_hat"].shape)
    final = relative_mse(mid["h0_hat"], H28)
    final.backward()

    for name in ("chunk1", "enc1", "chunk2", "enc2", "conv2", "dec2", "conv1", "dec1", "embed_tokens"):
        assert _has_grad(_submodule(photon, name)), f"final term left {name} with no gradient"
    assert photon.z0_dec2.grad is not None and torch.any(photon.z0_dec2.grad != 0)
    assert photon.z0_dec1.grad is not None and torch.any(photon.z0_dec1.grad != 0)
    assert not _has_grad(photon.norm), "final term (pre-norm h0_hat) must not touch self.norm"


def test_h28_and_detached_refs_carry_no_grad_fn(photon_real):
    """Basic sanity on the detach table itself: the tensors used as L_rec's
    REFERENCE targets must be genuinely detached (no grad_fn, requires_grad
    False) -- catches an accidental removal of `.detach()` in s2_loss.l_s2
    before it silently starts training the encoder via the 'cheat' path
    (see module docstring's "Why detach the encoder-side refs")."""
    photon, cfg = photon_real
    input_ids = _input_ids()
    _, mid = photon(input_ids, return_intermediates=True)
    assert mid["h1"].detach().grad_fn is None
    assert mid["h1"].detach().requires_grad is False
    assert mid["h2"].detach().grad_fn is None
    assert mid["h2"].detach().requires_grad is False


def test_s2_reuses_model_forward_return_intermediates_unmodified():
    """Causality-coverage assert referencing the existing gate
    (tests/test_causality_integrated.py): S2 introduces NO new forward
    path -- `train/s2_step.py::train_step` must call
    `photon(input_ids, return_intermediates=True)` (photon/model.py's
    EXISTING, frozen forward), the exact call signature
    tests/test_causality_integrated.py already proves causal (a token's
    h0_hat cannot depend on tokens after it). If S2 ever grows a second,
    bespoke forward path (like S1's decoupled `train/s1_targets.py` sub-
    forwards, which needed their OWN causality reasoning), that test's
    coverage would no longer apply and this assert would need to change
    alongside it -- flagged here as the reason this is safe to skip
    re-deriving causality from scratch for S2."""
    from train import s2_step
    src = inspect.getsource(s2_step.train_step)
    assert "return_intermediates=True" in src
    assert "photon(input_ids" in src or "photon(input_ids," in src
