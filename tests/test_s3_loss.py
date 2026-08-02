"""train/s3_loss.py -- the S3 KD-loss correctness gate (mirrors
tests/test_s2_loss.py's gradient-isolation style on the real Qwen3-0.6B,
CPU/fp32).

Pins the things a paid KD run depends on:
  - forward KL is a true KL: >= 0, and exactly 0 when student == teacher
    (catches a flipped direction or a dropped teacher-entropy constant);
  - the KD term trains THROUGH the norm + tied lm_head (S3's whole point --
    unlike S2's deliberately pre-norm `final` term) and back through the
    composed graph, while the teacher logits carry no gradient;
  - the optional CE term uses the correct next-token SHIFT
    (logits[:, :-1] vs gold[:, 1:]) -- a one-off shift would train the
    student to predict the wrong position;
  - the rec_B anchor keeps S2's exact reconstruction-path gradient reach.
"""
import inspect

import torch
import torch.nn.functional as F
import pytest

from photon.config import PhotonConfig
from photon.surgery import load_photon_qwen3
from train.s3_loss import forward_kl, l_s3


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


# ---- forward_kl: it is a real KL ----

def test_forward_kl_is_zero_when_student_equals_teacher():
    """KD must be exactly 0 (up to fp noise) when the student logits equal
    the teacher logits -- proves the teacher-entropy constant is included
    (a cross-entropy-only 'KD' would be a large positive number here) and
    the direction is consistent."""
    torch.manual_seed(0)
    z = torch.randn(2, 5, 137)
    kl = forward_kl(z.clone(), z.clone(), temperature=1.0)
    assert abs(kl.item()) < 1e-5, f"KL(student==teacher) should be ~0, got {kl.item()}"


def test_forward_kl_is_nonnegative_and_shift_matters():
    """Forward KL >= 0 for arbitrary logits; and it is POSITION-WISE (no
    shift) -- rolling the teacher by one position changes the value, so the
    student is matched to the teacher's SAME-position distribution, not a
    neighbour's."""
    torch.manual_seed(1)
    zs = torch.randn(2, 6, 91)
    zt = torch.randn(2, 6, 91)
    kl = forward_kl(zs, zt, temperature=1.0)
    assert kl.item() >= -1e-6
    kl_rolled = forward_kl(zs, torch.roll(zt, shifts=1, dims=1), temperature=1.0)
    assert abs(kl.item() - kl_rolled.item()) > 1e-4, \
        "KD is invariant to a teacher position-roll -- alignment/shift is wrong"


def test_temperature_scaling_returns_finite_positive():
    torch.manual_seed(2)
    zs, zt = torch.randn(1, 4, 60), torch.randn(1, 4, 60)
    for T in (0.5, 1.0, 2.0):
        kl = forward_kl(zs, zt, temperature=T)
        assert torch.isfinite(kl) and kl.item() >= -1e-6


# ---- l_s3: gradient flow ----

def test_kd_trains_through_norm_and_lm_head_and_composed_graph(photon_real):
    """The KD term (alone) must reach the norm + tied lm_head (via
    embed_tokens) AND back through the whole composed graph -- this is
    exactly what S2's hidden-matching `final` term could NOT do (it was
    pre-norm). Training the output head is the point of S3."""
    photon, cfg = photon_real
    input_ids = _input_ids()
    _zero_grads(photon)

    student_logits, mid = photon(input_ids, return_intermediates=True)
    teacher_logits = torch.randn_like(student_logits)  # a detached leaf, no grad
    _, parts = l_s3(student_logits, teacher_logits, mid, photon.chunk2,
                    gold_ids=input_ids, kd_temperature=1.0, ce_weight=0.0, rec_weight=0.0)
    parts["kd"].backward()

    assert _has_grad(photon.norm), "KD must train self.norm (S3 trains the output head, unlike S2)"
    for name in ("embed_tokens", "dec1", "dec2", "enc1", "enc2", "chunk1", "chunk2", "conv1", "conv2"):
        assert _has_grad(_submodule(photon, name)), f"KD left {name} with no gradient"
    assert teacher_logits.grad is None, "teacher logits must carry no gradient"


def test_ce_uses_the_correct_next_token_shift(photon_real):
    """With ce_weight>0, l_s3's reported `ce` must equal the standard shifted
    next-token cross-entropy exactly: logits[:, :-1] predict gold[:, 1:].
    Pins the shift (a bug here would misalign the CE anchor by one token)."""
    photon, cfg = photon_real
    input_ids = _input_ids()
    student_logits, mid = photon(input_ids, return_intermediates=True)
    teacher_logits = torch.randn_like(student_logits)
    _, parts = l_s3(student_logits, teacher_logits, mid, photon.chunk2,
                    gold_ids=input_ids, kd_temperature=1.0, ce_weight=0.5, rec_weight=0.0)

    V = student_logits.shape[-1]
    expected = F.cross_entropy(student_logits[:, :-1].float().reshape(-1, V),
                               input_ids[:, 1:].reshape(-1))
    assert torch.allclose(parts["ce"], expected, atol=1e-6), \
        "l_s3 CE does not match the expected shifted next-token cross-entropy"


def test_ce_is_zero_tensor_when_disabled(photon_real):
    photon, cfg = photon_real
    input_ids = _input_ids()
    student_logits, mid = photon(input_ids, return_intermediates=True)
    teacher_logits = torch.randn_like(student_logits)
    _, parts = l_s3(student_logits, teacher_logits, mid, photon.chunk2,
                    gold_ids=input_ids, kd_temperature=1.0, ce_weight=0.0, rec_weight=0.1)
    assert parts["ce"].item() == 0.0


def test_rec_b_anchor_reaches_the_reconstruction_stage_only(photon_real):
    """rec_B alone (S3's RecGen anchor -- identical form to S2's) must reach
    conv2/dec2/z0_dec2/chunk2 (the stage producing h1_hat / chunk2(h1_hat))
    but NOT dec1/conv1/z0_dec1 (strictly downstream of h1_hat). Same reach
    as tests/test_s2_loss.py, re-pinned for the S3 loss module."""
    photon, cfg = photon_real
    input_ids = _input_ids()
    _zero_grads(photon)

    student_logits, mid = photon(input_ids, return_intermediates=True)
    teacher_logits = torch.randn_like(student_logits)
    _, parts = l_s3(student_logits, teacher_logits, mid, photon.chunk2,
                    gold_ids=input_ids, kd_temperature=1.0, ce_weight=0.0, rec_weight=0.1)
    parts["rec_B"].backward()

    for name in ("conv2", "dec2", "chunk2"):
        assert _has_grad(_submodule(photon, name)), f"rec_B left {name} with no gradient"
    assert photon.z0_dec2.grad is not None and torch.any(photon.z0_dec2.grad != 0)
    for name in ("conv1", "dec1"):
        for p in _submodule(photon, name).parameters():
            assert p.grad is None, f"rec_B unexpectedly reached {name} (downstream-only of h1_hat)"
    assert photon.z0_dec1.grad is None


def test_s3_step_uses_return_intermediates_and_teacher_logits():
    """S3 introduces no new forward path in the model: train_step must call
    the existing frozen `photon(input_ids, return_intermediates=True)`
    (proven causal by tests/test_causality_integrated.py) and must feed the
    teacher's LOGITS (not hidden states) into the loss."""
    from train import s3_step
    src = inspect.getsource(s3_step.train_step)
    assert "return_intermediates=True" in src
    assert "teacher.logits(" in src
