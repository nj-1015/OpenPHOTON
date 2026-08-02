"""train/s1_targets.py -- the S1 crux correctness gate. Loads the real
Qwen3-0.6B (CPU/fp32), same fixture pattern as tests/test_shapes.py.

Three checks:
  1. shapes match the plan's crux table exactly, at a real (C1,C2,R1,R2).
  2. gradient ISOLATION: each group's backward populates grad on exactly
     that group's "trains" column (plan table) and nothing else -- this
     is the actual proof that "decoupled" holds, not just that the code
     runs.
  3. bypass-degenerate exactness (plan risk R-A1's "sane at init" claim,
     made concrete): with bypass=True (C1=C2=R1=R2=1, P_1 == identity,
     chunk*/conv* == identity), each group's sub-forward reduces to
     replaying that group's 7 real Qwen3 layers on the teacher's own
     upstream hidden state -- so at init (before any training), output
     should equal the teacher's own next hidden state near-exactly
     (student and teacher are two independently-loaded, numerically
     identical copies of the same pretrained checkpoint; nothing in
     either load path is randomly initialized).
"""
import torch
import pytest

from harness.teacher import TeacherHiddenExtractor
from photon.config import PhotonConfig
from photon.surgery import load_photon_qwen3
from train.s1_targets import GROUP_ORDER, build_group_forwards


@pytest.fixture(scope="module")
def photon_real():
    cfg = PhotonConfig(C1=2, C2=2, R1=2, R2=2)
    photon, stock = load_photon_qwen3(cfg)
    del stock
    return photon, cfg


def _fake_teacher_hs(B, T, d):
    torch.manual_seed(0)
    return {k: torch.randn(B, T, d) for k in ("H0", "H7", "H14", "H21", "H28")}


def test_group_output_shapes_match_the_crux_table(photon_real):
    photon, cfg = photon_real
    d = photon.d
    B, T = 1, 8  # divisible by C1*C2=4
    teacher_hs = _fake_teacher_hs(B, T, d)

    groups = build_group_forwards(photon, cfg, teacher_hs, pool_mode="mean")

    enc1_out, enc1_tgt = groups["enc1"]()
    assert enc1_out.shape == (B, T // cfg.C1, d)
    assert enc1_tgt.shape == enc1_out.shape

    enc2_out, enc2_tgt = groups["enc2"]()
    assert enc2_out.shape == (B, T // (cfg.C1 * cfg.C2), d)
    assert enc2_tgt.shape == enc2_out.shape

    dec2_out, dec2_tgt = groups["dec2"]()
    assert dec2_out.shape == (B, T // cfg.C1, d)   # h1_hat grid
    assert dec2_tgt.shape == dec2_out.shape

    dec1_out, dec1_tgt = groups["dec1"]()
    assert dec1_out.shape == (B, T, d)             # h0_hat grid
    assert dec1_tgt.shape == (B, T, d)


@pytest.mark.parametrize("group_name,trained_modules,untrained_modules", [
    ("enc1", ["chunk1", "enc1"], ["chunk2", "enc2", "conv2", "dec2", "conv1", "dec1"]),
    ("enc2", ["chunk2", "enc2"], ["chunk1", "enc1", "conv2", "dec2", "conv1", "dec1"]),
    ("dec2", ["conv2", "dec2"], ["chunk1", "enc1", "chunk2", "enc2", "conv1", "dec1"]),
    ("dec1", ["conv1", "dec1"], ["chunk1", "enc1", "chunk2", "enc2", "conv2", "dec2"]),
])
def test_group_backward_isolated_to_its_own_trainable_params(
        photon_real, group_name, trained_modules, untrained_modules):
    """Proves 'decoupled' at the autograd level: group X's backward must
    populate grad on exactly its 'trains' column (plan table), never on
    another group's parameters -- the actual mechanism behind "one 7-layer
    subgraph live at a time"."""
    photon, cfg = photon_real
    d = photon.d
    B, T = 1, 8
    teacher_hs = _fake_teacher_hs(B, T, d)

    for p in photon.parameters():
        p.grad = None

    groups = build_group_forwards(photon, cfg, teacher_hs, pool_mode="mean")
    out, target = groups[group_name]()
    loss = (out - target).pow(2).sum()
    loss.backward()

    def _submodule(name):
        return dict(chunk1=photon.chunk1, chunk2=photon.chunk2, conv1=photon.conv1,
                    conv2=photon.conv2, enc1=photon.enc1, enc2=photon.enc2,
                    dec1=photon.dec1, dec2=photon.dec2)[name]

    for name in trained_modules:
        grads = [p.grad for p in _submodule(name).parameters()]
        assert any(g is not None and torch.any(g != 0) for g in grads), \
            f"{group_name}'s backward left {name} with no gradient"

    for name in untrained_modules:
        for p in _submodule(name).parameters():
            assert p.grad is None, f"{group_name}'s backward leaked gradient into {name}"

    # z0_dec1/z0_dec2 -- only the matching group's z0 should get gradient
    if group_name == "dec2":
        assert photon.z0_dec2.grad is not None and torch.any(photon.z0_dec2.grad != 0)
        assert photon.z0_dec1.grad is None
    elif group_name == "dec1":
        assert photon.z0_dec1.grad is not None and torch.any(photon.z0_dec1.grad != 0)
        assert photon.z0_dec2.grad is None
    else:
        assert photon.z0_dec1.grad is None
        assert photon.z0_dec2.grad is None

    # embed_tokens / norm / lm_head are never touched by any S1 group
    for p in photon.embed_tokens.parameters():
        assert p.grad is None
    for p in photon.norm.parameters():
        assert p.grad is None


def test_bypass_degenerate_group_reproduces_teacher_next_stage_at_init():
    """Plan risk R-A1 / 'why defensible' point 3, made concrete: with
    bypass=True, running a group on the teacher's own upstream hidden
    state should reproduce the teacher's own next hidden state near-
    exactly at init, since student and teacher are two independently-
    loaded copies of the identical pretrained checkpoint (nothing in
    either load path -- surgery.load_photon_qwen3 / TeacherHiddenExtractor
    -- is randomly initialized) and bypass strips chunk*/conv*/shift down
    to identity (see photon/reshape.py, photon/modules.py)."""
    cfg = PhotonConfig(C1=1, C2=1, R1=1, R2=1, bypass=True)
    photon, stock = load_photon_qwen3(cfg)
    del stock

    teacher = TeacherHiddenExtractor(model_id="Qwen/Qwen3-0.6B", dtype=torch.float32,
                                      device="cpu", attn_implementation="eager")
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    with torch.no_grad():
        teacher_hs = teacher.capture(input_ids)

    groups = build_group_forwards(photon, cfg, teacher_hs, pool_mode="mean")
    with torch.no_grad():
        for name in GROUP_ORDER:
            out, target = groups[name]()
            max_abs = (out - target).abs().max().item()
            assert max_abs < 1e-3, f"{name}: bypass-degenerate output vs teacher target max_abs={max_abs}"
