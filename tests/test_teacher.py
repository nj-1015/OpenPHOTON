"""harness/teacher.py -- TeacherHiddenExtractor's selective hooks match
HF's own output_hidden_states=True exactly at the boundaries that should
match, and the H28 pre/post-norm distinction (reviewer should-fix #1) is
real, measured, and now runtime-locked (not just documented).

CPU/fp32, real Qwen3-0.6B (same discipline as tests/test_shapes.py).
"""
import torch
import pytest

from harness.teacher import TeacherHiddenExtractor


@pytest.fixture(scope="module")
def teacher():
    return TeacherHiddenExtractor(dtype=torch.float32, device="cpu", attn_implementation="eager")


def test_construction_runs_the_h28_convention_lock_without_raising(teacher):
    # if we got here, TeacherHiddenExtractor.__init__'s
    # _verify_h28_is_pre_norm() already ran and passed
    assert teacher is not None


def test_capture_after_construction_is_not_polluted_by_the_probe(teacher):
    """the one-time convention-check probe (T=4 zeros) must not leak into
    a real .capture() call's returned dict."""
    input_ids = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long)
    out = teacher.capture(input_ids)
    assert out["H0"].shape[1] == 6  # the real input's T, not the probe's T=4


def test_h0_h7_h14_h21_match_hf_output_hidden_states_exactly(teacher):
    """the 4 non-H28 hooks should match HF's own output_hidden_states=True
    tuple exactly (they're all pre-any-norm intermediate states -- only
    the very last one gets an extra norm() applied, see H28's test below)."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    captured = teacher.capture(input_ids)

    with torch.no_grad():
        ref = teacher.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

    assert torch.equal(captured["H0"], ref.hidden_states[0])
    assert torch.equal(captured["H7"], ref.hidden_states[7])
    assert torch.equal(captured["H14"], ref.hidden_states[14])
    assert torch.equal(captured["H21"], ref.hidden_states[21])


def test_h28_is_pre_norm_not_post_norm_hs28(teacher):
    """the exact claim reviewer should-fix #1 pins: H28 (layers[27]'s raw
    output) differs substantially from output_hidden_states=True's
    post-norm hs[28], but norm(H28) matches hs[28] exactly."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long)
    captured = teacher.capture(input_ids)
    h28 = captured["H28"]

    with torch.no_grad():
        ref = teacher.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hs28 = ref.hidden_states[-1]

    max_abs_raw_vs_hs28 = (h28 - hs28).abs().max().item()
    max_abs_normed_vs_hs28 = (teacher.model.model.norm(h28) - hs28).abs().max().item()

    print(f"\n[H28 convention] raw-vs-hs28 max_abs={max_abs_raw_vs_hs28:.3e}  "
          f"norm(raw)-vs-hs28 max_abs={max_abs_normed_vs_hs28:.3e}")

    assert max_abs_raw_vs_hs28 > 1.0, "H28 unexpectedly matches post-norm hs[28] -- convention drifted"
    assert max_abs_normed_vs_hs28 < 1e-4, "norm(H28) no longer matches hs[28] exactly"


def test_verify_h28_convention_can_be_disabled_for_speed():
    """opt-out flag exists (construction-time probe forward is cheap but
    not free) -- exercise the flag path itself."""
    t = TeacherHiddenExtractor(dtype=torch.float32, device="cpu",
                                attn_implementation="eager", verify_h28_convention=False)
    assert t is not None
