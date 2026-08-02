"""Decoding-knob tests for inference/recgen.py: top_p (nucleus), repetition
penalty, and seed reproducibility.

The pure-tensor tests exercise `_sample`/`_apply_top_p`/`_apply_repetition_penalty`
directly on hand-built logits so the filtering math is checked exactly, with no
model load. The integration test (`test_seeded_sampling_is_reproducible`) runs
one short real-model generation to prove the knobs survive the full
recgen_generate path.

These are the SAME knobs the paper's RecGen loop needs for usable longer
generation on the S3 checkpoint (README: "wanders over longer generations") --
repetition penalty in particular is the standard counter to RecGen's
known looping tendency.
"""
import torch
import torch.nn.functional as F
import pytest

from inference import recgen


# ---------------------------------------------------------------------------
# _apply_top_p
# ---------------------------------------------------------------------------
def test_top_p_keeps_smallest_sufficient_nucleus():
    # logits such that softmax = [0.6, 0.3, 0.07, 0.03]
    logits = torch.tensor([3.0, 2.0, 0.6, 0.0])
    filtered = recgen._apply_top_p(logits, top_p=0.95)
    probs = F.softmax(filtered, dim=-1)
    # cumulative mass of the top-3 = 0.97 >= 0.95; top-2 = 0.90 < 0.95,
    # so the nucleus is exactly the top-3 tokens -- the 4th is masked out
    # (softmax(-inf) == 0.0, so "nonzero mass" is the right probe).
    assert (probs > 0).sum().item() == 3
    assert probs[3].item() == 0.0
    # survivors renormalize to sum 1
    assert abs(probs.sum().item() - 1.0) < 1e-6


def test_top_p_never_drops_top_token():
    # a degenerate distribution: top token alone carries 0.99 mass
    logits = torch.tensor([10.0, 0.0, 0.0])
    filtered = recgen._apply_top_p(logits, top_p=0.5)
    probs = F.softmax(filtered, dim=-1)
    # even a tiny top_p must leave at least the argmax token
    assert (probs > 0).sum().item() == 1
    assert probs[0].item() == 1.0


def test_top_p_full_and_disabled():
    logits = torch.tensor([3.0, 2.0, 0.6, 0.0])
    # top_p >= 1.0 keeps everything
    filtered = recgen._apply_top_p(logits, top_p=1.0)
    assert (F.softmax(filtered, dim=-1) > 0).all()
    # and _sample with top_p=0.0 (disabled) never hits the filter
    assert recgen._sample(logits, temperature=0.0, top_p=0.0) == int(logits.argmax())


# ---------------------------------------------------------------------------
# _apply_repetition_penalty
# ---------------------------------------------------------------------------
def test_repetition_penalty_divides_positive_logits():
    logits = torch.tensor([5.0, 1.0, -2.0])
    penalized = recgen._apply_repetition_penalty(logits, torch.tensor([0]), penalty=2.0)
    # positive logit divided: 5/2 = 2.5
    assert penalized[0].item() == 2.5
    # untouched tokens unchanged
    assert penalized[1].item() == 1.0
    assert penalized[2].item() == -2.0


def test_repetition_penalty_multiplies_negative_logits():
    logits = torch.tensor([5.0, 1.0, -2.0])
    penalized = recgen._apply_repetition_penalty(logits, torch.tensor([2]), penalty=2.0)
    # negative logit multiplied: -2 * 2 = -4 (pushed further down)
    assert penalized[2].item() == -4.0
    assert penalized[0].item() == 5.0


def test_repetition_penalty_can_flip_greedy_choice():
    # token 0 is the argmax but has been generated already; with a strong
    # penalty the runner-up wins instead. (Only token 0 is in prev_ids, so
    # only token 0's logit is divided: 3.0/1.5 = 2.0 < 2.9.)
    logits = torch.tensor([3.0, 2.9])
    assert int(logits.argmax()) == 0
    tok = recgen._sample(logits, temperature=0.0, repetition_penalty=1.5,
                         prev_ids=torch.tensor([0, 5, 7]))
    assert tok == 1


def test_repetition_penalty_disabled_at_1():
    logits = torch.tensor([3.0, 2.9])
    tok = recgen._sample(logits, temperature=0.0, repetition_penalty=1.0,
                         prev_ids=torch.tensor([0, 1, 0]))
    assert tok == 0  # unchanged


# ---------------------------------------------------------------------------
# seed reproducibility through the full path
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def photon_real():
    from photon.config import PhotonConfig
    from photon.surgery import load_photon_qwen3

    cfg = PhotonConfig(C1=2, C2=2, R1=2, R2=2, split=(7, 7, 7, 7),
                       rope_mode="reset", bypass=False, dtype="float32")
    m, _ = load_photon_qwen3(cfg)
    return m.float().eval()


def test_seeded_sampling_is_reproducible(photon_real):
    torch.manual_seed(0)
    ids = torch.randint(0, 1000, (1, 8))
    a = recgen.recgen_generate(photon_real, ids, max_new_tokens=6, temperature=0.9,
                               top_k=0, top_p=0.9, repetition_penalty=1.0, seed=42)
    b = recgen.recgen_generate(photon_real, ids, max_new_tokens=6, temperature=0.9,
                               top_k=0, top_p=0.9, repetition_penalty=1.0, seed=42)
    assert torch.equal(a, b), "same seed must give identical temperature-sampled output"


def test_greedy_ignores_other_knobs(photon_real):
    # temperature=0 must produce the same output regardless of top_k/top_p/
    # penalty -- the default-knob path is untouched by the new parameters.
    torch.manual_seed(0)
    ids = torch.randint(0, 1000, (1, 8))
    base = recgen.recgen_generate(photon_real, ids, max_new_tokens=4, temperature=0.0)
    with_knobs = recgen.recgen_generate(photon_real, ids, max_new_tokens=4, temperature=0.0,
                                        top_k=50, top_p=0.9, repetition_penalty=1.0)
    assert torch.equal(base, with_knobs)
