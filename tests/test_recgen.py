"""inference/recgen.py -- RecGen generation correctness gates.

The load-bearing test is `test_dec1_plumbing_matches_forward`: the
continuous windowed bottom decode must reproduce model.forward's logits
position-by-position (weight-independent -- runs on a fresh surgery model,
CPU/fp32). This isolates the indexing/shift/positions/masks from model
quality; per Theorem A.6 we do NOT assert recgen_generate == model.forward
on real weights (that only holds at perfect h1_hat==h1)."""
import torch
import pytest

from photon.config import PhotonConfig
from photon.surgery import load_photon_qwen3
from inference import recgen


@pytest.fixture(scope="module")
def photon_real():
    cfg = PhotonConfig(C1=2, C2=2, R1=2, R2=2, split=(7, 7, 7, 7),
                        rope_mode="reset", bypass=False, dtype="float32")
    m, _ = load_photon_qwen3(cfg)
    return m.float().eval()


def test_dec1_plumbing_matches_forward(photon_real):
    """The verifiable gate: continuous windowed bottom decode == model.forward
    (fed model.forward's own h1_hat), to fp32 tolerance."""
    torch.manual_seed(0)
    ids = torch.randint(0, 1000, (1, 16))
    diff, ok = recgen.verify_dec1_plumbing(photon_real, ids, atol=1e-3, rtol=1e-3)
    assert ok, f"bottom decode diverged from model.forward (max_abs_diff={diff:.2e})"


def test_generate_produces_valid_tokens_and_length(photon_real):
    torch.manual_seed(0)
    ids = torch.randint(0, 1000, (1, 8))
    out = recgen.recgen_generate(photon_real, ids, max_new_tokens=8, temperature=0.0)
    assert out.shape[1] == 8 + 8                       # whole meta-contexts, exact count
    assert bool((out >= 0).all()) and int(out.max()) < photon_real.embed_tokens.weight.shape[0]


def test_greedy_is_deterministic(photon_real):
    torch.manual_seed(0)
    ids = torch.randint(0, 1000, (1, 8))
    a = recgen.recgen_generate(photon_real, ids, max_new_tokens=8, temperature=0.0)
    b = recgen.recgen_generate(photon_real, ids, max_new_tokens=8, temperature=0.0)
    assert torch.equal(a, b)


def test_prompt_length_must_be_multiple_of_c1c2(photon_real):
    ids = torch.randint(0, 1000, (1, 6))               # 6 not divisible by C1*C2=4
    with pytest.raises(AssertionError):
        recgen.recgen_generate(photon_real, ids, max_new_tokens=4)


def test_kv_footprint_is_16x(photon_real):
    vanilla, rec, ratio = recgen.kv_footprint(photon_real, 2048)
    assert vanilla == 28 * 2048
    assert rec == 7 * (2048 // 4)
    assert abs(ratio - 16.0) < 1e-6                    # the paper's (2,2) analytic reduction
