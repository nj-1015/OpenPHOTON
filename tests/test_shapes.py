"""M4 -- forward + shape gate. The full slide-23 assertion set, plus the
mask/take_last_C unit checks called out by S0 plan risk #6 ("Local mask over
[briefing;children]... Catch: masks.py unit test (lower-triangular;
take_last_C selects child rows)").

Loads the real Qwen3-0.6B (CPU/fp32) -- these are the only tests in this
suite that need the actual checkpoint, so the module-scoped fixture loads it
once and is reused across the static/param-count checks.
"""
import torch
import pytest
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from photon.config import PhotonConfig
from photon.surgery import load_photon_qwen3
from photon.masks import local_causal_mask
from photon.reshape import take_last_C


@pytest.fixture(scope="module")
def loaded():
    cfg = PhotonConfig(C1=2, C2=2, R1=2, R2=2)
    photon, stock = load_photon_qwen3(cfg)
    return photon, stock, cfg


def test_static_asserts_slide23(loaded):
    photon, stock, cfg = loaded
    hf_cfg = stock.config

    assert hf_cfg.hidden_size == 1024
    assert photon.lm_head.weight is photon.embed_tokens.weight

    layer0 = photon.enc1[0]
    assert layer0.self_attn.q_proj.out_features == 16 * 128
    assert layer0.self_attn.q_proj.out_features != hf_cfg.hidden_size
    assert isinstance(layer0.self_attn.q_norm, Qwen3RMSNorm)


def test_slices_are_verbatim_in_order(loaded):
    """slicing the live ModuleList (no key-renaming) -- object identity with
    the original stock layers, in order (S0 plan risk #5)."""
    photon, stock, cfg = loaded
    stock_layers = stock.model.layers
    for i, layer in enumerate(photon.enc1):
        assert layer is stock_layers[i]
    for i, layer in enumerate(photon.enc2):
        assert layer is stock_layers[7 + i]
    for i, layer in enumerate(photon.dec2):
        assert layer is stock_layers[14 + i]
    for i, layer in enumerate(photon.dec1):
        assert layer is stock_layers[21 + i]


def test_param_count_delta(loaded):
    """596.0M -> 604.4M (+8.39M), slide 16."""
    photon, stock, cfg = loaded
    stock_params = sum(p.numel() for p in stock.parameters())
    photon_params = sum(p.numel() for p in photon.parameters())
    print(f"stock={stock_params}  photon={photon_params}  delta={photon_params - stock_params}")

    assert abs(stock_params - 596_000_000) < 2_000_000
    delta = photon_params - stock_params
    # slide 16's table (8,388,608 = 2 chunkers + 2 converters, at R=2) covers
    # only the four new projections; it does not include z0_dec1/z0_dec2,
    # the learned start latents introduced separately on slide 19/22
    # ("z0 is a learned start latent"). Those add 2 * hidden_size = 2048.
    chunkers_converters = 8_388_608
    z0_params = 2 * photon.d  # z0_dec1 + z0_dec2, each (1,1,d)
    assert delta == chunkers_converters + z0_params
    assert delta == 8_390_656
    # still rounds to the deck's "+8.39M" (8.390656M)
    assert abs(delta / 1_000_000 - 8.39) < 0.005


@pytest.mark.parametrize("R", [1, 2, 4])
def test_runtime_shapes_slide23_at_R(R):
    cfg = PhotonConfig(C1=2, C2=2, R1=R, R2=R)
    photon, stock = load_photon_qwen3(cfg)
    d = stock.config.hidden_size
    B, T = 1, 16  # divisible by C1*C2 = 4

    ids = torch.randint(0, stock.config.vocab_size, (B, T))
    logits, mid = photon(ids, return_intermediates=True)

    assert mid["h1"].shape == (B, T // 2, d)
    assert mid["h2"].shape == (B, T // 4, d)
    assert mid["z_dec2_input"].shape == (B * (T // 4), R + cfg.C2, d)
    assert mid["h0_hat"].shape == mid["h0"].shape
    assert logits.shape == (B, T, stock.config.vocab_size)


def test_T_not_divisible_by_C1C2_raises():
    cfg = PhotonConfig(C1=2, C2=2, R1=2, R2=2)
    photon, stock = load_photon_qwen3(cfg)
    ids = torch.randint(0, stock.config.vocab_size, (1, 15))  # 15 not div by 4
    with pytest.raises(AssertionError):
        photon(ids)


def test_local_causal_mask_lower_triangular():
    L = 4
    m = local_causal_mask(L)[0, 0]
    neg = torch.finfo(torch.float32).min
    for i in range(L):
        for j in range(L):
            if j <= i:
                assert m[i, j] == 0
            else:
                assert m[i, j] == neg


def test_take_last_C_selects_child_rows_not_briefing_rows():
    B, M, R, C, d = 2, 3, 2, 2, 4
    z = torch.arange(B * M * (R + C) * d, dtype=torch.float32).reshape(B * M, R + C, d)
    out = take_last_C(z, C, B=B)
    assert out.shape == (B, M * C, d)
    expected = z[:, R:, :].reshape(B, M * C, d)  # last C rows, i.e. children
    assert torch.equal(out, expected)
    # sanity: it did NOT select the briefing rows instead
    briefing_first_row = z[:, 0, :]
    assert not torch.equal(out[:, 0, :], briefing_first_row)
