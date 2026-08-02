"""M1 -- Chunker/Converter identity-mean / replication init (slide 20).

Not random init: Chunker must return the arithmetic mean of its C inputs at
step 0; Converter must return R exact parent copies. Random init would leave
the decoder half seeing garbage for the first few thousand distillation
steps (S0 plan / slide 20 rationale).
"""
import torch

from photon.modules import Chunker, Converter


def test_chunker_returns_mean_of_C_inputs():
    torch.manual_seed(0)
    d, C, B, M = 8, 2, 3, 6   # M must be divisible by C
    chunker = Chunker(d, C)
    x = torch.randn(B, M, d)
    out = chunker(x)
    assert out.shape == (B, M // C, d)

    x_grouped = x.reshape(B, M // C, C, d)
    expected = x_grouped.mean(dim=2)
    assert torch.allclose(out, expected, atol=1e-6)


def test_chunker_mean_C4():
    torch.manual_seed(1)
    d, C, B, M = 16, 4, 2, 8
    chunker = Chunker(d, C)
    x = torch.randn(B, M, d)
    out = chunker(x)
    x_grouped = x.reshape(B, M // C, C, d)
    expected = x_grouped.mean(dim=2)
    assert torch.allclose(out, expected, atol=1e-6)


def test_converter_returns_R_exact_copies():
    torch.manual_seed(2)
    d, R, B, M = 8, 2, 3, 5
    converter = Converter(d, R)
    z = torch.randn(B, M, d)
    out = converter(z)
    assert out.shape == (B, M, R, d)
    for r in range(R):
        assert torch.allclose(out[:, :, r, :], z, atol=1e-6)


def test_converter_R4():
    torch.manual_seed(3)
    d, R, B, M = 8, 4, 1, 3
    converter = Converter(d, R)
    z = torch.randn(B, M, d)
    out = converter(z)
    for r in range(R):
        assert torch.allclose(out[:, :, r, :], z, atol=1e-6)


def test_chunker_C1_is_identity():
    """C=1 degenerate case -- required for the M5 bypass golden test."""
    torch.manual_seed(4)
    d, B, M = 8, 2, 5
    chunker = Chunker(d, 1)
    x = torch.randn(B, M, d)
    out = chunker(x)
    assert torch.allclose(out, x, atol=1e-6)


def test_converter_R1_is_identity():
    """R=1 degenerate case -- required for the M5 bypass golden test."""
    torch.manual_seed(5)
    d, B, M = 8, 2, 5
    converter = Converter(d, 1)
    z = torch.randn(B, M, d)
    out = converter(z)
    assert torch.allclose(out[:, :, 0, :], z, atol=1e-6)
