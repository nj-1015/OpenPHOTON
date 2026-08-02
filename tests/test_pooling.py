"""train/pooling.py -- P_k in {mean, last, strided} (plan risk R-A1)."""
import pytest
import torch

from train.pooling import pool


def test_pool_k1_is_identity_for_all_modes():
    h = torch.randn(2, 8, 4)
    for mode in ("mean", "last", "strided"):
        assert torch.equal(pool(h, 1, mode), h)


def test_pool_mean_matches_manual_chunk_average():
    B, T, d, k = 2, 8, 4, 2
    h = torch.randn(B, T, d)
    out = pool(h, k, "mean")
    assert out.shape == (B, T // k, d)
    expected = h.reshape(B, T // k, k, d).mean(dim=2)
    assert torch.allclose(out, expected)


def test_pool_last_takes_last_position_of_each_chunk():
    B, T, d, k = 1, 6, 3, 2
    # ramp so each position's value == its index, easy to check selection
    h = torch.arange(T, dtype=torch.float32).view(1, T, 1).expand(B, T, d).clone()
    out = pool(h, k, "last")
    assert out.shape == (B, T // k, d)
    # chunks are (0,1),(2,3),(4,5) -- last-in-chunk indices are 1,3,5
    expected_indices = [1.0, 3.0, 5.0]
    for i, v in enumerate(expected_indices):
        assert torch.allclose(out[0, i], torch.full((d,), v))


def test_pool_strided_takes_first_position_of_each_chunk():
    B, T, d, k = 1, 6, 3, 2
    h = torch.arange(T, dtype=torch.float32).view(1, T, 1).expand(B, T, d).clone()
    out = pool(h, k, "strided")
    assert out.shape == (B, T // k, d)
    expected_indices = [0.0, 2.0, 4.0]
    for i, v in enumerate(expected_indices):
        assert torch.allclose(out[0, i], torch.full((d,), v))


def test_pool_rejects_non_divisible_length():
    h = torch.randn(1, 7, 4)
    with pytest.raises(AssertionError):
        pool(h, 2, "mean")


def test_pool_rejects_unknown_mode():
    h = torch.randn(1, 4, 4)
    with pytest.raises(ValueError):
        pool(h, 2, "bogus")
