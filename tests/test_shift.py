"""M3 -- the off-by-one gate (S0 plan risk #1, THE highest-priority risk).

Chunk g must condition on the parent latent at index g-1, never its own;
z0 stands in for g=0's nonexistent parent. Get this off by one and the
model trivially cheats (sees its own parent) -- loss collapses and nothing
else measured means anything (slide 22, detail 1).

Isolated on an integer-ramp input: [0, 1, 2, 3] -> [z0, 0, 1, 2], run at two
different feature widths standing in for "both levels" (dec2's shift over
level-2 latents, dec1's shift over level-1 latents) -- shift() is one shared
function, so this exercises it at the two shapes it will actually see.
"""
import torch

from photon.reshape import shift


def _ramp(B, M, d, dtype=torch.float32):
    """[B, M, d] where every feature of position m is filled with value m."""
    idx = torch.arange(M, dtype=dtype).view(1, M, 1).expand(B, M, d)
    return idx.clone()


def test_shift_off_by_one_level2_d1024():
    B, M, d = 2, 4, 1024
    h = _ramp(B, M, d)
    z0 = torch.full((1, 1, d), -1.0)
    out = shift(h, z0)
    assert out.shape == h.shape
    # position 0 sees z0 (sentinel -1), never its own value 0
    assert torch.allclose(out[:, 0, :], torch.full((B, d), -1.0))
    # position g (g>=1) sees the parent at g-1, i.e. value g-1
    for g in range(1, M):
        assert torch.allclose(out[:, g, :], torch.full((B, d), float(g - 1)))
    # explicitly: never its own value
    for g in range(M):
        assert not torch.allclose(out[:, g, :], h[:, g, :])


def test_shift_off_by_one_level1_d1024_longer_seq():
    B, M, d = 3, 8, 1024
    h = _ramp(B, M, d)
    z0 = torch.full((1, 1, d), -2.0)
    out = shift(h, z0)
    assert torch.allclose(out[:, 0, :], torch.full((B, d), -2.0))
    for g in range(1, M):
        assert torch.allclose(out[:, g, :], torch.full((B, d), float(g - 1)))


def test_shift_smallest_case_M1():
    """M=1 (a single chunk): the only position is g=0, must see z0."""
    B, d = 2, 16
    h = _ramp(B, 1, d)  # value [0]
    z0 = torch.full((1, 1, d), 7.0)
    out = shift(h, z0)
    assert out.shape == (B, 1, d)
    assert torch.allclose(out[:, 0, :], torch.full((B, d), 7.0))


def test_shift_bypass_is_pass_through():
    B, M, d = 2, 4, 16
    h = _ramp(B, M, d)
    z0 = torch.full((1, 1, d), -1.0)
    out = shift(h, z0, bypass=True)
    assert torch.equal(out, h)
