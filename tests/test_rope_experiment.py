"""M6 -- RoPE reset-vs-carry, 1-layer experiment (S0 plan risk #7, slide 22
detail 3, the one thing the PHOTON paper never settles).

reset: every local-decoder chunk always sees position_ids 0..R+C-1,
regardless of where the chunk sits in the parent sequence -- by
construction (rope.decoder_position_ids("reset", ...) doesn't even take an
offset argument), so a chunk's local reconstruction is INVARIANT to its
absolute offset: the same content run through the same layer at reset
positions always produces bit-identical output. That invariance is exactly
the "parallel-safe" property S1 depends on -- every chunk becomes an
interchangeable batch row, computed the same way no matter where it sat in
the original sequence. Verified below by literally re-running the same
layer "as if" the chunk were at several different nominal offsets (reset
ignores the offset by construction, so this is expected to be, and is,
bit-exact) -- not just inspected in the code, executed.

carry: the local window instead sees its true absolute offset p (positions
p..p+R+C-1). Because RoPE is analytically a function of RELATIVE position
only (the rotation identity makes q_i . k_j depend on theta_i - theta_j,
not theta_i and theta_j separately), attention *logits* within a
fixed-width local window are, in EXACT math, also invariant to p -- sliding
the whole window doesn't change any pairwise relative offset inside it. So
the risk carry introduces isn't the attention pattern changing outright;
it's numerical: cos(p * inv_freq[0]) requires reducing a large argument mod
2*pi, and float32 loses mantissa precision doing that as p grows. Qwen3's
rope_theta=1e6 makes the fastest-rotating component's angle equal ~p
radians directly (inv_freq[0] = 1e6^0 = 1.0), so at p=32000 that's ~5000
full rotations -- computing cos(32000.0) in fp32 has measurably fewer
correct bits than cos(3.0). THAT precision loss is the "regime the
4-position window never occupied": under reset, the network's weights were
never asked to be robust to it.

This experiment measures both claims on one real, unmodified Qwen3-0.6B
decoder layer.
"""
import math

import torch

from photon import masks, rope
from photon.surgery import load_stock_qwen3

D = 1024      # hidden_size
R, C = 2, 2   # S0 default converter width / chunk size -> local window L=4
L = R + C
OFFSETS = [0, 100, 1_000, 2_048, 16_000, 32_000]  # 2048=train ctx, 16k=eval ctx, 32k~=max_position


def _one_layer_and_rotary():
    stock = load_stock_qwen3(dtype=torch.float32)
    layer = stock.model.layers[14]  # an arbitrary real, unmodified decoder layer (dec2's first)
    rotary_emb = stock.model.rotary_emb
    return layer, rotary_emb


def test_reset_mode_is_invariant_to_nominal_offset():
    """reset ignores any notion of 'where in the sequence' -- confirm by
    actually running the layer as if the same chunk sat at several
    different nominal offsets and checking the outputs are bit-identical."""
    layer, rotary_emb = _one_layer_and_rotary()
    layer.eval()

    torch.manual_seed(0)
    x = torch.randn(1, L, D)
    mask = masks.local_causal_mask(L, dtype=x.dtype)

    outputs = []
    for _nominal_offset in OFFSETS:
        # reset's decoder_position_ids doesn't take an offset argument at
        # all -- this loop exists to make the invariance an EXECUTED check,
        # not just a read of the function signature.
        pos = rope.decoder_position_ids("reset", L, batch=1)
        cos, sin = rope.build_cos_sin(rotary_emb, x, pos)
        with torch.no_grad():
            out = layer(x, attention_mask=mask, position_ids=pos, position_embeddings=(cos, sin))
        outputs.append(out)

    baseline = outputs[0]
    for i, out in enumerate(outputs[1:], start=1):
        assert torch.equal(baseline, out), (
            f"reset mode should be EXACTLY invariant to nominal offset "
            f"(offset[{i}]={OFFSETS[i]}), but output differs -- parallel-safety broken"
        )
    print(f"\n[M6] reset: bit-identical output across all {len(OFFSETS)} nominal "
          f"offsets {OFFSETS} -- confirmed parallel-safe.")


def test_carry_mode_drifts_with_absolute_offset():
    """carry: measure (a) fp32 cos/sin precision loss vs a float64
    reference as p grows, and (b) the resulting decoder-layer output drift
    vs the p=0 baseline. Both are expected to grow with p; reported, and
    checked for the qualitative ordering (large p >> small p)."""
    layer, rotary_emb = _one_layer_and_rotary()
    layer.eval()

    torch.manual_seed(0)
    x = torch.randn(1, L, D)
    mask = masks.local_causal_mask(L, dtype=x.dtype)

    # (a) fp32 cos/sin precision loss on the fastest-rotating rope component
    # (inv_freq[0] = rope_theta^0 = 1.0, so its angle for position p is p
    # radians exactly) -- compare naive fp32 cos(p) against a python
    # float64 cos(p), which does argument reduction in double precision and
    # is accurate to ~15-16 significant digits for p in this range.
    print(f"\n[M6] fp32 cos(p) precision loss vs float64 reference (fastest rope component):")
    cos_errors = {}
    for p in OFFSETS:
        naive = torch.cos(torch.tensor(float(p), dtype=torch.float32)).item()
        reference = math.cos(float(p))
        err = abs(naive - reference)
        cos_errors[p] = err
        print(f"    p={p:<7} fp32_cos={naive:.7f}  ref_cos={reference:.7f}  abs_err={err:.3e}")

    assert cos_errors[OFFSETS[0]] == 0.0, "p=0 should have zero argument-reduction error"
    assert cos_errors[OFFSETS[-1]] > cos_errors[OFFSETS[0]], (
        "expected fp32 cos precision loss to grow from p=0 to the largest offset"
    )

    # (b) full decoder-layer output drift vs the p=0 baseline, same content,
    # only the absolute base offset of the local window changes.
    print(f"\n[M6] decoder-layer output drift vs p=0 baseline (same content, carry positions):")
    outputs = {}
    for p in OFFSETS:
        chunk_offsets = torch.tensor([p], dtype=torch.long)
        pos = rope.decoder_position_ids("carry", L, batch=1, chunk_offsets=chunk_offsets)
        cos, sin = rope.build_cos_sin(rotary_emb, x, pos)
        with torch.no_grad():
            out = layer(x, attention_mask=mask, position_ids=pos, position_embeddings=(cos, sin))
        outputs[p] = out

    baseline = outputs[0]
    drifts = {}
    for p in OFFSETS[1:]:
        max_abs = (outputs[p] - baseline).abs().max().item()
        drifts[p] = max_abs
        print(f"    p={p:<7} max_abs vs p=0: {max_abs:.3e}")

    # p=100 is still a small, well-conditioned angle; p=32000 is deep in
    # the "many rotations" regime -- expect the largest offset to drift
    # substantially more than the smallest nonzero one.
    assert drifts[OFFSETS[-1]] > drifts[OFFSETS[1]], (
        "expected carry-mode drift at the largest offset to exceed drift at "
        "the smallest nonzero offset (novel-regime hypothesis)"
    )


def test_decoder_position_ids_carry_requires_chunk_offsets():
    """documents the deliberate half-wiring: carry's math is implemented
    (M6), but model.py's full hierarchical forward doesn't compute/pass
    chunk_offsets yet (S1 scope) -- calling without them must fail loudly,
    not silently fall back to something wrong."""
    import pytest
    with pytest.raises(ValueError, match="chunk_offsets"):
        rope.decoder_position_ids("carry", 4, batch=1)
