"""Integrated non-bypass causality gate (reviewer finding on T-0243's
first review pass): shape tests alone cannot detect an off-by-one or a
briefing/children order bug -- both leave every shape assertion green.
This file exercises the REAL (C1=C2=2) hierarchical forward end to end and
checks the actual causal INVARIANT the architecture promises (slide 22):
a token's reconstruction may depend on everything at-or-before it, and
NOTHING after it.

Why token-level (not just "chunk g"), and why this is strictly stronger:
  - Across chunk boundaries: encoder stages are causal per-chunk-index, and
    the local decoder briefing for chunk g always comes from the PRIOR
    chunk's reconstruction (shift()) -- so nothing in chunk g can reach
    h0_hat rows in any strictly earlier chunk.
  - WITHIN a chunk: dec1's own local causal window preserves the tokens'
    original left-to-right order (pack() puts children after briefing, in
    order; take_last_C() strips the briefing back off) -- so token offset
    c within a chunk cannot see offset c' > c in the SAME chunk either.
  Combined: h0_hat[p'] must be exactly unchanged for every p' < p when
  token p is perturbed, for ANY p (chunk-aligned or not). This subsumes
  and is stronger than a chunk-granularity check, and (worked through by
  hand, see task report) is exactly what breaks if shift() is dropped
  (a chunk's briefing starts leaking that chunk's own tail) or if pack()'s
  concat order is swapped (briefing and children rows get mixed up, and
  take_last_C() then extracts the wrong ones).

Mutation-tested against itself: with shift() temporarily removed from
model.py, or pack()'s concat order swapped, this test FAILS (confirmed
manually during development; see the task report for the two failure
transcripts -- not re-run automatically here since sabotaging source in a
committed test would defeat the point).
"""
import torch
import pytest

from photon.config import PhotonConfig
from photon.reshape import pack
from photon.surgery import load_photon_qwen3


@pytest.fixture(scope="module")
def photon_real():
    """Real (non-bypass) hierarchical model, default C1=C2=R1=R2=2."""
    cfg = PhotonConfig(C1=2, C2=2, R1=2, R2=2)
    photon, stock = load_photon_qwen3(cfg)
    return photon, stock


def test_token_level_causality_perturbation(photon_real):
    photon, stock = photon_real
    vocab = stock.config.vocab_size
    B, T = 1, 16   # 8 level-1 chunks (C1=2), 4 level-2 chunks (C2=2)

    torch.manual_seed(0)
    ids = torch.randint(0, vocab, (B, T))

    # perturb at a spread of positions: early / chunk-boundary-aligned /
    # mid-chunk (tests intra-chunk ordering) / last token.
    for p in (1, 8, 9, 15):
        ids2 = ids.clone()
        ids2[0, p] = (ids2[0, p].item() + 1) % vocab
        assert ids2[0, p].item() != ids[0, p].item()

        with torch.no_grad():
            _, mid1 = photon(ids, return_intermediates=True)
            _, mid2 = photon(ids2, return_intermediates=True)

        h0_hat_1 = mid1["h0_hat"][0]
        h0_hat_2 = mid2["h0_hat"][0]

        if p > 0:
            before = torch.equal(h0_hat_1[:p], h0_hat_2[:p])
            assert before, (
                f"CAUSALITY VIOLATION at p={p}: h0_hat rows before the "
                f"perturbed token changed (max_abs="
                f"{(h0_hat_1[:p] - h0_hat_2[:p]).abs().max().item():.3e})"
            )

        after_changed = not torch.equal(h0_hat_1[p:], h0_hat_2[p:])
        assert after_changed, (
            f"perturbing token {p} had NO effect on h0_hat[{p}:] -- "
            f"suspicious (dead input, or an off-by-one hiding a no-op)"
        )


def test_pack_briefing_then_children_row_layout():
    """Direct value test (not shape, not the take_last_C round-trip): pack()
    must lay out [briefing (R rows) ; children (C rows), in original order]
    per chunk, and the B*M batch-dim mapping must be row-major
    (row = b*M + m), matching take_last_C()'s inverse (`M = BM // B`)."""
    B, M, R, C, d = 2, 3, 2, 2, 4

    u = torch.zeros(B, M, R, d)
    x = torch.zeros(B, M * C, d)
    for b in range(B):
        for m in range(M):
            for r in range(R):
                u[b, m, r, :] = 1000 * b + 100 + m * 10 + r  # identifiable, distinct from x
            for c in range(C):
                x[b, m * C + c, :] = 1000 * b + m * 10 + c

    z = pack(u, x)
    assert z.shape == (B * M, R + C, d)

    for b in range(B):
        for m in range(M):
            row = z[b * M + m]                     # row-major B*M mapping
            for r in range(R):
                assert torch.equal(row[r], u[b, m, r]), \
                    f"briefing row {r} misplaced at batch-row (b={b},m={m})"
            for c in range(C):
                assert torch.equal(row[R + c], x[b, m * C + c]), \
                    f"children row {c} misplaced at batch-row (b={b},m={m}) -- " \
                    f"order or offset wrong"
