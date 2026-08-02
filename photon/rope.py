"""Position ids and rotary (cos, sin) for each stage.

Reuses the stock `Qwen3RotaryEmbedding` instance carried over from the loaded
HF model (photon/surgery.py keeps `stock_model.model.rotary_emb` verbatim) so
cos/sin values are bit-identical to what the stock model would compute --
no rope math is reimplemented here, only *which* position_ids each stage
sees (slide 22, detail 3).

rope_mode:
    "reset" -- every local-decoder chunk sees absolute positions
               0..R+C-1 (parallel-safe, S0 default). Encoders (enc1, enc2)
               always use absolute positions over their own sequence length;
               they never reset, so this flag only affects dec1/dec2.
    "carry" -- local decoders see their true absolute offset in the parent
               stream: chunk row r gets positions chunk_offsets[r] ..
               chunk_offsets[r]+length-1 instead of resetting to 0..L-1.
               M6 (tests/test_rope_experiment.py) implements and measures
               this; model.py's full hierarchical forward does NOT yet
               compute/pass chunk_offsets end-to-end (that wiring is S1
               scope, per the S0 plan -- calling decoder_position_ids with
               rope_mode="carry" and no chunk_offsets raises a clear
               ValueError rather than silently doing the wrong thing).
"""
import torch


def absolute_position_ids(length: int, batch: int, device=None) -> torch.Tensor:
    """[B, length] position ids 0..length-1, used by enc1/enc2 always, and by
    the local decoders when rope_mode == 'reset'."""
    return torch.arange(length, device=device).unsqueeze(0).expand(batch, -1)


def decoder_position_ids(rope_mode: str, length: int, batch: int, device=None,
                          chunk_offsets: torch.Tensor | None = None) -> torch.Tensor:
    """Position ids for a local-decoder stage (dec1 or dec2).

    chunk_offsets (rope_mode == "carry" only): one absolute base position
    per batch row (shape [batch], the B*M packed-chunk dimension). Row r's
    positions are chunk_offsets[r] + arange(length) instead of resetting to
    0..length-1. Ignored (and not required) for rope_mode == "reset" --
    reset's whole point is that the local window's positions never depend
    on where the chunk sits in the parent sequence, which is what makes
    reset-mode chunks parallel-safe / interchangeable batch rows.
    """
    if rope_mode == "reset":
        return absolute_position_ids(length, batch, device=device)
    if rope_mode == "carry":
        if chunk_offsets is None:
            raise ValueError(
                "rope_mode='carry' requires chunk_offsets (one absolute base "
                "position per batch row) -- model.py's full hierarchical "
                "forward does not compute/pass these yet (S1 scope, per the "
                "S0 plan). For the M6 1-layer experiment or any direct use, "
                "pass chunk_offsets explicitly."
            )
        offsets = chunk_offsets.to(device=device).view(batch, 1)
        return offsets + torch.arange(length, device=device).unsqueeze(0)
    raise ValueError(f"unknown rope_mode {rope_mode!r}")


def build_cos_sin(rotary_emb: torch.nn.Module, hidden_states: torch.Tensor,
                   position_ids: torch.Tensor):
    """cos, sin for `position_ids`, computed by the (verbatim, kept) stock
    Qwen3RotaryEmbedding module."""
    return rotary_emb(hidden_states, position_ids)
