"""Additive causal masks fed straight into `Qwen3DecoderLayer.forward`'s
`attention_mask` argument (eager attention does `attn_weights + attention_mask`
-- see transformers/models/qwen3/modeling_qwen3.py::eager_attention_forward).

One builder covers both uses in the S0 spec: the encoders' full causal mask
over their own (shrinking) sequence length, and the local decoders' causal
mask over the R+C window (slide 22, detail 2: "plain lower-triangular over
[briefing ; children] -- the same causal mask you already have, just 4x4
instead of TxT").
"""
import torch


def causal_mask(length: int, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
    """[1, 1, length, length] additive mask, 0 where allowed, dtype-min above
    the diagonal (query attends to itself and everything before it)."""
    neg = torch.finfo(dtype).min
    m = torch.full((length, length), neg, dtype=dtype, device=device)
    m = torch.triu(m, diagonal=1)
    return m.view(1, 1, length, length)


# Aliases kept distinct for readability at call sites (slide 21/22): the
# encoders use the full-T variant, the local decoders use the R+C variant.
# Both are the same lower-triangular construction.
full_causal_mask = causal_mask
local_causal_mask = causal_mask
