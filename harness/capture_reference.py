"""Capture stock Qwen3-0.6B logits + per-layer hidden states to a .npz, for
the M5 golden-test comparison.

fp32/CPU/eager, output_hidden_states=True, deterministic (no fused kernels)
so the captured numbers are a fair floating-point reference.

Four cases (reviewer finding on T-0243's first review pass: a single short
B=1 prompt under-tests the full-causal path -- a length- or batch-dependent
mask bug could hide behind it):
  p0    -- the original short prompt, B=1.
  p1    -- a longer prompt, B=1 (stresses mask/position-id construction at
           a length p0 never exercises).
  p2    -- a second, differently-worded prompt of comparable length to p1
           (exists only to pair with p1 for the batch case below).
  batch -- p1 and p2's *token ids*, truncated to their common length and
           stacked to B=2 (stresses batch-dim mask broadcasting -- a mask
           built with an assumed batch size of 1 would silently break here
           -- at a length comparable to p1, not degenerated down to p0's 5
           tokens, which truncating against p0 would have done).
"""
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPTS = {
    "p0": "The capital of France is",
    "p1": ("In the year 1969, the first humans landed on the surface of the "
           "Moon, and the whole world watched the historic event unfold "
           "live on television while scientists back on Earth held their"),
    "p2": ("The chef carefully arranged the vegetables on the plate, added "
           "a drizzle of olive oil, and explained to the nervous trainee "
           "exactly why searing the fish first would keep it from"),
}
BATCH_PAIR = ("p1", "p2")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "_ref", "qwen3_0.6b_ref.npz"
)


def _capture(model, tok, name, ids, out):
    with torch.no_grad():
        result = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    logits = result.logits.float().numpy()                    # (B, T, vocab)
    hs = [h.float().numpy() for h in result.hidden_states]     # each (B, T, d)
    out[f"{name}_input_ids"] = ids.numpy()
    out[f"{name}_logits"] = logits
    for i, h in enumerate(hs):
        out[f"{name}_hs_{i}"] = h
    print(f"{name}: input_ids.shape={tuple(ids.shape)}  logits.shape={logits.shape}  "
          f"num_hidden_states={len(hs)}")


def main():
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    print("model:", MODEL_ID)

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    print("model class:", model.__class__.__name__)

    out = {}
    ids_by_name = {}
    for name, prompt in PROMPTS.items():
        ids = tok(prompt, return_tensors="pt").input_ids  # [1, T_name]
        ids_by_name[name] = ids
        _capture(model, tok, name, ids, out)

    last = out["p0_logits"][0, -1]
    top5 = np.argsort(last)[-5:][::-1]
    print("p0 top5 next tokens:", [(int(t), tok.decode([int(t)])) for t in top5])

    # batched (B=2) case: p1+p2 (comparable length), truncated to their
    # common length so no padding/attention-mask machinery is needed --
    # deliberately NOT paired with the much-shorter p0, or the batch case
    # would degenerate to p0's length and stop testing anything new.
    batch_ids_list = [ids_by_name[name] for name in BATCH_PAIR]
    min_len = min(ids.shape[1] for ids in batch_ids_list)
    batch_ids = torch.cat([ids[:, :min_len] for ids in batch_ids_list], dim=0)  # [2, min_len]
    _capture(model, tok, "batch", batch_ids, out)

    np.savez(OUT, **out)
    print("saved ->", OUT, f"({len(out)} arrays)")


if __name__ == "__main__":
    main()
