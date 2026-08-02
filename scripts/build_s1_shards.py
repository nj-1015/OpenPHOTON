"""Production S1 shard builder — token-budgeted, weighted, INTERLEAVING
4-source mixer (plan slide 30/31).

Why interleave at the document level: data/stream_dataset.py reads shards
and rows strictly in order with NO shuffling, so the source mix must be
baked into the row stream. This yields documents by weighted-random pick
across the four sources (seeded → deterministic), enforcing a per-source
TOKEN budget so the realized ratio matches 70/15/10/5.

Sources confirmed streamable (Parquet, ungated) 2026-07-26 via _probe:
  EN  70%  HuggingFaceFW/fineweb-edu     cfg sample-10BT   field 'text'
  Code15%  codeparrot/codeparrot-clean   cfg -             field 'content'
           (the-stack-dedup is gated/no-token; github-code-clean is a
            removed loading-script dataset -> both unusable here; this is
            the working ungated Parquet substitute, Python-focused.)
  JA  10%  HuggingFaceFW/fineweb-2       cfg jpn_Jpan      field 'text'
  Math 5%  open-web-math/open-web-math   cfg -             field 'text'

Reuses data/tokenize_shard.pack_texts_to_shards (tested) for the actual
packing/sharding; this file only decides WHICH text to feed and in what
proportion. (pack re-tokenizes; the small double-tokenize cost buys
guaranteed-correct shard format that stream_dataset.py can read.)

Run (on the pod, from repo root):
    HF_HOME=/workspace/hf_cache python -m scripts.build_s1_shards \
        --total-tokens 500000000 --out-dir /workspace/data/s1_shards
"""
import argparse
import os
import random
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data import shard_format, tokenize_shard  # noqa: E402

TOKENIZER_ID = "Qwen/Qwen3-0.6B"
SEQ_LEN = 2048
TOKENS_PER_SHARD = 250_000_000  # ~1GB/shard at uint32 (slide 31)

# key, dataset_id, config, split, text_field, weight
SOURCES = [
    ("en",   "HuggingFaceFW/fineweb-edu",       "sample-10BT", "train", "text",    0.70),
    ("code", "codeparrot/codeparrot-clean",     None,          "train", "content", 0.15),
    ("ja",   "HuggingFaceFW/fineweb-2",         "jpn_Jpan",    "train", "text",    0.10),
    ("math", "open-web-math/open-web-math",     None,          "train", "text",    0.05),
]


def source_stream(ds_id, cfg, split, field):
    import datasets  # optional dep, present on the pod
    ds = (datasets.load_dataset(ds_id, cfg, split=split, streaming=True)
          if cfg is not None else
          datasets.load_dataset(ds_id, split=split, streaming=True))
    for ex in ds:
        t = ex.get(field)
        if t and isinstance(t, str):
            yield t


def build(total_tokens: int, out_dir: str, seed: int = 1234):
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    budgets = {k: int(total_tokens * w) for k, _, _, _, _, w in SOURCES}
    weights = {k: w for k, _, _, _, _, w in SOURCES}
    counts = {k: 0 for k in budgets}
    iters = {k: source_stream(i, c, s, f) for k, i, c, s, f, _ in SOURCES}
    active = set(iters)
    rng = random.Random(seed)

    print(f"target total={total_tokens:,}  per-source budgets={ {k: f'{v:,}' for k, v in budgets.items()} }",
          flush=True)

    def mixed():
        yielded = 0
        while active:
            ks = list(active)
            k = rng.choices(ks, weights=[weights[x] for x in ks])[0]
            if counts[k] >= budgets[k]:
                active.discard(k)
                continue
            try:
                doc = next(iters[k])
            except StopIteration:
                active.discard(k)
                print(f"  [{k}] source exhausted at {counts[k]:,} tokens", flush=True)
                continue
            n = len(tok(doc, add_special_tokens=False)["input_ids"]) + 1  # +EOS, matches pack
            counts[k] += n
            yielded += 1
            if yielded % 20000 == 0:
                print(f"  progress: { {kk: f'{vv:,}' for kk, vv in counts.items()} }", flush=True)
            yield doc

    stats = tokenize_shard.pack_texts_to_shards(
        mixed(), tok, out_dir, seq_len=SEQ_LEN, tokens_per_shard=TOKENS_PER_SHARD,
        source_name=("s1-mix: fineweb-edu(sample-10BT) 70 / codeparrot-clean 15 / "
                     "fineweb-2(jpn_Jpan) 10 / open-web-math 5"))

    total = sum(counts.values()) or 1
    realized = {k: (v, round(100 * v / total, 2)) for k, v in counts.items()}
    print("=== DONE ===", flush=True)
    print("PER-SOURCE tokens (fed) + realized %:", realized, flush=True)
    print("PACK STATS:", stats, flush=True)
    idx = shard_format.load_index(out_dir)
    print(f"INDEX: shards={len(idx['shards'])} total_rows={idx['total_rows']:,} "
          f"total_tokens={idx['total_tokens']:,}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-tokens", type=int, default=500_000_000)
    ap.add_argument("--out-dir", default="/workspace/data/s1_shards")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    build(a.total_tokens, a.out_dir, a.seed)
