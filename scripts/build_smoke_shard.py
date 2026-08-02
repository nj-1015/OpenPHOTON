"""Build the local, git-ignored smoke-test mini-shard (data/_smoke_shards/)
from data/sample_texts.py's tiny public-domain sample. Run ONCE, locally;
tests/test_s1_smoke.py's fixture calls this automatically (via subprocess,
same pattern as tests/test_golden_bf16.py's `ref` fixture regenerating
_ref/qwen3_0.6b_ref.npz) if the shard directory doesn't already exist.

Zero Colab CU: a CPU tokenization pass using the Qwen3-0.6B tokenizer
already locally HF-cached from S0's own test runs. NOT a stand-in for the
production data mix -- see configs/s1.py's TODO(Jun) for that.
"""
import os
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data import sample_texts, shard_format, tokenize_shard  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "_smoke_shards")
SEQ_LEN = 128
TOKENS_PER_SHARD = 20_000   # small on purpose -- exercises the loader's real multi-shard wraparound
N_REPEATS = 40              # cycles of data/sample_texts.py's paragraphs


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    texts = sample_texts.repeated(N_REPEATS)
    stats = tokenize_shard.pack_texts_to_shards(
        texts, tok, OUT_DIR, seq_len=SEQ_LEN, tokens_per_shard=TOKENS_PER_SHARD,
        source_name="smoke-public-domain-sample (NOT production data, see configs/s1.py TODO(Jun))")
    print("smoke shard stats:", stats)
    index = shard_format.load_index(OUT_DIR)
    print(f"shards={len(index['shards'])} total_rows={index['total_rows']} "
          f"total_tokens={index['total_tokens']}")


if __name__ == "__main__":
    main()
