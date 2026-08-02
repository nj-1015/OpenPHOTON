"""data/tokenize_shard.py -- pack_texts_to_shards() with a fake tokenizer
(no network/HF download needed for this file's own logic)."""
import numpy as np

from data import shard_format, tokenize_shard


class _FakeTokenizer:
    """Word-count tokenizer: token id == word's position in a fixed small
    vocab. Deterministic, no network -- exercises the packing/shard-
    boundary logic in isolation from a real BPE tokenizer."""
    name_or_path = "fake/word-tokenizer"
    eos_token_id = 999

    def __call__(self, text, add_special_tokens=False):
        ids = [hash(w) % 900 for w in text.split()]
        return {"input_ids": ids}

    def get_vocab(self):
        return {f"word{i}": i for i in range(900)}


def test_pack_texts_to_shards_produces_full_rows_no_padding(tmp_path):
    tok = _FakeTokenizer()
    texts = [f"doc number {i} has several words in it" for i in range(20)]
    out_dir = str(tmp_path)
    stats = tokenize_shard.pack_texts_to_shards(
        texts, tok, out_dir, seq_len=8, tokens_per_shard=64, source_name="unit-test")

    index = shard_format.load_index(out_dir)
    assert index["total_rows"] == stats["total_rows"]
    assert index["total_tokens"] == stats["total_tokens"]
    assert stats["total_tokens"] % 8 == 0  # every row is a full seq_len=8 row

    for sidecar in index["shards"]:
        rows = shard_format.read_shard(out_dir, sidecar)
        assert rows.shape[1] == 8
        assert rows.dtype == np.uint32


def test_pack_texts_to_shards_drops_final_partial_row(tmp_path):
    tok = _FakeTokenizer()
    # exactly 2 documents of 3 words + eos = 4 tokens each = 8 tokens total,
    # not a multiple of seq_len=5 -- expect the trailing 3 tokens dropped.
    texts = ["alpha beta gamma", "delta epsilon zeta"]
    out_dir = str(tmp_path)
    stats = tokenize_shard.pack_texts_to_shards(
        texts, tok, out_dir, seq_len=5, tokens_per_shard=100, source_name="unit-test")

    assert stats["total_tokens"] == 5  # one full row of 5, 3 tokens dropped
    assert stats["dropped_tail_tokens"] == 3
    assert stats["total_rows"] == 1


def test_pack_texts_to_shards_records_tokenizer_hash_in_sidecar(tmp_path):
    tok = _FakeTokenizer()
    out_dir = str(tmp_path)
    tokenize_shard.pack_texts_to_shards(
        ["one two three four five six seven eight"], tok, out_dir, seq_len=4,
        tokens_per_shard=100, source_name="unit-test")
    index = shard_format.load_index(out_dir)
    expected_hash = shard_format.tokenizer_hash(tok)
    for sidecar in index["shards"]:
        assert sidecar["tokenizer_hash"] == expected_hash
        assert sidecar["tokenizer_id"] == "fake/word-tokenizer"


def test_pack_texts_to_shards_splits_across_multiple_shards(tmp_path):
    tok = _FakeTokenizer()
    texts = [f"word{i} another one here please friend today" for i in range(50)]
    out_dir = str(tmp_path)
    stats = tokenize_shard.pack_texts_to_shards(
        texts, tok, out_dir, seq_len=4, tokens_per_shard=16, source_name="unit-test")
    index = shard_format.load_index(out_dir)
    assert stats["num_shards"] > 1
    assert len(index["shards"]) == stats["num_shards"]
