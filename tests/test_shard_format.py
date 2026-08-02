"""data/shard_format.py -- uint32 shard write/read + index.json accumulation."""
import json
import os

import numpy as np
import pytest

from data import shard_format


class _FakeTokenizer:
    """Minimal stand-in for a HF tokenizer's `.get_vocab()` -- exercises
    tokenizer_hash() without a real download."""

    def __init__(self, vocab: dict):
        self._vocab = vocab

    def get_vocab(self):
        return dict(self._vocab)


def test_tokenizer_hash_deterministic_and_order_independent():
    vocab_a = {"hello": 1, "world": 2, "!": 3}
    vocab_b = {"!": 3, "world": 2, "hello": 1}  # same contents, different insertion order
    h_a = shard_format.tokenizer_hash(_FakeTokenizer(vocab_a))
    h_b = shard_format.tokenizer_hash(_FakeTokenizer(vocab_b))
    assert h_a == h_b


def test_tokenizer_hash_changes_with_vocab():
    h1 = shard_format.tokenizer_hash(_FakeTokenizer({"a": 0, "b": 1}))
    h2 = shard_format.tokenizer_hash(_FakeTokenizer({"a": 0, "b": 2}))
    assert h1 != h2


def test_write_and_read_shard_roundtrip(tmp_path):
    out_dir = str(tmp_path)
    rows = np.arange(12, dtype=np.uint32).reshape(3, 4)  # 3 rows, seq_len=4
    sidecar = shard_format.write_shard(
        out_dir, 0, rows, tokenizer_id="fake/tok", tokenizer_hash="deadbeef",
        source_dataset="unit-test")

    assert sidecar["num_rows"] == 3
    assert sidecar["seq_len"] == 4
    assert sidecar["num_tokens"] == 12
    assert sidecar["dtype"] == "uint32"

    loaded = shard_format.read_shard(out_dir, 0)
    assert loaded.shape == (3, 4)
    assert loaded.dtype == np.uint32
    assert np.array_equal(np.asarray(loaded), rows)

    # sidecar json is readable independently
    with open(os.path.join(out_dir, "shard_00000.json")) as f:
        sidecar_from_disk = json.load(f)
    assert sidecar_from_disk == sidecar


def test_read_shard_accepts_sidecar_dict_directly(tmp_path):
    out_dir = str(tmp_path)
    rows = np.arange(8, dtype=np.uint32).reshape(2, 4)
    sidecar = shard_format.write_shard(out_dir, 5, rows, "tok", "hash", "src")
    loaded = shard_format.read_shard(out_dir, sidecar)
    assert np.array_equal(np.asarray(loaded), rows)


def test_update_index_accumulates_and_is_atomic(tmp_path):
    out_dir = str(tmp_path)
    rows0 = np.zeros((2, 4), dtype=np.uint32)
    rows1 = np.ones((3, 4), dtype=np.uint32)

    sc0 = shard_format.write_shard(out_dir, 0, rows0, "tok", "hash", "src")
    shard_format.update_index(out_dir, sc0)
    sc1 = shard_format.write_shard(out_dir, 1, rows1, "tok", "hash", "src")
    index = shard_format.update_index(out_dir, sc1)

    assert len(index["shards"]) == 2
    assert index["total_rows"] == 5
    assert index["total_tokens"] == 20

    reloaded = shard_format.load_index(out_dir)
    assert reloaded == index

    # no leftover temp files from the atomic write
    leftovers = [f for f in os.listdir(out_dir) if f.startswith("index.json.")]
    assert leftovers == []


def test_write_shard_rejects_1d_rows(tmp_path):
    with pytest.raises(AssertionError):
        shard_format.write_shard(str(tmp_path), 0, np.arange(4, dtype=np.uint32),
                                  "tok", "hash", "src")
