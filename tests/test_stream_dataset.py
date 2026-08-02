"""data/stream_dataset.py -- ShardStreamLoader resumability, CPU-only, no
model/GPU needed. Same "simulated kill" discipline as
tests/test_checkpoint_roundtrip.py, scoped to just the dataloader.
"""
import numpy as np
import torch

from data import shard_format
from data.stream_dataset import ShardStreamLoader


def _build_shards(out_dir, num_shards=3, rows_per_shard=4, seq_len=4):
    """Shard k's rows are filled with value k*1000 + row_index, so the
    exact (shard, row) a batch came from is recoverable by inspection."""
    for k in range(num_shards):
        rows = np.stack([
            np.full(seq_len, k * 1000 + r, dtype=np.uint32) for r in range(rows_per_shard)
        ])
        sidecar = shard_format.write_shard(out_dir, k, rows, "tok", "hash", "src")
        shard_format.update_index(out_dir, sidecar)


def test_loader_reads_rows_in_order_and_wraps_shards(tmp_path):
    out_dir = str(tmp_path)
    _build_shards(out_dir, num_shards=2, rows_per_shard=2, seq_len=4)
    loader = ShardStreamLoader(out_dir, batch_size=1)

    # shard 0: rows 0, 1 -> then wrap to shard 1: rows 0, 1 -> then wrap
    # back to shard 0: row 0 again
    expected_first_values = [0, 1, 1000, 1001, 0]
    for expected in expected_first_values:
        batch = loader.next_batch()
        assert batch.shape == (1, 4)
        assert int(batch[0, 0].item()) == expected


def test_loader_batches_multiple_rows(tmp_path):
    out_dir = str(tmp_path)
    _build_shards(out_dir, num_shards=1, rows_per_shard=4, seq_len=4)
    loader = ShardStreamLoader(out_dir, batch_size=3)
    batch = loader.next_batch()
    assert batch.shape == (3, 4)
    assert [int(batch[i, 0].item()) for i in range(3)] == [0, 1, 2]


def test_resume_from_state_reproduces_identical_next_batches(tmp_path):
    out_dir = str(tmp_path)
    _build_shards(out_dir, num_shards=3, rows_per_shard=4, seq_len=4)

    # --- uninterrupted reference: draw 5 batches ---
    ref_loader = ShardStreamLoader(out_dir, batch_size=2)
    ref_batches_pre = [ref_loader.next_batch() for _ in range(3)]
    state_at_3 = ref_loader.state()  # capture BEFORE drawing batch 4
    ref_batches_post = [ref_loader.next_batch() for _ in range(2)]

    # --- simulated kill: drop the loader, reconstruct fresh from state ---
    del ref_loader
    resumed_loader = ShardStreamLoader.from_state(out_dir, batch_size=2, state=state_at_3)
    resumed_batches = [resumed_loader.next_batch() for _ in range(2)]

    for a, b in zip(ref_batches_post, resumed_batches):
        assert torch.equal(a, b)

    # sanity: the pre-checkpoint batches are NOT the same as the post ones
    # (else this test would pass vacuously on an always-returns-batch-0 bug)
    assert not any(torch.equal(pre, post) for pre in ref_batches_pre for post in resumed_batches)


def test_state_offset_out_of_range_raises(tmp_path):
    out_dir = str(tmp_path)
    _build_shards(out_dir, num_shards=1, rows_per_shard=2, seq_len=4)
    import pytest
    with pytest.raises(ValueError):
        ShardStreamLoader(out_dir, batch_size=1, shard_idx=0, offset=99)
