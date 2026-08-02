"""Resumable streaming reader over uint32 shards (data/shard_format.py),
feeding checkpoint/ckpt.py's `DataloaderState` directly (plan: "Resume =
(shard_idx, offset) via ckpt.DataloaderState").

State is exactly `(shard_idx, offset)`: `shard_idx` indexes into
index.json's shard list (wrapping around for multi-epoch runs over a small
shard set -- the smoke test's mini-shard is far smaller than 50 steps'
worth of unique rows), `offset` is the next unread ROW inside that shard.
Reading is fully deterministic (shard order + row order, no intra-shard
shuffling) -- the resumable unit is exactly the pair `DataloaderState`
already carries, so this reader never needs anything beyond it.
`rng_state` is still threaded through (round-tripped, not consumed) purely
to satisfy `DataloaderState`'s existing contract/shape for forward
compatibility with a future shard-order shuffle -- see ckpt.py's own
docstring on what the dataclass is for.
"""
import numpy as np
import torch

from checkpoint.ckpt import DataloaderState

from . import shard_format


class ShardStreamLoader:
    def __init__(self, shard_dir: str, batch_size: int, shard_idx: int = 0, offset: int = 0):
        self.shard_dir = shard_dir
        self.index = shard_format.load_index(shard_dir)
        self.shards = self.index["shards"]
        if not self.shards:
            raise ValueError(f"no shards found in index.json at {shard_dir!r}")
        self.batch_size = batch_size
        self.shard_idx = shard_idx % len(self.shards)
        self.offset = offset
        self._rows = shard_format.read_shard(shard_dir, self.shards[self.shard_idx])
        if not (0 <= self.offset <= self._rows.shape[0]):
            raise ValueError(
                f"offset {offset} out of range for shard {self.shard_idx} "
                f"({self._rows.shape[0]} rows)")

    def _advance_shard(self) -> None:
        self.shard_idx = (self.shard_idx + 1) % len(self.shards)
        self.offset = 0
        self._rows = shard_format.read_shard(self.shard_dir, self.shards[self.shard_idx])

    def next_batch(self) -> torch.LongTensor:
        """[batch_size, seq_len] int64 token ids. Wraps to the next shard
        (and, at the end of the shard list, back to shard 0 -- multi-epoch)
        whenever the current shard is exhausted."""
        rows = []
        while len(rows) < self.batch_size:
            if self.offset >= self._rows.shape[0]:
                self._advance_shard()
            rows.append(np.array(self._rows[self.offset]))  # copies out of the memmap
            self.offset += 1
        batch = np.stack(rows).astype(np.int64)
        return torch.from_numpy(batch)

    def state(self) -> DataloaderState:
        """Snapshot the position BEFORE the next `next_batch()` call --
        mirrors tests/test_checkpoint_roundtrip.py's discipline (capture
        the state that, once resumed, reproduces the identical next draw)."""
        return DataloaderState(shard_idx=self.shard_idx, offset=self.offset,
                                rng_state=torch.get_rng_state())

    @classmethod
    def from_state(cls, shard_dir: str, batch_size: int, state: DataloaderState) -> "ShardStreamLoader":
        loader = cls(shard_dir, batch_size, shard_idx=state.shard_idx, offset=state.offset)
        torch.set_rng_state(state.rng_state)
        return loader
