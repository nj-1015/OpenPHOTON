"""Stream text -> Qwen3 tokenizer -> pack to fixed length -> uint32 shards
(plan slide 31: "stream HF datasets -> Qwen3 tokenizer -> pack to 2048 ->
uint32 shards -> HF Hub / GCS -> stream to trainer").

Deliberately takes `texts: Iterable[str]` rather than a hardcoded dataset
id: the production mix (slide 30 -- Code 15% / Japanese 10% / Math 5% /
EN-web rest) needs Jun to pick and license-check exact dataset ids/
revisions (see configs/s1.py's TODO(Jun)); this module has no opinion on
where the text comes from. `hf_streaming_texts()` below is the intended
production adapter (HF `datasets`, streaming=True, pinned revision);
`data/sample_texts.py` is the tiny, git-ignored-shard-only smoke source
used by scripts/build_smoke_shard.py -- never a stand-in for the real mix.

Packing: token ids are concatenated across documents (EOS appended after
each document), split into fixed `seq_len` rows, no padding. The final
partial row (< seq_len tokens) is dropped, matching "pack to 2048, drop
remainder" (slide 31's own phrasing generalizes directly to any seq_len).
"""
from typing import Iterable, Optional

import numpy as np

from . import shard_format


def pack_texts_to_shards(texts: Iterable[str], tokenizer, out_dir: str, seq_len: int,
                          tokens_per_shard: int, source_name: str,
                          eos_id: Optional[int] = None) -> dict:
    """Streams `texts`, tokenizes, packs into `seq_len`-token rows, and
    writes ~`tokens_per_shard`-sized shards (data/shard_format.py) to
    `out_dir`, updating index.json after each shard. Returns summary stats.

    Buffers per-document token arrays (small numpy arrays, not a growing
    Python list of ints) and only concatenates once per shard flush, so
    memory/time cost is O(total tokens seen), not O(total^2) -- matters at
    the production 10B-token scale even though this file is exercised at
    a few-thousand-token scale by the smoke test.
    """
    eos_id = tokenizer.eos_token_id if eos_id is None else eos_id
    assert eos_id is not None, "tokenizer has no eos_token_id -- pass eos_id explicitly"
    tok_hash = shard_format.tokenizer_hash(tokenizer)
    tokenizer_id = getattr(tokenizer, "name_or_path", "unknown")

    rows_per_shard = max(1, tokens_per_shard // seq_len)
    shard_tokens = rows_per_shard * seq_len

    pending: list[np.ndarray] = []
    pending_len = 0
    shard_idx = 0
    total_tokens = 0
    total_rows = 0

    def flush():
        nonlocal pending, pending_len, shard_idx, total_tokens, total_rows
        buf = pending[0] if len(pending) == 1 else np.concatenate(pending)
        n_rows = buf.size // seq_len
        used = n_rows * seq_len
        rows = buf[:used].reshape(n_rows, seq_len)
        sidecar = shard_format.write_shard(
            out_dir, shard_idx, rows, tokenizer_id=tokenizer_id,
            tokenizer_hash=tok_hash, source_dataset=source_name)
        shard_format.update_index(out_dir, sidecar)
        total_tokens += int(used)
        total_rows += n_rows
        shard_idx += 1
        leftover = buf[used:]
        pending = [leftover] if leftover.size else []
        pending_len = int(leftover.size)

    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        arr = np.asarray(list(ids) + [eos_id], dtype=shard_format.DTYPE)
        pending.append(arr)
        pending_len += arr.size
        if pending_len >= shard_tokens:
            flush()

    if pending_len >= seq_len:
        flush()

    dropped_tail = pending_len  # whatever's left after the final flush() is < seq_len, dropped
    return dict(num_shards=shard_idx, total_tokens=total_tokens, total_rows=total_rows,
                dropped_tail_tokens=dropped_tail, tokenizer_hash=tok_hash)


def hf_streaming_texts(dataset_id: str, split: str, text_field: str,
                        revision: Optional[str] = None, limit: Optional[int] = None):
    """Production text source (slide 31: HF streaming, pinned revision).
    Requires the optional `datasets` package (not in requirements.txt --
    add it once real dataset ids are chosen; see configs/s1.py's
    TODO(Jun)). Never called with a hardcoded dataset_id from this repo --
    the caller (a future production tokenize run) supplies it."""
    import datasets  # optional dependency, only needed for this function

    ds = datasets.load_dataset(dataset_id, split=split, streaming=True, revision=revision)
    for i, example in enumerate(ds):
        if limit is not None and i >= limit:
            return
        yield example[text_field]
