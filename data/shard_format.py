"""S1 data-pipeline shard format (plan slide 31): flat uint32 token-id
arrays + a JSON sidecar per shard + a directory-level index.json.

vocab_size = 151,936 > uint16's max (65,535), so the usual "uint16 token
id" trick used by smaller-vocab tokenizers does not apply here -- every
shard is raw **uint32** (4 bytes/token; 0.5B tok ~= 2GB, full 10B mix
~= 40GB -- slide 31).

Layout, per shard `shard_{idx:05d}`:
    .bin   flat little-endian uint32 array, length == num_rows * seq_len
           (already packed to fixed-length rows -- see tokenize_shard.py;
           no padding, no ragged rows).
    .json  sidecar: {shard_idx, num_tokens, seq_len, num_rows, dtype,
           tokenizer_id, tokenizer_hash, source_dataset, bin_file}.
`index.json` (one per shard directory): {"shards": [sidecar, ...],
"total_tokens": N, "total_rows": M} -- rebuilt incrementally as each shard
finishes (update_index()), so a partially-tokenized run still leaves a
valid, loadable index of whatever shards completed.

tokenizer_hash is stored in every sidecar so a stream_dataset.py reader
fed shards from a DIFFERENT tokenizer is a loud mismatch, not silent
corruption -- see tokenizer_hash() below.

Atomic index write (temp file + os.replace) mirrors checkpoint/ckpt.py's
save() discipline -- a crash mid-write must not leave a torn index.json.
"""
import hashlib
import json
import os
import tempfile

import numpy as np

DTYPE = np.uint32


def tokenizer_hash(tokenizer) -> str:
    """Stable hash of a loaded tokenizer's vocab, stored in every shard
    sidecar. Deliberately over `get_vocab()` (token -> id), not the
    tokenizer's file bytes/revision string, so it's comparable across two
    independently-loaded instances of "the same" tokenizer regardless of
    how each was obtained."""
    h = hashlib.sha256()
    vocab = tokenizer.get_vocab()
    for tok, idx in sorted(vocab.items()):
        h.update(tok.encode("utf-8", errors="ignore"))
        h.update(int(idx).to_bytes(4, "little"))
    return h.hexdigest()[:16]


def _atomic_write_json(path: str, obj: dict) -> None:
    abspath = os.path.abspath(path)
    directory = os.path.dirname(abspath) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(abspath) + ".", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, abspath)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_shard(out_dir: str, shard_idx: int, rows: np.ndarray, tokenizer_id: str,
                 tokenizer_hash: str, source_dataset: str) -> dict:
    """rows: [num_rows, seq_len] uint32, already packed (no padding).
    Writes shard_{idx:05d}.bin + .json, returns the sidecar dict (caller
    passes it to update_index())."""
    assert rows.ndim == 2, f"rows must be [num_rows, seq_len], got shape {rows.shape}"
    os.makedirs(out_dir, exist_ok=True)
    num_rows, seq_len = rows.shape
    stem = f"shard_{shard_idx:05d}"
    bin_name = stem + ".bin"
    bin_path = os.path.join(out_dir, bin_name)
    rows.astype(DTYPE, copy=False).tofile(bin_path)

    sidecar = dict(
        shard_idx=shard_idx,
        num_tokens=int(num_rows) * int(seq_len),
        seq_len=int(seq_len),
        num_rows=int(num_rows),
        dtype="uint32",
        tokenizer_id=tokenizer_id,
        tokenizer_hash=tokenizer_hash,
        source_dataset=source_dataset,
        bin_file=bin_name,
    )
    with open(os.path.join(out_dir, stem + ".json"), "w") as f:
        json.dump(sidecar, f, indent=2)
    return sidecar


def read_shard(out_dir: str, shard_idx_or_sidecar) -> np.memmap:
    """Returns a read-only [num_rows, seq_len] uint32 memmap (no full
    shard load into RAM -- shards are up to ~1GB in the production mix,
    slide 31)."""
    if isinstance(shard_idx_or_sidecar, dict):
        sidecar = shard_idx_or_sidecar
    else:
        with open(os.path.join(out_dir, f"shard_{shard_idx_or_sidecar:05d}.json")) as f:
            sidecar = json.load(f)
    bin_path = os.path.join(out_dir, sidecar["bin_file"])
    return np.memmap(bin_path, dtype=DTYPE, mode="r",
                      shape=(sidecar["num_rows"], sidecar["seq_len"]))


def update_index(out_dir: str, sidecar: dict) -> dict:
    """Append `sidecar` to out_dir/index.json (creating it if missing),
    recompute totals, write atomically. Returns the updated index."""
    idx_path = os.path.join(out_dir, "index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            index = json.load(f)
    else:
        index = dict(shards=[], total_tokens=0, total_rows=0)
    index["shards"].append(sidecar)
    index["total_tokens"] = sum(s["num_tokens"] for s in index["shards"])
    index["total_rows"] = sum(s["num_rows"] for s in index["shards"])
    _atomic_write_json(idx_path, index)
    return index


def load_index(out_dir: str) -> dict:
    with open(os.path.join(out_dir, "index.json")) as f:
        return json.load(f)
