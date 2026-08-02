"""Checkpoint/resume: model + optimizer state_dict + dataloader
position (shard idx + intra-shard offset + RNG state), round-tripped
through a local file.

`path` is intentionally just a local filesystem path -- swapping in
cloud-backed storage (e.g. a bucket URI) for larger training runs is a
path-handling detail behind the same save()/load() API, not a scope change
here; callers that later pass a bucket-style URI just need a small io shim,
this module's contract (what gets saved, what round-trips bit-identically)
doesn't change.

Atomic write (temp file + os.replace) -- a crash mid-write must not leave a
torn checkpoint on disk.
"""
import os
import tempfile
from dataclasses import dataclass

import torch


@dataclass
class DataloaderState:
    """Enough to resume a dataloader exactly where it left off: which
    shard, how far into that shard (tokens/examples/bytes -- caller's
    choice of unit), and the RNG state governing shuffling/sampling."""
    shard_idx: int
    offset: int
    rng_state: torch.Tensor  # torch.get_rng_state() -- a ByteTensor

    def to_dict(self) -> dict:
        return dict(shard_idx=self.shard_idx, offset=self.offset, rng_state=self.rng_state)

    @classmethod
    def from_dict(cls, d: dict) -> "DataloaderState":
        return cls(shard_idx=d["shard_idx"], offset=d["offset"], rng_state=d["rng_state"])


def _state_dict_of(obj) -> dict:
    """Accept either a raw state_dict already, or an object with
    .state_dict() (nn.Module / torch.optim.Optimizer)."""
    if isinstance(obj, dict):
        return obj
    return obj.state_dict()


def save(path: str, model, optimizer, dataloader_state: DataloaderState) -> None:
    """model / optimizer: either a state_dict already, or an object with
    .state_dict() (nn.Module / torch.optim.Optimizer -- the common case)."""
    ckpt = dict(
        model_state=_state_dict_of(model),
        optimizer_state=_state_dict_of(optimizer),
        dataloader_state=dataloader_state.to_dict(),
    )
    abspath = os.path.abspath(path)
    directory = os.path.dirname(abspath) or "."
    os.makedirs(directory, exist_ok=True)

    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(abspath) + ".", dir=directory)
    os.close(fd)
    try:
        torch.save(ckpt, tmp)
        os.replace(tmp, abspath)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load(path: str, model=None, optimizer=None):
    """Returns (model_state, optimizer_state, dataloader_state). If `model`/
    `optimizer` OBJECTS (not raw dicts) are passed, also calls
    .load_state_dict() on them in place, so the common case is just:

        model_state, opt_state, dl_state = ckpt.load(path, model, optimizer)

    weights_only=False: this is a fully local/trusted checkpoint file (not
    an untrusted download) -- optimizer state_dicts carry plain Python
    ints/floats in param_groups that torch's weights_only unpickler
    restricts, so the safer-by-default loader isn't applicable here.
    """
    ckpt = torch.load(path, weights_only=False)
    model_state = ckpt["model_state"]
    optimizer_state = ckpt["optimizer_state"]
    dataloader_state = DataloaderState.from_dict(ckpt["dataloader_state"])

    if model is not None and not isinstance(model, dict):
        model.load_state_dict(model_state)
    if optimizer is not None and not isinstance(optimizer, dict):
        optimizer.load_state_dict(optimizer_state)

    return model_state, optimizer_state, dataloader_state
