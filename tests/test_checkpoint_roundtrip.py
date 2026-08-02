"""M7 -- checkpoint round-trip: model + optimizer state_dict + dataloader
position (shard idx + offset + RNG state) survive a simulated kill
bit-identically. CPU-only, no GPU required.

"Simulated kill" = drop every in-memory object (model, optimizer, RNG
state) and start over from fresh, uninitialized ones, then load() and
compare against what the ORIGINAL run's state actually was -- not just
that save-then-immediately-load round-trips (trivial), but that a genuinely
independent, freshly-constructed model/optimizer/RNG ends up bit-identical
to the pre-kill state after loading.
"""
import os

import pytest
import torch
import torch.nn as nn

from checkpoint.ckpt import DataloaderState, load, save


def _tiny_model():
    torch.manual_seed(123)
    return nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))


def test_checkpoint_roundtrip_after_simulated_kill(tmp_path):
    ckpt_path = str(tmp_path / "ckpt.pt")

    # --- pre-kill: train a couple of steps so optimizer state (Adam's
    # exp_avg / exp_avg_sq momentum buffers) is actually populated, and
    # advance the RNG so its state isn't just the fresh-seed default ---
    model = _tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    torch.manual_seed(7)
    for _ in range(3):
        optimizer.zero_grad()
        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

    # snapshot the RNG state HERE (before drawing the "next batch" below),
    # so we can later prove resuming reproduces the identical RNG stream
    rng_state_at_save = torch.get_rng_state().clone()
    expected_next_batch = torch.randn(2, 8)  # what would come next, unsaved

    dl_state = DataloaderState(shard_idx=3, offset=123_456, rng_state=rng_state_at_save)
    save(ckpt_path, model, optimizer, dl_state)

    # sanity: optimizer state really is populated (else the test would pass
    # vacuously -- there'd be nothing to break)
    assert len(optimizer.state) > 0
    original_model_tensors = {k: v.clone() for k, v in model.state_dict().items()}
    # optimizer.state is keyed by the live Parameter OBJECT, which won't
    # match across two independently-constructed model instances -- compare
    # via state_dict()'s integer-indexed form instead (the same remapping
    # save()/load() themselves go through).
    original_opt_state = {
        idx: {k: v.clone() for k, v in st.items() if torch.is_tensor(v)}
        for idx, st in optimizer.state_dict()["state"].items()
    }

    # --- simulated kill: drop everything, start genuinely fresh ---
    del model, optimizer
    torch.manual_seed(999)  # scramble the RNG to something unrelated
    _ = torch.randn(5)      # burn some draws too, for good measure

    torch.manual_seed(4242)  # different init seed than the pre-kill model used
    fresh_model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    fresh_optimizer = torch.optim.Adam(fresh_model.parameters(), lr=1e-2)
    assert len(fresh_optimizer.state) == 0  # genuinely fresh, no momentum yet

    # --- resume ---
    model_state, opt_state, loaded_dl_state = load(ckpt_path, fresh_model, fresh_optimizer)

    # 1. model weights: bit-identical
    for k, v in original_model_tensors.items():
        assert torch.equal(v, fresh_model.state_dict()[k]), f"model tensor {k!r} mismatch after resume"

    # 2. optimizer momentum tensors: bit-identical (compared via the same
    # integer-indexed state_dict() form used above, not live .state, whose
    # keys are Parameter objects that differ by identity across instances)
    resumed_opt_state = fresh_optimizer.state_dict()["state"]
    assert set(resumed_opt_state.keys()) == set(original_opt_state.keys())
    for idx, st in original_opt_state.items():
        resumed_st = resumed_opt_state[idx]
        for k, v in st.items():
            assert torch.equal(v, resumed_st[k]), f"optimizer state {k!r} (param idx {idx}) mismatch after resume"

    # 3. dataloader position: bit-identical shard/offset
    assert loaded_dl_state.shard_idx == 3
    assert loaded_dl_state.offset == 123_456
    assert torch.equal(loaded_dl_state.rng_state, rng_state_at_save)

    # 4. the actual point of saving RNG state: resuming reproduces the
    # IDENTICAL next-batch draw, not just an identical byte tensor sitting
    # unused
    torch.set_rng_state(loaded_dl_state.rng_state)
    resumed_next_batch = torch.randn(2, 8)
    assert torch.equal(resumed_next_batch, expected_next_batch), (
        "RNG stream did not resume identically -- next batch differs"
    )


def test_checkpoint_save_is_atomic_no_torn_file_on_crash(tmp_path, monkeypatch):
    """a crash mid-write must not leave a corrupt/partial file at `path`."""
    import checkpoint.ckpt as ckpt_mod

    ckpt_path = str(tmp_path / "ckpt.pt")
    model = _tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    dl_state = DataloaderState(shard_idx=0, offset=0, rng_state=torch.get_rng_state())

    def _boom(*a, **k):
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr(ckpt_mod.torch, "save", _boom)
    with pytest.raises(RuntimeError):
        save(ckpt_path, model, optimizer, dl_state)

    assert not os.path.exists(ckpt_path), "a failed save must not leave a file at the destination path"
    leftover_tmp = [f for f in os.listdir(tmp_path) if f.startswith("ckpt.pt.")]
    assert leftover_tmp == [], f"a failed save must not leave a temp file behind either: {leftover_tmp}"
