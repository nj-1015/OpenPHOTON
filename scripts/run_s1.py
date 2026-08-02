"""Standalone CLI launcher for S1 runs -- mirrors a training notebook's
local smoke gate and real production-run logic, for use outside a
notebook (e.g. a cloud GPU pod via scripts/run_s1_pod.sh).

`--mode production` without `--go` only prints the run plan and exits --
same default-safe behavior as the notebook's `RUN_PRODUCTION_S1 = False`.
This script does not tokenize data itself (that's the tokenization
pipeline's job, see scripts/build_s1_shards.py) and does not pick dataset
ids (configs/s1.py's `DATASET_MIX_TODO` stays empty until a maintainer
fills it in) -- it only trains against an already-tokenized shard_dir.

Usage:
    python scripts/run_s1.py --mode smoke
    python scripts/run_s1.py --mode production \
        --shard-dir /workspace/data/s1_shards \
        --ckpt-dir  /workspace/checkpoints/s1 \
        [--resume] [--go]
"""
import argparse
import os
import runpy
import sys
import time

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO_ROOT)


def run_smoke() -> None:
    """Delegates to scripts/smoke_s1.py in-process (not a subprocess) so a
    failure surfaces as a normal traceback with a non-zero exit code."""
    runpy.run_path(os.path.join(REPO_ROOT, "scripts", "smoke_s1.py"), run_name="__main__")


def run_production(shard_dir: str, ckpt_dir: str, do_resume: bool, go: bool) -> None:
    from configs.s1 import DATASET_MIX_TODO, S1Config
    from harness.g1_metrics import g1a_report
    from train.train_s1 import build_state, checkpoint, train_step
    from train.train_s1 import resume as resume_state

    cfg = S1Config()
    cfg.shard_dir = shard_dir
    cfg.ckpt_path = os.path.join(ckpt_dir, "ckpt.pt")

    if not os.path.exists(os.path.join(shard_dir, "index.json")):
        print(f"REFUSING: no index.json under {shard_dir!r} -- run T-0244's tokenize "
              f"pipeline to produce real shards there first; this script does not "
              f"tokenize data itself.")
        sys.exit(1)

    steps_total = cfg.total_tokens // (cfg.micro_batch_size * cfg.seq_len * cfg.grad_accum_steps)
    print("=== S1 PRODUCTION RUN PLAN ===")
    print(f"  shard_dir:        {shard_dir}")
    print(f"  ckpt_path:        {cfg.ckpt_path}")
    print(f"  total_tokens:     {cfg.total_tokens:,}")
    print(f"  planned steps:    {steps_total}")
    print(f"  micro_batch_size: {cfg.micro_batch_size}  grad_accum_steps: {cfg.grad_accum_steps}")
    print(f"  seq_len:          {cfg.seq_len}")
    print(f"  lr:               {cfg.lr}")
    print(f"  use_8bit_adam:    {cfg.use_8bit_adam}")
    print(f"  DATASET_MIX_TODO set: {bool(DATASET_MIX_TODO)}  (tokenize-time only -- this "
          f"trainer reads shard_dir directly, not the mix config)")
    print(f"  resume:           {do_resume}")

    if not go:
        print("\n--go not passed -- plan printed, NOT starting training (same "
              "default-safe behavior as notebooks/S1_train.ipynb's RUN_PRODUCTION_S1=False).")
        return

    os.makedirs(ckpt_dir, exist_ok=True)
    if do_resume and os.path.exists(cfg.ckpt_path):
        state, opt_name = resume_state(cfg, cfg.ckpt_path, device="cuda")
        # Resume step-accounting (2026-07-27 coordinator fix, improved by
        # the checkpoint-metadata change): train/train_s1.checkpoint now
        # stores `state.step` in the checkpoint's metadata dict, and
        # train_s1.resume restores it -- so a resumed invocation continues
        # the SAME budget (`for step in range(steps_done, steps_total)`)
        # instead of restarting a fresh full steps_total. The dataloader
        # position remains the source of truth for pre-metadata checkpoints
        # (metadata step falls back to 0, so it's re-derived below).
        steps_done = state.step
        if steps_done == 0:
            _idx = state.loader.index
            _rows_done = (sum(s["num_rows"] for s in _idx["shards"][:state.loader.shard_idx])
                          + state.loader.offset)
            steps_done = _rows_done // cfg.micro_batch_size
        print(f"resumed from {cfg.ckpt_path}, optimizer={opt_name}, "
              f"steps_done={steps_done}/{steps_total} (step from ckpt metadata, "
              f"or dataloader pos for pre-metadata checkpoints)")
    else:
        state, opt_name = build_state(cfg, device="cuda")
        steps_done = 0
        print("optimizer:", opt_name)

    last_ckpt_time = time.time()
    for step in range(steps_done, steps_total):
        losses = train_step(state)
        if step % 50 == 0:
            print(f"step {step}/{steps_total}", losses)
        if time.time() - last_ckpt_time > cfg.ckpt_every_seconds:
            checkpoint(state, cfg.ckpt_path)
            last_ckpt_time = time.time()
            print(f"checkpointed at step {step} -> {cfg.ckpt_path}")

    checkpoint(state, cfg.ckpt_path)
    print("done, final checkpoint saved to", cfg.ckpt_path)

    report = g1a_report(state.loss_history)
    for name in ("enc1", "enc2", "dec2", "dec1"):
        r = report[name]
        print(f"{name}: initial={r['initial']:.4e} final={r['final']:.4e} "
              f"slope={r['slope']:.4e} fell={r['fell']}")
    print("dec_dominates_enc:", report["dec_dominates_enc"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["smoke", "production"], default="smoke")
    p.add_argument("--shard-dir", default=None,
                   help="production mode: path (or, once data/upload.py is wired, a URI) to "
                        "already-tokenized uint32 shards (index.json + *.bin). Produced "
                        "separately by T-0244's pipeline -- this script does not tokenize.")
    p.add_argument("--ckpt-dir", default=None,
                   help="production mode: directory (must be on /workspace, never the pod's "
                        "small / overlay) to write ckpt.pt to.")
    p.add_argument("--resume", action="store_true",
                   help="resume from <ckpt-dir>/ckpt.pt if it already exists.")
    p.add_argument("--go", action="store_true",
                   help="actually start training (production mode only) -- without this flag "
                        "the plan is printed and nothing runs.")
    args = p.parse_args()

    if args.mode == "smoke":
        run_smoke()
        return

    if not args.shard_dir or not args.ckpt_dir:
        p.error("--shard-dir and --ckpt-dir are required for --mode production")
    run_production(args.shard_dir, args.ckpt_dir, do_resume=args.resume, go=args.go)


if __name__ == "__main__":
    main()
