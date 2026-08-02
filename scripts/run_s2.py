"""Standalone CLI launcher for S2 runs -- fork of scripts/run_s1.py's
shape, for the composed-forward + trained-L_rec crux (train/s2_step.py).

`--mode production` without `--go` only prints the run plan and exits --
same default-safe behavior as run_s1.py / notebooks/S1_train.ipynb. This
script does not tokenize data itself and does NOT pick a new shard_dir by
default -- S2 REUSES the S1 shards verbatim (build plan P3: no-op), so
`configs.s2.S2Config.shard_dir` already defaults to
`/workspace/data/s1_shards`.

AUTHORED, NOT RUN (paid-compute boundary) -- same discipline as
scripts/run_s1.py.

Usage:
    python scripts/run_s2.py --mode smoke
    python scripts/run_s2.py --mode production \\
        --shard-dir /workspace/data/s1_shards \\
        --ckpt-dir  /workspace/checkpoints/s2 \\
        --init-from /workspace/checkpoints/s1/ckpt.pt \\
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
    """Delegates to scripts/smoke_s2.py in-process (not a subprocess) so a
    failure surfaces as a normal traceback with a non-zero exit code."""
    runpy.run_path(os.path.join(REPO_ROOT, "scripts", "smoke_s2.py"), run_name="__main__")


def run_production(shard_dir: str, ckpt_dir: str, init_from: str | None,
                    do_resume: bool, go: bool) -> None:
    from configs.s2 import S2Config
    from harness.g1_metrics import g1a_report
    from train.s2_step import S2_LOSS_KEYS, build_state, checkpoint, train_step
    from train.s2_step import resume as resume_state

    cfg = S2Config()
    cfg.shard_dir = shard_dir
    cfg.ckpt_path = os.path.join(ckpt_dir, "ckpt.pt")
    if init_from is not None:
        cfg.init_from = init_from

    if not os.path.exists(os.path.join(shard_dir, "index.json")):
        print(f"REFUSING: no index.json under {shard_dir!r} -- S2 reuses the S1 shards "
              f"verbatim (build plan P3), this script does not tokenize data itself.")
        sys.exit(1)
    # The actual fresh-vs-resume branch below takes the FRESH (build_state,
    # init_from-seeded) path whenever `not (do_resume and ckpt_path exists)`
    # -- this refuse-check must use the SAME condition, not just `not
    # do_resume`. scripts/autoresume_s2.sh always passes --resume (even on
    # its very first attempt, before any ckpt.pt exists), so gating this on
    # `not do_resume` alone would silently skip the check for the one
    # caller that matters, letting a bad --init-from surface as a raw
    # traceback deep in build_state instead of this clean refusal (and the
    # autoresume wrapper would then retry that blindly for up to 25 attempts).
    about_to_take_fresh_branch = not (do_resume and os.path.exists(cfg.ckpt_path))
    if cfg.init_from and about_to_take_fresh_branch and not os.path.exists(cfg.init_from):
        print(f"REFUSING: --init-from {cfg.init_from!r} does not exist -- S2 needs the S1 "
              f"checkpoint to warm-start from (pass --init-from explicitly, or --resume if "
              f"continuing an S2 run already in progress).")
        sys.exit(1)

    steps_total = cfg.total_tokens // (cfg.micro_batch_size * cfg.seq_len * cfg.grad_accum_steps)
    print("=== S2 PRODUCTION RUN PLAN ===")
    print(f"  shard_dir:        {shard_dir}  (REUSED from S1, plan P3)")
    print(f"  ckpt_path:        {cfg.ckpt_path}")
    print(f"  init_from:        {cfg.init_from}")
    print(f"  total_tokens:     {cfg.total_tokens:,}")
    print(f"  planned steps:    {steps_total}")
    print(f"  micro_batch_size: {cfg.micro_batch_size}  grad_accum_steps: {cfg.grad_accum_steps}")
    print(f"  seq_len:          {cfg.seq_len}")
    print(f"  lr:               {cfg.lr}")
    print(f"  alpha_A/alpha_B:  {cfg.alpha_A} / {cfg.alpha_B}")
    print(f"  use_8bit_adam:    {cfg.use_8bit_adam}")
    print(f"  resume:           {do_resume}")

    if not go:
        print("\n--go not passed -- plan printed, NOT starting training (same "
              "default-safe behavior as scripts/run_s1.py's --go gate).")
        return

    os.makedirs(ckpt_dir, exist_ok=True)
    if do_resume and os.path.exists(cfg.ckpt_path):
        state, opt_name = resume_state(cfg, cfg.ckpt_path, device="cuda")
        # steps_done comes from the checkpoint's metadata step counter now
        # (train/s2_step.checkpoint stores it; resume restores it) -- the
        # dataloader-position derivation remains as a fallback for
        # pre-metadata checkpoints (metadata step falls back to 0).
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
    for name in S2_LOSS_KEYS:
        r = report[name]
        print(f"{name}: initial={r['initial']:.4e} final={r['final']:.4e} "
              f"slope={r['slope']:.4e} fell={r['fell']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["smoke", "production"], default="smoke")
    p.add_argument("--shard-dir", default=None,
                   help="production mode: path to already-tokenized uint32 shards "
                        "(index.json + *.bin) -- REUSES the S1 shards (plan P3), pass "
                        "/workspace/data/s1_shards.")
    p.add_argument("--ckpt-dir", default=None,
                   help="production mode: directory (must be on /workspace) to write "
                        "the S2 ckpt.pt to -- distinct from S1's own ckpt-dir.")
    p.add_argument("--init-from", default=None,
                   help="production mode: path to the S1 final checkpoint to warm-start "
                        "STUDENT WEIGHTS ONLY from (configs.s2.S2Config.init_from's default "
                        "is checkpoints/s1_final_ckpt.pt -- pass this to override, e.g. "
                        "/workspace/checkpoints/s1/ckpt.pt on the pod).")
    p.add_argument("--resume", action="store_true",
                   help="resume from <ckpt-dir>/ckpt.pt if it already exists (restores the "
                        "FULL S2 trainer state -- distinct from --init-from's weights-only "
                        "S1 warm start, which only applies on a fresh run).")
    p.add_argument("--go", action="store_true",
                   help="actually start training (production mode only) -- without this flag "
                        "the plan is printed and nothing runs.")
    args = p.parse_args()

    if args.mode == "smoke":
        run_smoke()
        return

    if not args.shard_dir or not args.ckpt_dir:
        p.error("--shard-dir and --ckpt-dir are required for --mode production")
    run_production(args.shard_dir, args.ckpt_dir, init_from=args.init_from,
                    do_resume=args.resume, go=args.go)


if __name__ == "__main__":
    main()
