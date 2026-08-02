"""Standalone CLI launcher for S3 runs -- fork of scripts/run_s2.py's
shape, for the logit/KL distillation crux (train/s3_step.py).

`--mode production` without `--go` only prints the run plan and exits.
S3 REUSES the S1/S2 shards verbatim (no new tokenization) and warm-starts
STUDENT WEIGHTS from the S2 final checkpoint.

Usage:
    python scripts/run_s3.py --mode smoke
    python scripts/run_s3.py --mode production \\
        --shard-dir /workspace/data/s1_shards \\
        --ckpt-dir  /workspace/checkpoints/s3 \\
        --init-from /workspace/checkpoints/s2/ckpt.pt \\
        [--micro-batch N] [--resume] [--go]
"""
import argparse
import json
import os
import runpy
import sys
import time

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO_ROOT)


def run_smoke() -> None:
    runpy.run_path(os.path.join(REPO_ROOT, "scripts", "smoke_s3.py"), run_name="__main__")


def run_production(shard_dir: str, ckpt_dir: str, init_from: str | None,
                    micro_batch: int | None, rec_weight: float | None,
                    total_tokens: int | None, do_resume: bool, go: bool) -> None:
    from configs.s3 import S3Config
    from harness.g1_metrics import g1a_report
    from train.s3_step import S3_LOSS_KEYS, build_state, checkpoint, train_step
    from train.s3_step import resume as resume_state

    cfg = S3Config()
    cfg.shard_dir = shard_dir
    cfg.ckpt_path = os.path.join(ckpt_dir, "ckpt.pt")
    if init_from is not None:
        cfg.init_from = init_from
    if micro_batch is not None:
        # micro_batch only affects steps_total (not the token budget) -- it is
        # exposed so a memory probe on the pod can raise it if there's headroom.
        cfg.micro_batch_size = micro_batch
    if rec_weight is not None:
        # rec_weight sweep: the coefficient on the RecGen-consistency term
        # (rec_B). Higher -> pull h1_hat back toward h1 (better RecGen inference
        # fidelity) at some prediction-quality cost. Default 0.1 was too weak.
        cfg.rec_weight = rec_weight
    if total_tokens is not None:
        # short continued-training budget for a sweep arm (e.g. 50M), overriding
        # the 500M trial default.
        cfg.total_tokens = total_tokens

    if not os.path.exists(os.path.join(shard_dir, "index.json")):
        print(f"REFUSING: no index.json under {shard_dir!r} -- S3 reuses the S1/S2 shards "
              f"verbatim, this script does not tokenize data itself.")
        sys.exit(1)
    # Enforce the seq_len <-> shard-row-length invariant the token budget
    # depends on (reviewer defense-in-depth: this is EXACTLY where the S1
    # accounting bug lived). steps_total is computed from cfg.seq_len, but the
    # loader yields shard-row-length sequences and ignores cfg.seq_len -- so a
    # mismatch silently mis-budgets total_tokens. The index.json carries a
    # per-shard seq_len; refuse loudly on any disagreement rather than train
    # the wrong number of tokens.
    with open(os.path.join(shard_dir, "index.json")) as _f:
        _shard_seqs = {s["seq_len"] for s in json.load(_f).get("shards", []) if "seq_len" in s}
    if _shard_seqs and _shard_seqs != {cfg.seq_len}:
        print(f"REFUSING: shard seq_len(s) {_shard_seqs} != cfg.seq_len {cfg.seq_len} -- the "
              f"token budget (steps_total) is computed from cfg.seq_len but the loader yields "
              f"shard-row-length sequences; a mismatch silently mis-budgets total_tokens.")
        sys.exit(1)
    # Same fresh-vs-resume gating as run_s2.py: the autoresume wrapper always
    # passes --resume (even on the first attempt, before ckpt.pt exists), so
    # the init_from existence check must use the SAME condition the fresh
    # branch below uses, not `not do_resume` alone.
    about_to_take_fresh_branch = not (do_resume and os.path.exists(cfg.ckpt_path))
    if cfg.init_from and about_to_take_fresh_branch and not os.path.exists(cfg.init_from):
        print(f"REFUSING: --init-from {cfg.init_from!r} does not exist -- S3 needs the S2 "
              f"checkpoint to warm-start from (pass --init-from explicitly, or --resume if "
              f"continuing an S3 run already in progress).")
        sys.exit(1)

    steps_total = cfg.total_tokens // (cfg.micro_batch_size * cfg.seq_len * cfg.grad_accum_steps)
    print("=== S3 PRODUCTION RUN PLAN (logit/KL distillation) ===")
    print(f"  shard_dir:        {shard_dir}  (REUSED from S1/S2)")
    print(f"  ckpt_path:        {cfg.ckpt_path}")
    print(f"  init_from:        {cfg.init_from}")
    print(f"  total_tokens:     {cfg.total_tokens:,}")
    print(f"  planned steps:    {steps_total}")
    print(f"  micro_batch_size: {cfg.micro_batch_size}  grad_accum_steps: {cfg.grad_accum_steps}")
    print(f"  seq_len:          {cfg.seq_len}")
    print(f"  lr:               {cfg.lr}")
    print(f"  kd_temperature:   {cfg.kd_temperature}")
    print(f"  ce_weight:        {cfg.ce_weight}")
    print(f"  rec_weight:       {cfg.rec_weight}")
    print(f"  use_8bit_adam:    {cfg.use_8bit_adam}")
    print(f"  resume:           {do_resume}")

    if not go:
        print("\n--go not passed -- plan printed, NOT starting training.")
        return

    os.makedirs(ckpt_dir, exist_ok=True)
    if do_resume and os.path.exists(cfg.ckpt_path):
        state, opt_name = resume_state(cfg, cfg.ckpt_path, device="cuda")
        # steps_done from the checkpoint's metadata step counter
        # (train/s3_step.checkpoint stores it; resume restores it) -- the
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
        if step % 100 == 0:
            print(f"step {step}/{steps_total}", losses)
        if time.time() - last_ckpt_time > cfg.ckpt_every_seconds:
            try:
                checkpoint(state, cfg.ckpt_path)
                print(f"checkpointed at step {step} -> {cfg.ckpt_path}")
            except Exception as e:
                # Transient MFS network-volume write faults (observed on this
                # pod: 'PytorchStreamWriter failed writing ... file write
                # failed') must NOT kill a multi-hour run. ckpt.save writes
                # atomically (tempfile + os.replace), so a failed write leaves
                # the LAST good checkpoint intact -- log and retry at the next
                # interval instead of crash-resuming and losing steps.
                print(f"WARNING: checkpoint at step {step} failed "
                      f"({type(e).__name__}: {e}); keeping last good ckpt, retry next interval")
            last_ckpt_time = time.time()   # reset either way (don't hammer a struggling volume)

    # Final checkpoint is the deliverable -- retry through transient write
    # faults; the completion marker the autoresume wrapper greps for is only
    # printed once the save actually lands (a total failure leaves no marker,
    # so the wrapper resumes and re-attempts the final save).
    for _fa in range(6):
        try:
            checkpoint(state, cfg.ckpt_path)
            print("done, final checkpoint saved to", cfg.ckpt_path)
            break
        except Exception as e:
            print(f"final checkpoint attempt {_fa} failed ({type(e).__name__}: {e}); retry in 30s")
            time.sleep(30)
    else:
        print("final checkpoint FAILED after retries -- wrapper will resume and re-attempt")
        return

    report = g1a_report(state.loss_history)
    for name in S3_LOSS_KEYS:
        r = report[name]
        print(f"{name}: initial={r['initial']:.4e} final={r['final']:.4e} "
              f"slope={r['slope']:.4e} fell={r['fell']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["smoke", "production"], default="smoke")
    p.add_argument("--shard-dir", default=None,
                   help="production: path to the already-tokenized uint32 shards "
                        "(index.json + *.bin) -- REUSES the S1/S2 shards, pass "
                        "/workspace/data/s1_shards.")
    p.add_argument("--ckpt-dir", default=None,
                   help="production: directory (on /workspace) to write the S3 ckpt.pt to.")
    p.add_argument("--init-from", default=None,
                   help="production: path to the S2 final checkpoint to warm-start STUDENT "
                        "WEIGHTS ONLY from (e.g. /workspace/checkpoints/s2/ckpt.pt on the pod).")
    p.add_argument("--micro-batch", type=int, default=None,
                   help="production: override S3Config.micro_batch_size (memory-probe knob -- "
                        "affects steps_total, NOT the token budget).")
    p.add_argument("--rec-weight", type=float, default=None,
                   help="production: override S3Config.rec_weight (RecGen-consistency term; "
                        "the rec_weight sweep varies this).")
    p.add_argument("--total-tokens", type=int, default=None,
                   help="production: override S3Config.total_tokens (e.g. 50000000 for a short "
                        "sweep arm; default is the 500M trial budget).")
    p.add_argument("--resume", action="store_true",
                   help="resume from <ckpt-dir>/ckpt.pt if it exists (restores the FULL S3 "
                        "trainer state -- distinct from --init-from's weights-only S2 warm start).")
    p.add_argument("--go", action="store_true",
                   help="actually start training (production only) -- without it the plan prints "
                        "and nothing runs.")
    args = p.parse_args()

    if args.mode == "smoke":
        run_smoke()
        return

    if not args.shard_dir or not args.ckpt_dir:
        p.error("--shard-dir and --ckpt-dir are required for --mode production")
    run_production(args.shard_dir, args.ckpt_dir, init_from=args.init_from,
                    micro_batch=args.micro_batch, rec_weight=args.rec_weight,
                    total_tokens=args.total_tokens, do_resume=args.resume, go=args.go)


if __name__ == "__main__":
    main()
