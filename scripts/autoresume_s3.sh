#!/usr/bin/env bash
# Auto-resume wrapper for the S3 production run (logit/KL distillation) --
# same relaunch-on-any-non-completion-exit discipline as
# scripts/autoresume_s2.sh (the A40 container cgroup-OOM'd once during S1;
# KD adds two full-vocab logit tensors so an OOM is if anything more
# plausible here -- hence the smaller default micro_batch and the smoke's
# memory probe). Relaunch with --resume on any non-completion exit; break
# when run_s3 prints its completion marker. Correct total-token accounting
# is handled INSIDE run_s3 (steps_done derived from the restored dataloader
# position), so repeated resumes converge to exactly the 500M-token budget,
# not a fresh epoch each time. Logs are APPENDED so the full history is kept.
#
# NB: --init-from is only consulted by run_s3.py on a FRESH run (no existing
# ckpt.pt at --ckpt-dir yet); once ckpt.pt exists, --resume restores the full
# S3 state and --init-from is ignored (harmless to keep passing it).
#
# --micro-batch defaults to S3Config's 4 (KD memory ceiling). Override the
# MB= env var below only after a production-size memory probe shows headroom.
set -u
cd /workspace/photon-qwen3
export HF_HOME=/workspace/hf_cache
export PYTHONUNBUFFERED=1
LOG=/workspace/s3_train.log
MB="${MB:-4}"

for attempt in $(seq 1 25); do
  echo "[wrapper $(date '+%F %T')] attempt $attempt starting (micro_batch=$MB)" >> "$LOG"
  python -u -m scripts.run_s3 --mode production \
      --shard-dir  /workspace/data/s1_shards \
      --ckpt-dir   /workspace/checkpoints/s3 \
      --init-from  /workspace/checkpoints/s2/ckpt.pt \
      --micro-batch "$MB" \
      --resume --go >> "$LOG" 2>&1
  ec=$?
  if grep -q 'done, final checkpoint saved' "$LOG"; then
    echo "[wrapper $(date '+%F %T')] COMPLETE (exit $ec)" >> "$LOG"
    break
  fi
  echo "[wrapper $(date '+%F %T')] run exited (code $ec) without completion; resuming in 20s" >> "$LOG"
  sleep 20
done
echo "[wrapper $(date '+%F %T')] wrapper exiting" >> "$LOG"
