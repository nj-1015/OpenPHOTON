#!/usr/bin/env bash
# Auto-resume wrapper for the S1 production run on a flaky pod (the A40
# container cgroup-OOM'd once at ~3.75h). Relaunch with --resume on any
# non-completion exit; break when run_s1 prints its completion marker.
# Correct total-token accounting is handled INSIDE run_s1 (steps_done is
# derived from the restored dataloader position), so repeated resumes
# converge to exactly the intended 500M-token budget -- not a fresh epoch
# each time. Logs are APPENDED so the full history (incl. pre-death steps)
# is preserved.
set -u
cd /workspace/photon-qwen3
export HF_HOME=/workspace/hf_cache
export PYTHONUNBUFFERED=1
LOG=/workspace/s1_train.log

for attempt in $(seq 1 25); do
  echo "[wrapper $(date '+%F %T')] attempt $attempt starting" >> "$LOG"
  python -u -m scripts.run_s1 --mode production \
      --shard-dir /workspace/data/s1_shards \
      --ckpt-dir  /workspace/checkpoints/s1 \
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
