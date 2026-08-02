#!/usr/bin/env bash
# Auto-resume wrapper for the S2 production run -- fork of
# scripts/autoresume_s1.sh's discipline (the A40 container cgroup-OOM'd
# once at ~3.75h during S1; same relaunch-on-any-non-completion-exit
# strategy applies here). Relaunch with --resume on any non-completion
# exit; break when run_s2 prints its completion marker. Correct
# total-token accounting is handled INSIDE run_s2 (steps_done is derived
# from the restored dataloader position, same as run_s1), so repeated
# resumes converge to exactly the intended 500M-token budget -- not a
# fresh epoch each time. Logs are APPENDED so the full history (incl.
# pre-death steps) is preserved.
#
# AUTHORED, NOT RUN (paid-compute boundary) -- same discipline as
# scripts/autoresume_s1.sh.
#
# NB: --init-from is only consulted by run_s2.py on a FRESH run (no
# existing ckpt.pt at --ckpt-dir yet); once ckpt.pt exists, --resume takes
# over and restores the full S2 state (student weights already
# S1-warm-started + since-trained, optimizer state, dataloader position)
# -- passing --init-from on every relaunch is harmless (run_s2.py ignores
# it once --resume finds an existing checkpoint).
set -u
cd /workspace/photon-qwen3
export HF_HOME=/workspace/hf_cache
export PYTHONUNBUFFERED=1
LOG=/workspace/s2_train.log

for attempt in $(seq 1 25); do
  echo "[wrapper $(date '+%F %T')] attempt $attempt starting" >> "$LOG"
  python -u -m scripts.run_s2 --mode production \
      --shard-dir  /workspace/data/s1_shards \
      --ckpt-dir   /workspace/checkpoints/s2 \
      --init-from  /workspace/checkpoints/s1/ckpt.pt \
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
