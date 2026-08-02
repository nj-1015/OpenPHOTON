#!/usr/bin/env bash
# rec_weight sweep orchestrator. Waits for the running S3 trial to finish
# (so the GPU is free -- no contention), snapshots its final checkpoint as a
# common warm-start, then trains each rec_weight arm sequentially for a short
# budget. Each arm continues from the SAME init and differs ONLY in rec_weight
# (the coefficient on the RecGen-consistency term rec_B) -- so the comparison
# isolates rec_weight's effect on RecGen fidelity vs prediction quality.
#
# Checkpoints go to /root/* (container overlay, 20GB, reliable) NOT /workspace
# (its ~19GB quota is nearly full and silently truncates 3.6GB writes). Each
# arm's ckpt is downloaded + evaluated locally as it completes.
set -u
cd /workspace/photon-qwen3
export HF_HOME=/workspace/hf_cache
export PYTHONUNBUFFERED=1
LOG=/workspace/sweep.log
TRIALLOG=/workspace/s3_train.log
ARMS="0.3 1.0 3.0"
TOKENS=50000000                       # 50M tokens/arm (~6100 steps, ~3h on the A40)

echo "[sweep $(date '+%F %T')] waiting for the S3 trial completion marker..." >> "$LOG"
while ! grep -q 'done, final checkpoint saved' "$TRIALLOG"; do sleep 120; done
echo "[sweep $(date '+%F %T')] trial complete; freeing GPU (30s) then snapshotting init" >> "$LOG"
sleep 30
cp /workspace/checkpoints/s3/ckpt.pt /root/sweep_init.pt
echo "[sweep $(date '+%F %T')] init snapshot: $(stat -c %s /root/sweep_init.pt) bytes" >> "$LOG"

for RW in $ARMS; do
  DIR=/root/sweep_rw${RW}
  CK="$DIR/ckpt.pt"
  echo "[sweep $(date '+%F %T')] === ARM rec_weight=$RW -> $DIR (tokens=$TOKENS) ===" >> "$LOG"
  for attempt in 1 2 3; do
    python -u -m scripts.run_s3 --mode production \
        --shard-dir  /workspace/data/s1_shards \
        --ckpt-dir   "$DIR" \
        --init-from  /root/sweep_init.pt \
        --rec-weight "$RW" --total-tokens "$TOKENS" \
        --micro-batch 4 --resume --go >> "$LOG" 2>&1
    if grep -q "final checkpoint saved to $CK" "$LOG"; then break; fi
    echo "[sweep $(date '+%F %T')] arm $RW attempt $attempt exited without completion; retry in 20s" >> "$LOG"
    sleep 20
  done
  echo "[sweep $(date '+%F %T')] ARM rec_weight=$RW DONE ($CK)" >> "$LOG"
done
echo "[sweep $(date '+%F %T')] ALL SWEEP ARMS COMPLETE" >> "$LOG"
