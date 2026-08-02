#!/usr/bin/env bash
# Thin launcher wrapping scripts/run_s1.py for a cloud GPU pod (mirrors the
# same smoke-gate / production-run logic used outside a Colab notebook).
# Run scripts/setup_pod_a40.sh first.
#
# Smoke mode (0 real training tokens, safe to run anytime -- this is what
# the A40 pod validation actually ran):
#   bash scripts/run_s1_pod.sh --mode smoke
#
# Production mode (the real 0.5B-token S1 run -- needs explicit sign-off
# AND real dataset ids already tokenized into shard_dir by the
# tokenization pipeline. Printing the plan without --go is always safe and
# does not spend compute):
#   bash scripts/run_s1_pod.sh --mode production \
#       --shard-dir /workspace/data/s1_shards \
#       --ckpt-dir  /workspace/checkpoints/s1 \
#       [--resume] [--go]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# HF_HOME must point at /workspace (the pod's large persistent volume) so
# model downloads never land on the small / overlay -- setup_pod_a40.sh
# sets this in /root/.bashrc, but default it here too in case this script
# is invoked from a non-interactive shell that didn't source it.
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"

python3 scripts/run_s1.py "$@"
