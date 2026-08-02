#!/usr/bin/env bash
# Environment setup for a cloud GPU pod (e.g. an A40 instance) running
# PHOTON-Qwen3 S1.
#
# Assumes an existing pod with a WORKING torch+cu12x already installed and
# verified with a real CUDA matmul (not just torch.cuda.is_available() --
# some driver/toolkit mismatches report is_available() == True while every
# kernel launch still fails). This script DOES NOT reinstall torch -- pods
# commonly ship a pinned, already-working torch build; replacing it "to get
# a newer one" is a common way to break a working GPU environment.
#
# /workspace is assumed to be the pod's large, persistent network volume
# (survives restarts/most migrations); / is a small overlay. HF cache,
# checkpoints, and data shards all go on /workspace -- never on /.
#
# Usage (on the pod):
#   bash scripts/setup_pod_a40.sh
set -euo pipefail

HF_CACHE_DIR="${HF_HOME:-/workspace/hf_cache}"
export HF_HOME="$HF_CACHE_DIR"
mkdir -p "$HF_CACHE_DIR"

if ! grep -qs "export HF_HOME=$HF_CACHE_DIR" /root/.bashrc 2>/dev/null; then
    echo "export HF_HOME=$HF_CACHE_DIR" >> /root/.bashrc
fi

echo "=== torch (NOT reinstalled -- must already work) ==="
python3 -c "
import torch
assert torch.cuda.is_available(), 'torch.cuda.is_available() is False -- fix the base image before running this script'
x = torch.randn(8, 8, device='cuda') @ torch.randn(8, 8, device='cuda')
x.sum().item()
print('torch', torch.__version__, '-- CUDA matmul OK on', torch.cuda.get_device_name(0))
"

echo "=== installing repo deps (torch left untouched) ==="
# transformers is pinned to 5.3.0 here (requirements.txt's floor is only
# >=5.3): transformers 5.14.1 (the latest at the time this was written)
# imports torch.distributed.tensor.DTensor at module load time
# (transformers/distributed/sharding_utils.py), which does not exist in
# this pod's torch 2.4.1+cu124 -- confirmed via a real ImportError
# ("cannot import name 'DTensor' from 'torch.distributed.tensor'").
# 5.3.0 (requirements.txt's own floor) is confirmed clean; the exact
# version where the break was introduced between 5.3.0 and 5.14.1 was
# NOT bisected. If a future pod ships a newer torch, re-check whether
# this pin can be relaxed by importing transformers.models.qwen3.modeling_qwen3
# first (the sanity check right below this block).
pip install --no-cache-dir \
    "transformers==5.3.0" \
    bitsandbytes \
    datasets \
    huggingface_hub \
    numpy \
    pytest

echo "=== sanity: torch still working after pip install (deps above are unpinned aside from transformers -- a transitive bump, e.g. numpy, could in principle disturb it) ==="
python3 -c "
import torch
assert torch.cuda.is_available()
(torch.randn(8, 8, device='cuda') @ torch.randn(8, 8, device='cuda')).sum().item()
print('torch', torch.__version__, '-- still OK on', torch.cuda.get_device_name(0))
"

echo "=== sanity: transformers imports cleanly against this torch ==="
python3 -c "from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm; print('transformers import OK')"

echo "=== sanity: bitsandbytes Adam8bit live kernel check (import can succeed even when the CUDA kernel doesn't) ==="
python3 -c "
import torch, bitsandbytes as bnb
print('bitsandbytes', bnb.__version__)
lin = torch.nn.Linear(8, 8).to('cuda', dtype=torch.bfloat16)
opt = bnb.optim.Adam8bit(lin.parameters(), lr=1e-3)
lin(torch.randn(2, 8, device='cuda', dtype=torch.bfloat16)).sum().backward()
opt.step()
print('bitsandbytes Adam8bit .step() OK on', torch.cuda.get_device_name(0))
"

echo ""
echo "=== done. HF_HOME=$HF_HOME (persisted to /root/.bashrc for future shells) ==="
echo "Next: bash scripts/run_s1_pod.sh --mode smoke   (0 real training tokens, verifies the wiring)"
