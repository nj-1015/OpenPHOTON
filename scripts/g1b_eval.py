"""G1-b (plan slide 37 / structural-capability gate): run the student's OWN
chained forward (teacher-forcing OFF) on a real batch and measure whether
dec2's reconstruction tracks the encoder streams -- cos(h1_hat, h1) and
cos(chunk2(h1_hat), h2). Compares the TRAINED S1 checkpoint vs an UNTRAINED
(fresh-surgery) baseline: G1-b passes if the trained cosines are high AND
improved over untrained -> S2's L_rec has a fixed point to converge to.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from configs.s1 import S1Config
from photon.surgery import load_photon_qwen3
from checkpoint import ckpt
from data.stream_dataset import ShardStreamLoader
from harness.g1_metrics import g1b_student_running_probe

DEV = "cuda"
CKPT = "/workspace/checkpoints/s1/ckpt.pt"
SHARDS = "/workspace/data/s1_shards"

cfg = S1Config()
# a real batch from the training shards (same distribution S1 saw)
ids = ShardStreamLoader(SHARDS, batch_size=8).next_batch().to(DEV)
print("batch:", tuple(ids.shape))

# UNTRAINED baseline (fresh surgery, identity-init modules)
u, _ = load_photon_qwen3(cfg.photon); u.to(DEV)
base = g1b_student_running_probe(u, ids)
print("UNTRAINED:", {k: round(v, 4) for k, v in base.items()})
del u; torch.cuda.empty_cache()

# TRAINED (final S1 checkpoint)
t, _ = load_photon_qwen3(cfg.photon); t.to(DEV)
dummy_opt = torch.optim.AdamW([p for p in t.parameters() if p.requires_grad], lr=1e-4)
ckpt.load(CKPT, t, dummy_opt)
trained = g1b_student_running_probe(t, ids)
print("TRAINED:  ", {k: round(v, 4) for k, v in trained.items()})

print("--- DELTA (trained - untrained) ---")
for k in trained:
    print(f"  {k}: {trained[k]-base[k]:+.4f}   (untrained {base[k]:.4f} -> trained {trained[k]:.4f})")
