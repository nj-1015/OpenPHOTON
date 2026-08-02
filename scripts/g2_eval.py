"""G2 eval: held-out composed next-token perplexity + next-token accuracy
(student vs teacher), plus reconstruction cosine
(harness.g1_metrics.g1b_student_running_probe, S2 vs S1), on a fixed
held-out paragraph.

Composed = the model's OWN chained forward (teacher-forcing OFF, the same
`photon(input_ids, return_intermediates=True)` call S2 trains against) --
NOT S1's decoupled per-block teacher-forced targets. S1 was never trained
on this composed objective directly (that's what S2 optimizes for), so a
real gap between teacher/S1/S2 PPL here is expected.

Evaluates the teacher and any checkpoints that exist on disk: the S1 row
runs if --s1-ckpt exists; the S2 row is optional and is skipped if
--s2-ckpt is not passed or the path doesn't exist, so this script also
works standalone against just an S1 checkpoint.

Usage (CPU or CUDA):
    python scripts/g2_eval.py --s1-ckpt checkpoints/s1_final_ckpt.pt \\
        [--s2-ckpt checkpoints/s2_final_ckpt.pt] [--device cuda]
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO_ROOT)

# Held-out neutral English (not a training-shard artifact) -- same probe
# text as the original throwaway eval this script commits/formalizes.
HELD_OUT_TEXT = (
    "The lighthouse had stood on the rocky headland for more than a century, its "
    "white tower a familiar landmark to the fishing boats that returned each evening. "
    "In winter the storms came hard off the open water, and the keeper would climb the "
    "narrow iron stairs to check that the great lamp still turned. Over the years the "
    "village below had grown and then shrunk again; the young people left for the cities, "
    "and the harbour that once held forty vessels now sheltered only a handful. Yet the "
    "light kept its steady rhythm, sweeping the dark bay every few seconds, a small act "
    "of faith repeated through the longest nights. When the automatic system was finally "
    "installed, the last keeper packed his few belongings and walked down the hill for the "
    "final time, pausing once to look back at the tower that had been his home and his "
    "responsibility for thirty-one years. The gulls wheeled overhead as they always had, "
    "indifferent to the change, and the sea went on as the sea does, patient and enormous "
    "and entirely without memory of the people who had watched it for so long."
)


def _held_out_ids(device: str) -> torch.LongTensor:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    ids = tok(HELD_OUT_TEXT, return_tensors="pt")["input_ids"]
    L = (ids.shape[1] // 4) * 4  # PHOTON needs seq_len divisible by C1*C2=4
    return ids[:, :L].to(device)


def ppl_and_next_tok_acc(logits: torch.Tensor, ids: torch.LongTensor) -> tuple[float, float]:
    lp = logits[:, :-1].float()
    tgt = ids[:, 1:]
    ce = F.cross_entropy(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1))
    acc = (lp.argmax(-1) == tgt).float().mean().item()
    return torch.exp(ce).item(), acc


def eval_teacher(ids: torch.LongTensor, device: str) -> dict:
    from transformers import AutoModelForCausalLM
    teacher = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.float32).to(device).eval()
    with torch.no_grad():
        ppl, acc = ppl_and_next_tok_acc(teacher(ids).logits, ids)
    del teacher
    return dict(ppl=ppl, next_tok_acc=acc)


def eval_student_ckpt(ckpt_path: str, ids: torch.LongTensor, device: str) -> dict:
    """Composed-forward PPL/next-tok-acc + g1b recon-cos for ONE checkpoint
    (works for both an S1 and an S2 checkpoint -- same PhotonConfig shape,
    same checkpoint/ckpt.py format)."""
    from checkpoint import ckpt
    from harness.g1_metrics import g1b_student_running_probe
    from photon.config import PhotonConfig
    from photon.surgery import load_photon_qwen3

    cfg = PhotonConfig(C1=2, C2=2, R1=2, R2=2, split=(7, 7, 7, 7),
                        rope_mode="reset", bypass=False, dtype="float32")
    student, _ = load_photon_qwen3(cfg)
    student = student.float().to(device).eval()
    ckpt.load(ckpt_path, student, optimizer=None)

    with torch.no_grad():
        out = student(ids)
        logits = out[0] if isinstance(out, (tuple, list)) else out
    ppl, acc = ppl_and_next_tok_acc(logits, ids)
    probe = g1b_student_running_probe(student, ids)
    del student
    return dict(ppl=ppl, next_tok_acc=acc, **probe)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--s1-ckpt", default="checkpoints/s1_final_ckpt.pt")
    p.add_argument("--s2-ckpt", default=None,
                   help="optional -- skipped if not passed or the file doesn't exist")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    ids = _held_out_ids(args.device)
    V = 151936
    print(f"eval tokens: {ids.shape[1]}  (random-chance ppl ~ {V:,})")

    rows = {}
    rows["teacher"] = eval_teacher(ids, args.device)
    print(f"TEACHER  Qwen3-0.6B : ppl={rows['teacher']['ppl']:8.2f}  "
          f"next-tok-acc={rows['teacher']['next_tok_acc']:.3f}")

    if os.path.exists(args.s1_ckpt):
        rows["S1"] = eval_student_ckpt(args.s1_ckpt, ids, args.device)
        r = rows["S1"]
        print(f"S1 (decoupled)      : ppl={r['ppl']:8.2f}  next-tok-acc={r['next_tok_acc']:.3f}  "
              f"cos(h1_hat,h1)={r['cos_h1_hat_h1']:.3f}  "
              f"cos(chunk2(h1_hat),h2)={r['cos_chunk2_h1_hat_h2']:.3f}  "
              f"gap={r['ppl'] / rows['teacher']['ppl']:.1f}x")
    else:
        print(f"S1 checkpoint not found at {args.s1_ckpt!r} -- skipping S1 row")

    if args.s2_ckpt and os.path.exists(args.s2_ckpt):
        rows["S2"] = eval_student_ckpt(args.s2_ckpt, ids, args.device)
        r = rows["S2"]
        print(f"S2 (composed+L_rec) : ppl={r['ppl']:8.2f}  next-tok-acc={r['next_tok_acc']:.3f}  "
              f"cos(h1_hat,h1)={r['cos_h1_hat_h1']:.3f}  "
              f"cos(chunk2(h1_hat),h2)={r['cos_chunk2_h1_hat_h2']:.3f}  "
              f"gap={r['ppl'] / rows['teacher']['ppl']:.1f}x")
        if "S1" in rows:
            print(f"\nS2 vs S1: ppl {rows['S1']['ppl']:.1f} -> {rows['S2']['ppl']:.1f}   "
                  f"(gap {rows['S1']['ppl'] / rows['teacher']['ppl']:.1f}x -> "
                  f"{rows['S2']['ppl'] / rows['teacher']['ppl']:.1f}x)")
    else:
        print("S2 checkpoint not passed / not found -- skipping S2 row")


if __name__ == "__main__":
    main()
