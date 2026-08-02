"""The M5 / S0 golden-test gate.

Runs PhotonQwen3ForCausalLM in bypass mode (C1=C2=R1=R2=1) on the fixed
cases captured by capture_reference.py (short prompt, long prompt, and a
B=2 batch of both truncated to a common length -- see that script's
docstring for why the second/third cases were added), and checks it
reproduces stock Qwen3-0.6B to floating-point tolerance: max_abs <~ 1e-4,
100% argmax agreement (fp32/CPU leg -- the bf16/L4 leg is out of scope for
this run).

Per-stage hidden comparison (embed / after enc1 / after enc2 / after dec2 /
after dec1, post-norm) localises a failure to one of the four stages, per
the S0 plan's "per-stage hidden log" verification strategy -- lifted in
*method* from parity_check.py's per-layer max-abs/MSE table and
extended_parity's argmax-agreement / first-divergence scoring, adapted from
a per-generation-step comparison to a per-sequence-position one (this
harness runs one fixed forward pass, not autoregressive generation).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from photon.config import PhotonConfig
from photon.surgery import load_photon_qwen3

REF = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "_ref", "qwen3_0.6b_ref.npz"
)
CASES = ["p0", "p1", "p2", "batch"]


def diff_report(name: str, a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a - b)
    max_abs = float(d.max())
    mse = float((d ** 2).mean())
    print(f"    {name:<28}max_abs={max_abs:>12.3e}  mse={mse:>12.3e}")
    return dict(name=name, max_abs=max_abs, mse=mse)


def run_case(photon, ref, case: str) -> dict:
    input_ids = torch.tensor(ref[f"{case}_input_ids"], dtype=torch.long)
    with torch.no_grad():
        logits, mid = photon(input_ids, return_intermediates=True)
        h0_hat_normed = photon.norm(mid["h0_hat"])

    logits = logits.float().numpy()
    ref_logits = ref[f"{case}_logits"]

    print(f"\n=== case '{case}'  (B={input_ids.shape[0]}, T={input_ids.shape[1]}) ===")
    print("  per-stage hidden comparison:")
    stage_reports = [
        diff_report("embed (h0 vs hs_0)", mid["h0"].numpy(), ref[f"{case}_hs_0"]),
        diff_report("after enc1 (hs_7)", mid["h1"].numpy(), ref[f"{case}_hs_7"]),
        diff_report("after enc2 (hs_14)", mid["h2"].numpy(), ref[f"{case}_hs_14"]),
        diff_report("after dec2 (hs_21)", mid["h1_hat"].numpy(), ref[f"{case}_hs_21"]),
        diff_report("after dec1, post-norm (hs_28)", h0_hat_normed.numpy(), ref[f"{case}_hs_28"]),
    ]

    print("  logits:")
    logit_report = diff_report("logits", logits, ref_logits)

    argmax_agree = (logits.argmax(-1) == ref_logits.argmax(-1))
    agree_frac = float(argmax_agree.mean())
    print(f"  argmax agreement (all positions, all batch rows): {agree_frac * 100:.1f}%")

    first_div = None
    flat = argmax_agree.reshape(-1)
    for i, ok in enumerate(flat):
        if not ok:
            first_div = i
            break
    print(f"  first argmax divergence: {'none' if first_div is None else f'flat index {first_div}'}")

    gate_max_abs = logit_report["max_abs"] < 1e-4
    gate_argmax = agree_frac == 1.0
    print(f"  GATE: max_abs<1e-4 = {gate_max_abs}   100% argmax = {gate_argmax}")

    return dict(case=case, stage_reports=stage_reports, logit_report=logit_report,
                agree_frac=agree_frac, first_div=first_div,
                green=gate_max_abs and gate_argmax)


def main():
    ref = np.load(REF)

    cfg = PhotonConfig(C1=1, C2=1, R1=1, R2=1, bypass=True)
    photon, stock = load_photon_qwen3(cfg)
    del stock  # photon holds shared references to all its parameters

    results = [run_case(photon, ref, case) for case in CASES]

    print("\n=== SUMMARY ===")
    all_green = True
    for r in results:
        print(f"  {r['case']:<8} {'GREEN' if r['green'] else 'RED'}")
        all_green = all_green and r["green"]
    print("\nGATE RESULT:", "GREEN" if all_green else "RED")
    return results


if __name__ == "__main__":
    main()
