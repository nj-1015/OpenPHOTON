"""Online, selective-hook teacher hidden-state capture for S1 (plan P2).

Adapted from harness/capture_reference.py's capture discipline (same model
id) but:
  - ONLINE: the teacher stays loaded for the whole training run (it's a
    live nn.Module the trainer calls every step), it doesn't dump to an
    .npz and reload.
  - SELECTIVE: forward hooks on exactly the 4 transferred-layer boundaries
    {6, 13, 20, 27} (zero-indexed -- their OUTPUTS are Qwen3's
    hs[7], hs[14], hs[21], hs[28]) plus the embedding, NOT
    `output_hidden_states=True`. Capturing and retaining all 29 hidden
    states every step (as capture_reference.py does, fine for a one-shot
    .npz dump) would multiply activation memory for nothing here -- S1
    only ever needs 5 of them, and the teacher is frozen (no backward
    through it, so nothing downstream of the last hook needs to be kept
    around either).

Hook pattern: register_forward_hook on each target layer, detach the
captured tensor inside the callback so nothing keeps the autograd graph
alive past the hook.

**H28 pre-norm convention (locked here):** H28 is
`layers[27]`'s raw forward-hook output -- PRE-norm. This is NOT the same
tensor as `harness/capture_reference.py`'s (and HF's own
`output_hidden_states=True`) `hidden_states[28]`, which is POST-norm
(`model.model.norm` applied once, after the layer loop -- see
`transformers.models.qwen3.modeling_qwen3.Qwen3Model.forward`). Measured
on a real prompt: `(H28 - hs[28]).abs().max()` is O(1e3) (~1.45e3-2.1e3
depending on prompt), while `(model.model.norm(H28) - hs[28]).abs().max()
== 0.0` exactly. This is the RIGHT convention for S1/S2: `photon/model.py`
applies `norm` exactly once, at the final `lm_head` step
(`logits = self.lm_head(self.norm(h0_hat))`), so dec1's own output
(`h0_hat`) is pre-norm too -- H28 must stay pre-norm or S1/S2 would be
training dec1 to match the wrong (normed) tensor. `_verify_h28_is_pre_norm`
below pins this with a real, cheap, one-time runtime check (not just a
comment) so a future transformers version or hook refactor that silently
starts capturing post-norm can't pass unnoticed.
"""
import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-0.6B"

# zero-indexed decoder-layer indices whose OUTPUT equals Qwen3's
# hs[7]/hs[14]/hs[21]/hs[28] (hs[0] is the embedding output; hs[i] for
# i >= 1 is the output of layer i-1) -- matches PhotonConfig's default
# split=(7,7,7,7) stage boundaries exactly. H28 specifically is PRE-norm
# (see module docstring's "H28 pre-norm convention").
CAPTURE_LAYER_IDXS = {"H7": 6, "H14": 13, "H21": 20, "H28": 27}
CAPTURE_KEYS = ("H0", "H7", "H14", "H21", "H28")


class TeacherHiddenExtractor:
    """Frozen Qwen3-0.6B with forward hooks at the embedding + 4 layer
    boundaries. `.capture(input_ids)` runs one `no_grad` forward and
    returns `{H0, H7, H14, H21, H28}`, each `[B, T, d]`, detached, in the
    model's loaded dtype (bf16 for the GPU trainer, matching the student).
    H28 is PRE-norm -- see module docstring."""

    def __init__(self, model_id: str = MODEL_ID, dtype: torch.dtype = torch.bfloat16,
                 device: str | torch.device = "cuda", attn_implementation: str | None = None,
                 verify_h28_convention: bool = True):
        kwargs = dict(dtype=dtype)
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._captured: dict[str, torch.Tensor] = {}
        self._hooks = []
        self._register_hooks()

        if verify_h28_convention:
            self._verify_h28_is_pre_norm()

    def _verify_h28_is_pre_norm(self) -> None:
        """One-time, cheap (T=4, B=1) runtime lock on the H28 pre-norm
        convention documented in this module's docstring: asserts
        `norm(H28) == hs[-1]` (the relationship DOES hold) and
        `H28 != hs[-1]` (H28 itself is genuinely pre-norm, not already
        normed) against a fresh `output_hidden_states=True` reference
        forward on the same tiny probe input. Catches a future
        transformers version / hook refactor that silently changes what
        the layers[27] forward hook sees, before it corrupts a real
        training run."""
        device = next(self.model.parameters()).device
        probe_ids = torch.zeros(1, 4, dtype=torch.long, device=device)
        h28 = self.capture(probe_ids)["H28"]
        with torch.no_grad():
            ref = self.model(input_ids=probe_ids, output_hidden_states=True, use_cache=False)
        ref_hs28 = ref.hidden_states[-1].to(h28.dtype)
        normed_h28 = self.model.model.norm(h28)

        assert torch.allclose(normed_h28, ref_hs28, atol=1e-2, rtol=1e-2), (
            "H28 pre-norm convention check FAILED: norm(H28) no longer matches "
            "output_hidden_states=True's hs[-1]. S1/S2 rely on H28 being "
            "exactly layers[27]'s pre-norm output (see harness/teacher.py's "
            "module docstring) -- do not proceed without re-deriving the "
            "correspondence."
        )
        assert not torch.allclose(h28, ref_hs28, atol=1.0), (
            "H28 unexpectedly already matches the POST-norm hs[-1] -- the "
            "layers[27] forward hook may now be firing after norm, which "
            "would silently break the pre-norm convention S1/S2 rely on "
            "(see harness/teacher.py's module docstring)."
        )
        self._captured.clear()  # the probe capture must not leak into a real .capture() call

    def _register_hooks(self) -> None:
        def make_hook(name):
            def hook(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                self._captured[name] = h.detach()
            return hook

        layers = self.model.model.layers
        self._hooks.append(self.model.model.embed_tokens.register_forward_hook(make_hook("H0")))
        for name, idx in CAPTURE_LAYER_IDXS.items():
            self._hooks.append(layers[idx].register_forward_hook(make_hook(name)))

    @torch.no_grad()
    def capture(self, input_ids: torch.LongTensor) -> dict[str, torch.Tensor]:
        self._captured.clear()
        self.model(input_ids=input_ids, use_cache=False)
        missing = set(CAPTURE_KEYS) - set(self._captured)
        assert not missing, f"teacher hooks did not fire for {missing} -- layer-index drift?"
        return dict(self._captured)

    @torch.no_grad()
    def logits(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Frozen-teacher next-token logits `[B, T, V]` for S3 KD, detached,
        in the model's loaded dtype. Unlike `capture()` (which S1/S2 use for
        the teacher's HIDDEN states), S3 distils the teacher's OUTPUT
        DISTRIBUTION, so it needs the LM-head logits, not the layer hooks.
        The registered forward hooks still fire and repopulate
        `self._captured` here (harmless -- S3 ignores it), so this shares
        the single frozen teacher instance rather than loading a second
        copy of the model just for logits."""
        out = self.model(input_ids=input_ids, use_cache=False)
        return out.logits.detach()

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
