"""OpenPHOTON: a friendly top-level API around the PHOTON-Qwen3 conversion.

Everything here is a thin convenience wrapper over the real implementation
(``photon``, ``inference.recgen``, ``configs.s3``, ``checkpoint.ckpt``) --
this module does no numerics of its own, it just wires them together for
the common case:

    from openphoton import load_openphoton, generate, kv_footprint

    model = load_openphoton("OpenPHOTON/Qwen3-0.6B")   # HF repo id OR local checkpoint path
    print(generate(model, "In a small village by the sea,", max_new_tokens=40, temperature=0.7))
    print(kv_footprint(model, seq_len=2048))            # -> (57344, 3584, 16.0)
"""
import glob
import os
from dataclasses import dataclass
from typing import Optional, Union

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from checkpoint import ckpt
from configs.s3 import S3Config
from inference.recgen import kv_footprint as _recgen_kv_footprint
from inference.recgen import recgen_generate
from photon.config import PhotonConfig
from photon.model import PhotonQwen3ForCausalLM
from photon.surgery import MODEL_ID as _MODEL_ID
from photon.surgery import load_photon_qwen3

__all__ = ["load_openphoton", "generate", "kv_footprint", "OpenPhoton"]


@dataclass
class OpenPhoton:
    """A loaded PHOTON-Qwen3 model, bundled with its tokenizer and config.

    `_stock` keeps the original stock Qwen3 model alive: `model`'s
    embed_tokens/norm/lm_head/rotary_emb are shared by reference with it
    (surgery.py slices submodules out rather than copying them), so letting
    `_stock` get garbage-collected would be safe in CPython (the shared
    submodules are still referenced by `model` itself) but is kept around
    anyway to make that sharing explicit and avoid relying on it.
    """
    model: PhotonQwen3ForCausalLM
    tokenizer: PreTrainedTokenizerBase
    cfg: PhotonConfig
    _stock: Optional[torch.nn.Module] = None


def _resolve_checkpoint_path(source: str) -> str:
    """Turn `source` into a concrete local checkpoint file path.

    `source` may be:
      - a path to an existing checkpoint file -- used directly.
      - a path to an existing local directory -- `ckpt.pt` inside it if
        present, else the single `*.pt` file inside it if unambiguous.
      - anything else -- treated as a Hugging Face Hub repo id and
        downloaded via `huggingface_hub.hf_hub_download`.
    """
    if os.path.isfile(source):
        return source
    if os.path.isdir(source):
        preferred = os.path.join(source, "ckpt.pt")
        if os.path.isfile(preferred):
            return preferred
        candidates = sorted(glob.glob(os.path.join(source, "*.pt")))
        if len(candidates) == 1:
            return candidates[0]
        raise FileNotFoundError(
            f"{source!r} is a local directory but no unambiguous checkpoint "
            f"file was found inside it (looked for ckpt.pt, or a single "
            f"*.pt file) -- pass the exact checkpoint file path instead."
        )
    return _download_from_hf_hub(source)


def _download_from_hf_hub(repo_id: str, filename: str = "ckpt.pt") -> str:
    """Fetch `filename` from a Hugging Face Hub model repo, with a clear
    error if the repo/file isn't there yet (the trained-weights upload for
    this project is a pending follow-up, not a bug in this loader)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to load a checkpoint by Hugging "
            "Face repo id -- `pip install huggingface_hub`, or pass a "
            "local checkpoint file/directory path instead."
        ) from e
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as e:
        raise RuntimeError(
            f"Could not download {filename!r} from Hugging Face Hub repo "
            f"{repo_id!r}. If you're trying the pretrained release "
            f"(OpenPHOTON/Qwen3-0.6B), the trained weights are a pending "
            f"upload -- pass a local checkpoint path instead in the "
            f"meantime (e.g. checkpoints/s3_final_ckpt.pt)."
        ) from e


def load_openphoton(source: str, device: str = "cpu",
                     dtype: Optional[torch.dtype] = None) -> OpenPhoton:
    """Build a PHOTON-Qwen3 model at the production (2,2) architecture
    (`configs.s3.S3Config().photon`) and load `source`'s weights into it.

    `source` is either a Hugging Face Hub repo id (e.g.
    "OpenPHOTON/Qwen3-0.6B") or a local checkpoint file/directory path
    (see `_resolve_checkpoint_path`). `dtype` defaults to the config's
    compute dtype (bfloat16); pass e.g. `torch.float32` to override.

    Returns an `OpenPhoton` bundle: `.model`, `.tokenizer`, `.cfg`.
    """
    cfg = S3Config().photon
    load_dtype = dtype if dtype is not None else cfg.torch_dtype

    ckpt_path = _resolve_checkpoint_path(source)

    model, stock = load_photon_qwen3(cfg, dtype=load_dtype)
    ckpt.load(ckpt_path, model=model, optimizer=None)
    model = model.to(device=device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
    return OpenPhoton(model=model, tokenizer=tokenizer, cfg=cfg, _stock=stock)


def _resolve(bundle_or_model):
    """Return (model, tokenizer) whether given an `OpenPhoton` bundle or a
    raw `PhotonQwen3ForCausalLM` (in which case the stock Qwen3 tokenizer,
    which this project never modifies, is loaded on demand)."""
    if isinstance(bundle_or_model, OpenPhoton):
        return bundle_or_model.model, bundle_or_model.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
    return bundle_or_model, tokenizer


def generate(bundle_or_model: Union[OpenPhoton, PhotonQwen3ForCausalLM], prompt: str,
             max_new_tokens: int = 40, temperature: float = 0.0, top_k: int = 0,
             top_p: float = 0.0, repetition_penalty: float = 1.0,
             seed: Optional[int] = None) -> str:
    """Generate text continuing `prompt` via RecGen (`inference.recgen`),
    the architecture's growing-KV-efficient autoregressive decode path.

    RecGen's prefill (`inference.recgen._prefill`) asserts
    `prompt_len % (C1*C2) == 0`, and there is no attention mask to make it
    skip padding, so `prompt` is left-padded with a pad/bos/eos token up to
    the next multiple of C1*C2 before prefill. Only the newly generated
    tokens are decoded back into text -- the padding never reaches the
    returned string.

    Decoding knobs (passed straight through to `inference.recgen`):
      * `temperature=0.0` (the default) is greedy argmax decoding.
      * `top_k>0` restricts sampling to the top-k logits.
      * `top_p` in (0, 1) enables nucleus sampling.
      * `repetition_penalty != 1.0` down-weights already-generated tokens
        (HF-style), which helps break RecGen's known longer-generation
        loops on the S3 checkpoint.
      * `seed`, when not None, seeds torch's RNG before the first sample so
        a temperature-sampled run is reproducible.
    """
    model, tokenizer = _resolve(bundle_or_model)
    device = next(model.parameters()).device
    chunk = model.cfg.C1 * model.cfg.C2

    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = ids.shape[1]
    pad_len = (-prompt_len) % chunk
    if pad_len:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.bos_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError(
                "tokenizer has no pad/bos/eos token id to left-pad the "
                "prompt with, so RecGen's prompt_len % (C1*C2) == 0 "
                "requirement can't be satisfied."
            )
        pad = ids.new_full((1, pad_len), pad_id)
        ids = torch.cat([pad, ids], dim=1)

    eos_id = tokenizer.eos_token_id
    out_ids = recgen_generate(model, ids, max_new_tokens=max_new_tokens,
                              temperature=temperature, top_k=top_k, top_p=top_p,
                              repetition_penalty=repetition_penalty, seed=seed,
                              eos_id=eos_id)
    new_ids = out_ids[0, prompt_len + pad_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def kv_footprint(bundle_or_model: Union[OpenPhoton, PhotonQwen3ForCausalLM], seq_len: int):
    """Analytic growing-KV entry count at `seq_len` tokens: `(vanilla,
    recgen, ratio)`. Delegates to `inference.recgen.kv_footprint`; accepts
    either an `OpenPhoton` bundle or a raw model."""
    model = bundle_or_model.model if isinstance(bundle_or_model, OpenPhoton) else bundle_or_model
    return _recgen_kv_footprint(model, seq_len)
