"""Load Qwen3-0.6B, slice the live 28-layer ModuleList 7/7/7/7 (no
key-renaming -- slicing preserves object identity, so "verbatim, in order"
is true by construction; see test in tests/test_config.py that confirms
`sub[i] is full[i]`), keep embed_tokens/model.norm/lm_head verbatim, and
attach the four new modules.

`AutoModelForCausalLM.from_pretrained` already loads and ties all weights
(embed_tokens/lm_head), so this module does not re-map any state-dict keys
into a from-scratch module -- it just slices live submodules out of the
already-loaded stock model.
"""
import torch
from transformers import AutoModelForCausalLM

from .config import PhotonConfig
from .model import PhotonQwen3ForCausalLM

MODEL_ID = "Qwen/Qwen3-0.6B"


def load_stock_qwen3(model_id: str = MODEL_ID, dtype: torch.dtype = torch.float32):
    """fp32/CPU/eager -- deterministic, no fused kernels, so results are
    directly comparable to the golden-test reference captures."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, attn_implementation="eager"
    )
    model.eval()
    return model


def build_photon_from_stock(stock_model, cfg: PhotonConfig) -> PhotonQwen3ForCausalLM:
    hf_cfg = stock_model.config
    s0, s1, s2, s3 = cfg.split
    assert s0 + s1 + s2 + s3 == hf_cfg.num_hidden_layers, (
        f"split {cfg.split} does not cover num_hidden_layers={hf_cfg.num_hidden_layers}"
    )

    layers = stock_model.model.layers          # live nn.ModuleList[28]
    enc1 = layers[0:s0]
    enc2 = layers[s0:s0 + s1]
    dec2 = layers[s0 + s1:s0 + s1 + s2]
    dec1 = layers[s0 + s1 + s2:s0 + s1 + s2 + s3]

    photon = PhotonQwen3ForCausalLM(
        cfg=cfg,
        embed_tokens=stock_model.model.embed_tokens,
        enc1=enc1, enc2=enc2, dec2=dec2, dec1=dec1,
        norm=stock_model.model.norm,
        lm_head=stock_model.lm_head,
        rotary_emb=stock_model.model.rotary_emb,
        hf_config=hf_cfg,
    )
    photon.eval()
    return photon


def load_photon_qwen3(cfg: PhotonConfig, model_id: str = MODEL_ID,
                       dtype: torch.dtype | None = None) -> tuple[PhotonQwen3ForCausalLM, "AutoModelForCausalLM"]:
    """Convenience: load + slice + attach in one call. Returns
    (photon_model, stock_model) -- keep stock_model alive, photon shares its
    parameters by reference (no copies).

    dtype defaults to `cfg.torch_dtype` (i.e. `cfg.dtype`, S0's single source
    of truth for load precision) -- pass an explicit dtype only to override
    the config for a one-off run.
    """
    if dtype is None:
        dtype = cfg.torch_dtype
    stock = load_stock_qwen3(model_id, dtype=dtype)
    photon = build_photon_from_stock(stock, cfg)
    return photon, stock
