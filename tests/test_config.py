"""M0 -- environment & config pin.

Confirms the installed `transformers` version's `Qwen3DecoderLayer.forward`
/ `Qwen3Attention.forward` signatures still accept the external
attention_mask / position_ids / position_embeddings seam PHOTON relies on,
and asserts every slide-4 config value for Qwen/Qwen3-0.6B. Cheapest place
to catch the head_dim(128) != hidden(1024) trap and any transformers API
drift (S0 plan risk list items 2, 10).
"""
import inspect

import pytest
import torch
import transformers
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

MODEL_ID = "Qwen/Qwen3-0.6B"


def test_transformers_version_pinned_and_printed():
    # Not a hard pin (no requirements.txt lock in this CPU-only S0 pass) --
    # just make the installed version visible in CI/test output so a future
    # API drift is easy to correlate.
    print("transformers version:", transformers.__version__)
    print("torch version:", torch.__version__)
    assert transformers.__version__  # sanity: importable, non-empty


def test_decoder_layer_forward_signature_has_photon_seam():
    sig = inspect.signature(Qwen3DecoderLayer.forward)
    params = sig.parameters
    for name in ("hidden_states", "attention_mask", "position_ids", "position_embeddings"):
        assert name in params, f"Qwen3DecoderLayer.forward lost param {name!r} -- API drift"


def test_attention_forward_signature_has_position_embeddings():
    sig = inspect.signature(Qwen3Attention.forward)
    params = sig.parameters
    assert "position_embeddings" in params
    assert "attention_mask" in params


def test_rotary_embedding_forward_signature():
    sig = inspect.signature(Qwen3RotaryEmbedding.forward)
    params = sig.parameters
    assert "x" in params
    assert "position_ids" in params


def test_qwen3_0_6b_config_matches_slide_4():
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    assert cfg.hidden_size == 1024
    assert cfg.num_hidden_layers == 28
    assert cfg.num_attention_heads == 16
    assert cfg.num_key_value_heads == 8
    assert cfg.head_dim == 128
    assert cfg.rope_parameters["rope_theta"] == 1_000_000
    assert cfg.tie_word_embeddings is True
    assert cfg.vocab_size == 151936

    # the head_dim trap (S0 plan risk #2): 16*128 = 2048 != hidden_size 1024
    assert cfg.num_attention_heads * cfg.head_dim == 2048
    assert cfg.num_attention_heads * cfg.head_dim != cfg.hidden_size


def test_qk_norm_is_rmsnorm_on_head_dim():
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    layer = Qwen3DecoderLayer(cfg, layer_idx=0)
    assert isinstance(layer.self_attn.q_norm, Qwen3RMSNorm)
    assert isinstance(layer.self_attn.k_norm, Qwen3RMSNorm)
    assert layer.self_attn.q_norm.weight.shape[0] == cfg.head_dim


def test_photon_config_dtype_is_wired_not_dead():
    """reviewer nit: PhotonConfig.dtype must resolve to a real torch.dtype
    and be the value surgery.load_photon_qwen3() actually loads at (checked
    here cheaply, without a model load, via the .torch_dtype property that
    load_photon_qwen3 defaults its `dtype` arg from -- see surgery.py)."""
    from photon.config import PhotonConfig

    cfg = PhotonConfig()
    assert cfg.dtype == "float32"
    assert cfg.torch_dtype is torch.float32

    cfg_bf16 = PhotonConfig(dtype="bfloat16")
    assert cfg_bf16.torch_dtype is torch.bfloat16

    with pytest.raises(ValueError):
        PhotonConfig(dtype="not_a_real_dtype")
