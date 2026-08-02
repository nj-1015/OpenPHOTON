"""PhotonQwen3ForCausalLM.forward -- the slide-21 forward pass.

Six new linkages (slide 19) wired here, outside the (unmodified) Qwen3
layers:
  1 chunk1  -- embed -> enc1, compress xC1
  2 chunk2  -- enc1 -> enc2, compress xC2
  3 shift+conv2 -- enc2 briefs dec2
  4 skip    -- enc1 states become dec2 inputs (teacher-forced)
  5 shift+conv1 -- dec2 briefs dec1
  6 skip    -- embeddings become dec1 inputs (teacher-forced)

The `[B, T/2, d] -> [B*T/4, R+C, d]` batch-dim reshape happens inside
reshape.pack(): the chunk index moves into the batch dimension so the
attention kernel in dec1/dec2 never sees a sequence longer than R+C.

bypass mode (cfg.bypass=True, forced C1=C2=R1=R2=1) additionally switches
the decoder stages from a local causal(R+C)/reset-position window to a full
causal(T)/absolute-position one -- see reshape.py's module docstring for why
that reduces the whole forward to running all 28 stock layers in order on
one [B, T, d] stream (the M5 golden-test target).
"""
import torch
import torch.nn as nn

from . import masks, reshape, rope
from .config import PhotonConfig
from .modules import Chunker, Converter


class PhotonQwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: PhotonConfig, embed_tokens, enc1, enc2, dec2, dec1,
                 norm, lm_head, rotary_emb, hf_config):
        super().__init__()
        self.cfg = cfg
        self.hf_config = hf_config
        self.d = hf_config.hidden_size

        # kept verbatim from stock Qwen3 (surgery.py slices these out)
        self.embed_tokens = embed_tokens
        self.enc1 = enc1
        self.enc2 = enc2
        self.dec2 = dec2
        self.dec1 = dec1
        self.norm = norm
        self.lm_head = lm_head
        self.rotary_emb = rotary_emb

        # genuinely new (slide 16). dtype follows the loaded stock model's
        # actual weight dtype (not cfg.dtype directly -- embed_tokens'
        # dtype is the ground truth for whatever the caller loaded), so a
        # bf16-loaded model doesn't feed bf16 activations into a silently
        # fp32 Chunker/Converter (found running the M5b bf16 golden leg --
        # see photon/modules.py).
        weight_dtype = embed_tokens.weight.dtype
        self.chunk1 = Chunker(self.d, cfg.C1, dtype=weight_dtype)
        self.chunk2 = Chunker(self.d, cfg.C2, dtype=weight_dtype)
        self.conv2 = Converter(self.d, cfg.R2, dtype=weight_dtype)   # briefs dec2 from h2
        self.conv1 = Converter(self.d, cfg.R1, dtype=weight_dtype)   # briefs dec1 from h1_hat
        self.z0_dec2 = nn.Parameter(torch.zeros(1, 1, self.d, dtype=weight_dtype))
        self.z0_dec1 = nn.Parameter(torch.zeros(1, 1, self.d, dtype=weight_dtype))

        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.embed_tokens.weight

    def _run_stage(self, layers, h, mask, position_ids):
        cos, sin = rope.build_cos_sin(self.rotary_emb, h, position_ids)
        for layer in layers:
            h = layer(h, attention_mask=mask, position_ids=position_ids,
                       position_embeddings=(cos, sin))
        return h

    def forward(self, input_ids: torch.LongTensor, return_intermediates: bool = False):
        cfg = self.cfg
        B, T = input_ids.shape
        device = input_ids.device
        assert T % (cfg.C1 * cfg.C2) == 0, \
            f"T={T} must be divisible by C1*C2={cfg.C1 * cfg.C2}"

        h0 = self.embed_tokens(input_ids)                      # [B, T, d]

        # --- enc1: chunk1, then full causal over the shrunk sequence ---
        h1 = self.chunk1(h0)                                    # [B, T/C1, d]
        M1 = h1.shape[1]
        pos1 = rope.absolute_position_ids(M1, B, device=device)
        mask1 = masks.full_causal_mask(M1, dtype=h1.dtype, device=device)
        h1 = self._run_stage(self.enc1, h1, mask1, pos1)

        # --- enc2: chunk2, then full causal over the further-shrunk seq ---
        h2 = self.chunk2(h1)                                    # [B, T/(C1*C2), d]
        M2 = h2.shape[1]
        pos2 = rope.absolute_position_ids(M2, B, device=device)
        mask2 = masks.full_causal_mask(M2, dtype=h2.dtype, device=device)
        h2 = self._run_stage(self.enc2, h2, mask2, pos2)

        # --- dec2: briefing from h2 (linkage 3), children = h1 (linkage 4) ---
        u2 = self.conv2(reshape.shift(h2, self.z0_dec2, bypass=cfg.bypass))
        z = reshape.pack(u2, h1, bypass=cfg.bypass)
        if cfg.bypass:
            Lz = z.shape[1]
            posz = rope.absolute_position_ids(Lz, B, device=device)
            maskz = masks.full_causal_mask(Lz, dtype=z.dtype, device=device)
        else:
            Lz = cfg.R2 + cfg.C2
            # reset-RoPE positions follow pack()'s row order directly:
            # briefing occupies local positions 0..R2-1, children occupy
            # R2..R2+C2-1 (pack() puts [briefing ; children] in that order,
            # take_last_C() then strips the briefing rows back off). So a
            # child attending back to the briefing sees it at a RoPE
            # relative offset of +R2 (child_position - briefing_position),
            # not +1 -- a deliberate S1 modeling choice (how "recent" the
            # global summary reads to the local decoder), not a silent one.
            # Left as-is for S0; revisit if S1 distillation wants briefing
            # felt as adjacent (R2=1) regardless of R2's actual width.
            posz = rope.decoder_position_ids(cfg.rope_mode, Lz, z.shape[0], device=device)
            maskz = masks.local_causal_mask(Lz, dtype=z.dtype, device=device)
        z_dec2 = self._run_stage(self.dec2, z, maskz, posz)
        h1_hat = reshape.take_last_C(z_dec2, cfg.C2, B=B, bypass=cfg.bypass)  # [B, M1, d]

        # --- dec1: briefing from h1_hat (linkage 5), children = h0 (linkage 6) ---
        u1 = self.conv1(reshape.shift(h1_hat, self.z0_dec1, bypass=cfg.bypass))
        z1 = reshape.pack(u1, h0, bypass=cfg.bypass)
        if cfg.bypass:
            Lz1 = z1.shape[1]
            posz1 = rope.absolute_position_ids(Lz1, B, device=device)
            maskz1 = masks.full_causal_mask(Lz1, dtype=z1.dtype, device=device)
        else:
            Lz1 = cfg.R1 + cfg.C1
            # same reset-RoPE layout as dec2 above: briefing at 0..R1-1,
            # children at R1..R1+C1-1, so briefing->child carries a +R1
            # relative offset.
            posz1 = rope.decoder_position_ids(cfg.rope_mode, Lz1, z1.shape[0], device=device)
            maskz1 = masks.local_causal_mask(Lz1, dtype=z1.dtype, device=device)
        z_dec1 = self._run_stage(self.dec1, z1, maskz1, posz1)
        h0_hat = reshape.take_last_C(z_dec1, cfg.C1, B=B, bypass=cfg.bypass)  # [B, T, d]

        assert h0_hat.shape == h0.shape

        logits = self.lm_head(self.norm(h0_hat))

        if return_intermediates:
            return logits, dict(h0=h0, h1=h1, h2=h2, z_dec2_input=z,
                                 h1_hat=h1_hat, h0_hat=h0_hat)
        return logits
