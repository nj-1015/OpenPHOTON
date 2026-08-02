"""RecGen generation for PHOTON-Qwen3 (paper Appendix A: Defs A.2-A.4,
Theorem A.6, Eq 6/13; Section 2.1.2/2.3.3).

RecGen = "recursive generation": at decode time the bottom-up encoder is
NOT re-run over generated tokens. The top-down decoder's own reconstruction
(`h1_hat`) is fed back up through the chunker to refresh the one growing
stream (enc2). Only enc2 keeps a growing KV cache -- 7 of the 28 layers,
over the T/(C1*C2) coarsest stream -- which is where the ~16x growing-KV
reduction comes from. enc1 and chunk1 run at PREFILL only.

Index conventions (the two off-by-ones the paper distinguishes):
  * ACROSS chunks (briefing, Eq 6): chunk g conditions on the PARENT at g-1,
    with a learned start latent z0 at g=0 (our `reshape.shift(., z0)`).
  * CONTINUOUS next-token at the bottom: token n+1 is sampled from the logits
    of `h0_hat[n]`, where n is a CHILD position of its chunk's local window.
    The R briefing-row outputs are discarded by `take_last_C` in training, so
    they are NEVER trained to predict a token -- generating a token from a
    briefing-only window would read untrained outputs. So the bottom level is
    a normal next-token loop; the chunk structure only decides which briefing
    (h1_hat[g-1]) and window each h0_hat[n] uses. Crossing a chunk boundary
    is automatic: the last child of chunk g-1 predicts the first token of
    chunk g.
  * `dec2` (level-1 latents) IS fed autoregressively from its OWN continuous
    outputs (no sampling) -- children = h1_hat[<j], position 0 sees only the
    R2 briefing. h1_hat[g-1] briefs bottom chunk g; the level-1 latents are
    produced one meta-context (C2 of them) at a time, lazily, just before the
    bottom loop needs them.

Local decoders: reset RoPE (0..R+C-1) + local causal mask, IDENTICAL to
training. enc2 (the streaming stream) uses absolute positions over its own
short sequence, never reset.

Exactness (Theorem A.6): RecGen == the non-recursive HierGen iff h1_hat == h1
(Def A.4 recursive consistency = what L_rec/rec_A trains). A trained-but-
imperfect model has h1_hat != h1, so RecGen output diverges from an exact
decode by exactly the reconstruction error -- `recgen_fidelity()` measures it.
Do NOT expect recgen_generate to equal model.forward on a real checkpoint;
the VERIFIABLE gate is `verify_dec1_plumbing` (the bottom-level windowed
decode reproduces model.forward when fed model.forward's own h1_hat), which
isolates the indexing/positions/masks from model quality.

enc2 refresh re-runs enc2 over the (short) grown level-2 sequence each
meta-context rather than an incremental KV STEP -- numerically identical
(AR/KV-cache equivalence), keeps the code verifiable, and the memory story
(only the level-2 latents resident) is unchanged.
"""
import torch
import torch.nn.functional as F

from photon import masks, rope


# ----------------------------------------------------------------------------
# stage helpers (reuse the frozen model's own submodules + orchestration)
# ----------------------------------------------------------------------------
def _local_decode(model, layers, seq):
    """Local decoder stack (dec1/dec2) on `seq` [n, L, d], reset RoPE + local
    causal mask -- exactly the training-time local window."""
    n, L, _ = seq.shape
    pos = rope.decoder_position_ids(model.cfg.rope_mode, L, n, device=seq.device)
    mask = masks.local_causal_mask(L, dtype=seq.dtype, device=seq.device)
    return model._run_stage(layers, seq, mask, pos)


def _enc_stage(model, layers, h):
    """Full-causal encoder stage (enc1/enc2) over h [B, M, d], absolute
    positions -- identical to model.forward's encoder calls."""
    M = h.shape[1]
    pos = rope.absolute_position_ids(M, h.shape[0], device=h.device)
    mask = masks.full_causal_mask(M, dtype=h.dtype, device=h.device)
    return model._run_stage(layers, h, mask, pos)


def _brief(converter, parent, R, d):
    """conv{1,2}(parent[1,1,d]) -> [1, R, d] briefing rows."""
    return converter(parent).reshape(1, R, d)


def _sample(logits, temperature=0.0, top_k=0):
    if temperature <= 0.0:
        return int(logits.argmax())
    if top_k and top_k < logits.shape[-1]:
        v, _ = torch.topk(logits, top_k)
        logits = logits.masked_fill(logits < v[-1], float("-inf"))
    p = F.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(p, 1))


def _dec2_meta(model, U2):
    """One meta-context of dec2: produce C2 level-1 reconstructions h1_hat,
    children = dec2's own previous continuous outputs (position 0 sees only
    the R2 briefing). Returns list of C2 tensors [1,1,d]."""
    C2 = model.cfg.C2
    h1_hats = []
    for j in range(C2):
        seq = U2 if j == 0 else torch.cat([U2] + h1_hats, dim=1)
        out = _local_decode(model, model.dec2, seq)
        h1_hats.append(out[:, -1:, :])
    return h1_hats


def _h0_hat_at(model, ids, n, brief_src):
    """Bottom-level windowed decode: h0_hat[n] for token position n, using
    chunk g=n//C1's briefing conv1(brief_src) + the true/generated embeddings
    of tokens [g*C1 .. n]. Returns logits [V] that predict token n+1."""
    C1, R1, d = model.cfg.C1, model.cfg.R1, model.d
    g = n // model.cfg.C1
    U1 = _brief(model.conv1, brief_src, R1, d)
    cs = g * C1
    embeds = model.embed_tokens(ids[:, cs:n + 1])           # [1, n+1-cs, d]
    seq = torch.cat([U1, embeds], dim=1)
    out = _local_decode(model, model.dec1, seq)
    return model.lm_head(model.norm(out[:, -1, :]))[0]       # [V]


# ----------------------------------------------------------------------------
# verifiable plumbing gate
# ----------------------------------------------------------------------------
@torch.no_grad()
def verify_dec1_plumbing(model, ids, atol=1e-4, rtol=1e-4):
    """Reproduce model.forward's logits position-by-position with the
    continuous windowed bottom decode, using model.forward's OWN h1_hat as
    the per-chunk briefings and the true tokens as children. This isolates
    the bottom-level indexing/shift/positions/mask from model quality: it
    MUST match model.forward exactly (same computation, done per-position).
    Returns (max_abs_diff, ok)."""
    C1 = model.cfg.C1
    ref_logits, mid = model(ids, return_intermediates=True)
    h1_hat = mid["h1_hat"]                                   # [1, M1, d]
    T = ids.shape[1]
    got = []
    for n in range(T):
        g = n // C1
        brief_src = model.z0_dec1 if g == 0 else h1_hat[:, g - 1:g, :]
        got.append(_h0_hat_at(model, ids, n, brief_src))
    got = torch.stack(got, dim=0).unsqueeze(0)              # [1, T, V]
    diff = (got.float() - ref_logits.float()).abs().max().item()
    return diff, bool(torch.allclose(got.float(), ref_logits.float(), atol=atol, rtol=rtol))


# ----------------------------------------------------------------------------
# prefill + generate
# ----------------------------------------------------------------------------
@torch.no_grad()
def _prefill(model, prompt_ids):
    """Bootstrap RecGen state from the prompt via the full (encoder-using)
    forward: all level-1 reconstructions h1_hat, the level-2 input stream
    a_stream = chunk2(h1_hat), and its enc2 output h2_stream. enc1/chunk1
    run HERE only. Requires len(prompt) % (C1*C2) == 0."""
    C1, C2 = model.cfg.C1, model.cfg.C2
    assert prompt_ids.shape[1] % (C1 * C2) == 0, "prompt length must be a multiple of C1*C2"
    _, mid = model(prompt_ids, return_intermediates=True)
    h1_hat = mid["h1_hat"]                                  # [1, M1, d]
    a_stream = model.chunk2(h1_hat)                        # [1, M2, d]
    h2_stream = _enc_stage(model, model.enc2, a_stream)
    h1_list = [h1_hat[:, i:i + 1, :] for i in range(h1_hat.shape[1])]
    return h1_list, a_stream, h2_stream


@torch.no_grad()
def recgen_generate(model, prompt_ids, max_new_tokens=40, temperature=0.0,
                    top_k=0, eos_id=None):
    """Autoregressive RecGen generation. enc1 is never run after prefill;
    level-1 latents are produced lazily one meta-context at a time (dec2),
    and the bottom level is a continuous next-token loop. Returns the full
    id sequence [1, P + n_new]."""
    C1, C2, d, R2 = model.cfg.C1, model.cfg.C2, model.d, model.cfg.R2
    device = prompt_ids.device
    h1_list, a_stream, h2_stream = _prefill(model, prompt_ids)
    ids = prompt_ids
    sk = dict(temperature=temperature, top_k=top_k)

    def ensure_h1(idx):
        """Lazily decode meta-contexts (dec2 + stream refresh) until h1_list
        has level-1 latent `idx`."""
        nonlocal a_stream, h2_stream
        while len(h1_list) <= idx:
            U2 = _brief(model.conv2, h2_stream[:, -1:, :], R2, d)   # parent g-1
            new = _dec2_meta(model, U2)                             # C2 latents
            h1_list.extend(new)
            a_new = model.chunk2(torch.cat(new, dim=1))            # [1,1,d]
            a_stream = torch.cat([a_stream, a_new], dim=1)
            h2_stream = _enc_stage(model, model.enc2, a_stream)

    n_new = 0
    while n_new < max_new_tokens:
        n = ids.shape[1] - 1                # last placed token; its logits -> token n+1
        g = n // C1
        if g == 0:
            brief_src = model.z0_dec1
        else:
            ensure_h1(g - 1)
            brief_src = h1_list[g - 1]
        logits = _h0_hat_at(model, ids, n, brief_src)
        tok = _sample(logits, **sk)
        ids = torch.cat([ids, torch.tensor([[tok]], device=device)], dim=1)
        n_new += 1
        if eos_id is not None and tok == eos_id:
            break
    return ids


# ----------------------------------------------------------------------------
# growing-KV accounting (the point of the architecture)
# ----------------------------------------------------------------------------
def kv_footprint(model, seq_len):
    """Analytic growing-KV entry count at `seq_len` tokens. Vanilla: every
    token at every layer. PHOTON-RecGen: only enc2's stream (its layers x
    seq_len/(C1*C2) positions); local decoder windows are bounded (R+C) and
    dropped at chunk boundaries -> excluded. Returns (vanilla, recgen, ratio)."""
    n_layers = len(model.enc1) + len(model.enc2) + len(model.dec1) + len(model.dec2)
    vanilla = n_layers * seq_len
    recgen = len(model.enc2) * (seq_len // (model.cfg.C1 * model.cfg.C2))
    return vanilla, recgen, vanilla / max(recgen, 1)
