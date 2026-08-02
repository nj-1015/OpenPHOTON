"""S3 crux loss -- logit / KL distillation:

    L_S3 = KD  +  ce_weight * CE  +  rec_weight * rec_B
    KD    = (T^2) * mean_pos  D_KL( p_teacher || p_student )   [forward KL, full vocab]
    CE    = cross_entropy(student next-token logits, gold)      [optional; ce_weight=0 -> off]
    rec_B = 1 - cos( chunk2(h1_hat), h2.detach() )             [RecGen fixed-point anchor]

**The objective switch (vs S1/S2):** S1/S2 matched the teacher's HIDDEN
states; S3 matches its OUTPUT DISTRIBUTION. `student_logits` come from the
composed `photon(input_ids, return_intermediates=True)` forward (the tied
lm_head output); `teacher_logits` come from `harness.teacher`'s frozen
`.logits(input_ids)` (already `@torch.no_grad()` + detached).

**Position alignment (no shift for KD):** `student_logits[:, t]` and
`teacher_logits[:, t]` are BOTH "the distribution over the next token given
tokens 0..t" -- the SAME position. Forward KL is therefore a position-wise
distribution match with NO shift. (The optional CE term, by contrast, IS
shifted: `logits[:, :-1]` predict the gold `input_ids[:, 1:]`, standard
next-token CE.) Getting this right matters -- shifting the KD by one would
train the student to predict the teacher's PREVIOUS-position distribution.

**Full vocab, fp32 (not top-k, not bf16):** the KL is computed over the
whole 151,936-way distribution in fp32. Top-k would be an approximation we
don't need (peak memory is dominated by the student's own logits, which we
must materialize regardless), and a 152k-way log-softmax in bf16 is
numerically dicey for a value that goes straight into a gradient (same
rationale as s1_loss.relative_mse's fp32 discipline). The forward-KL
constant `p_t*log p_t` (negative teacher entropy) is included so the
reported `kd` scalar is a true KL (>= 0, and exactly 0 when the student
matches the teacher) rather than just the cross-entropy part -- this makes
`kd` directly interpretable as "nats away from the teacher".

**rec_B anchor:** identical form to `s2_loss`'s rec_B (fp32 cosine,
`h2.detach()` as the reference so the encoder isn't degraded to make
reconstruction trivial). Kept at a small `rec_weight` so the RecGen
self-consistency the S2 stage established doesn't drift once the hidden
states are no longer pinned by a hidden-matching loss.
"""
import torch
import torch.nn.functional as F

S3_LOSS_KEYS = ("kd", "ce", "rec_B")


def forward_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
               temperature: float = 1.0) -> torch.Tensor:
    """(T^2) * mean over positions of D_KL(softmax(teacher/T) || softmax(student/T)).

    Full-vocab, fp32. `teacher_logits` must already be detached (no gradient
    flows into the teacher). Returns a scalar >= 0 that is exactly 0 iff the
    student logits equal the teacher logits (up to an additive per-position
    constant, which softmax quotients out)."""
    T = temperature
    logp_s = F.log_softmax(student_logits.float() / T, dim=-1)
    logp_t = F.log_softmax(teacher_logits.float() / T, dim=-1)
    p_t = logp_t.exp()
    # per-position KL = sum_v p_t (log p_t - log p_s); mean over B,T; * T^2.
    kl = (p_t * (logp_t - logp_s)).sum(dim=-1)     # [B, T]
    return kl.mean() * (T * T)


def l_s3(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
         mid: dict, chunk2, gold_ids: torch.Tensor,
         kd_temperature: float = 1.0, ce_weight: float = 0.0,
         rec_weight: float = 0.1) -> tuple[torch.Tensor, dict]:
    """student_logits: `[B, T, V]` from `photon(input_ids, ...)` (grad-carrying).
    teacher_logits: `[B, T, V]` from `harness.teacher.logits(input_ids)` (detached).
    mid: the intermediates dict from the SAME composed forward (needs h1_hat/h2).
    chunk2: `photon.chunk2` (the live module, so `chunk2(h1_hat)` stays in graph).
    gold_ids: the input token ids `[B, T]` (used ONLY by the optional CE term).

    Returns `(total, {"kd": ..., "ce": ..., "rec_B": ...})` -- differentiable
    scalars (caller `.item()`s after backward, same discipline as train_s1/s2).
    """
    kd = forward_kl(student_logits, teacher_logits, temperature=kd_temperature)

    if ce_weight > 0.0:
        # shifted next-token CE: position t predicts gold token t+1.
        V = student_logits.shape[-1]
        ce = F.cross_entropy(
            student_logits[:, :-1].float().reshape(-1, V),
            gold_ids[:, 1:].reshape(-1))
    else:
        # keep a real (zero) tensor so the reported dict is uniform; no graph.
        ce = student_logits.new_zeros(())

    # RecGen fixed-point anchor -- verbatim rec_B form from s2_loss.
    cos_B = F.cosine_similarity(chunk2(mid["h1_hat"]).float(),
                                mid["h2"].detach().float(), dim=-1)
    rec_B = (1.0 - cos_B).mean()

    total = kd + ce_weight * ce + rec_weight * rec_B
    return total, {"kd": kd, "ce": ce, "rec_B": rec_B}
