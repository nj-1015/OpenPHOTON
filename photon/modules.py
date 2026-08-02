"""The two new PHOTON modules — Chunker and Converter (slide 20).

Neither is random-initialised. Chunker starts at the mean of its C inputs;
Converter starts by replicating its parent R times. At step 0 the network
then computes "average-pool, then broadcast" -- a lossy but sane function
instead of noise, so S1 distillation is refinement rather than discovery.
"""
import torch
import torch.nn as nn


class Chunker(nn.Module):
    """concat(C x d) -> d.  Folds C sibling states into one parent latent."""

    def __init__(self, d: int, C: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d = d
        self.C = C
        # dtype must match the stream this module sits in (embed_tokens'
        # loaded dtype, threaded down from model.py) -- a bf16-loaded model
        # feeding a silently-fp32 Chunker/Converter is a dtype mismatch at
        # the very first linkage, found running the M5b bf16 golden leg.
        self.proj = nn.Linear(C * d, d, bias=False, dtype=dtype)
        self._init_identity_mean()

    @torch.no_grad()
    def _init_identity_mean(self):
        # weight blocks [I/C ; I/C ; ... ; I/C] so proj(concat(x_0..x_{C-1}))
        # == mean_c(x_c) exactly at init.
        dtype = self.proj.weight.dtype
        w = torch.zeros(self.d, self.C * self.d, dtype=dtype)
        block = torch.eye(self.d, dtype=dtype) / self.C
        for c in range(self.C):
            w[:, c * self.d:(c + 1) * self.d] = block
        self.proj.weight.copy_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, M, d]  ->  [B, M//C, d]
        B, M, d = x.shape
        assert d == self.d, f"Chunker expected d={self.d}, got {d}"
        assert M % self.C == 0, f"Chunker seq len {M} not divisible by C={self.C}"
        x = x.reshape(B, M // self.C, self.C * d)
        return self.proj(x)


class Converter(nn.Module):
    """d -> R x d.  Expands one parent latent into R conditioning vectors."""

    def __init__(self, d: int, R: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d = d
        self.R = R
        self.proj = nn.Linear(d, R * d, bias=False, dtype=dtype)
        self._init_replicate()

    @torch.no_grad()
    def _init_replicate(self):
        # weight blocks [I ; I ; ... ; I] (stacked on the output axis) so
        # proj(z) == R exact copies of z at init.
        dtype = self.proj.weight.dtype
        w = torch.zeros(self.R * self.d, self.d, dtype=dtype)
        eye = torch.eye(self.d, dtype=dtype)
        for r in range(self.R):
            w[r * self.d:(r + 1) * self.d, :] = eye
        self.proj.weight.copy_(w)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, M, d]  ->  [B, M, R, d]
        B, M, d = z.shape
        assert d == self.d, f"Converter expected d={self.d}, got {d}"
        u = self.proj(z)
        return u.view(B, M, self.R, d)
