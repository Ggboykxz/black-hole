"""Black Hole — decoder-only Transformer language model, implemented from scratch.

No pretrained weights, no ``AutoModelForCausalLM``: every layer below is written here.

Design choices
--------------
* RMSNorm pre-norm residuals (plus a final RMSNorm before the LM head).
* RoPE rotary position embeddings applied to Q/K, with the "rotate half" formulation.
* Grouped-query attention (GQA): ``n_kv_heads <= n_heads``; MQA when ``n_kv_heads == 1``.
* SwiGLU feed-forward, sized to keep the parameter count close to a classic 4x FFN.
* Optional weight tying between the token embedding and the LM head.
* GPT-2 scaled initialization for residual-branch output projections.
* Causal attention via an explicit additive mask (memory-safe for long contexts, and
  avoids the sdpa-mask interplay bugs when compiling).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .config import BlackHoleConfig

__all__ = ["BlackHole", "BlackHoleBlock", "RMSNorm", "RoPE", "CausalSelfAttention", "SwiGLU"]


# --------------------------------------------------------------------------- normalization
class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (x * self.weight.float()).to(dtype)


# --------------------------------------------------------------------------- rotary embeddings
class RoPE(nn.Module):
    """Rotary position embeddings (Su et al., 2021), precomputed up to ``max_seq_len``."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # cache: (max_seq_len, head_dim/2, 2) cos/sin
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)                    # (T, head_dim/2)
        # NB: ``cat`` (not ``stack``) so that freq[i] == freq[i + head_dim/2], which is
        # exactly the pairing used by rotate_half -> the rotation stays orthogonal.
        emb = torch.cat((freqs, freqs), dim=-1)             # (T, head_dim)
        self.register_buffer("cos_cached", emb.cos()[None], persistent=False)  # (1,T,D)
        self.register_buffer("sin_cached", emb.sin()[None], persistent=False)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, pos_offset: int = 0):
        """Apply rotation. q,k: (B, H, T, D)."""
        seq_len = q.shape[-2]
        cos = self.cos_cached[:, pos_offset : pos_offset + seq_len].to(q.dtype)
        sin = self.sin_cached[:, pos_offset : pos_offset + seq_len].to(q.dtype)
        q_rot = q * cos + self.rotate_half(q) * sin
        k_rot = k * cos + self.rotate_half(k) * sin
        return q_rot, k_rot


# --------------------------------------------------------------------------- attention
class CausalSelfAttention(nn.Module):
    """Multi-head / grouped-query causal self-attention."""

    def __init__(self, cfg: BlackHoleConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)

        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPE,
        attn_mask: Optional[torch.Tensor] = None,
        pos_offset: int = 0,
    ) -> torch.Tensor:
        b, t, _ = x.shape

        q = rearrange(self.wq(x), "b t (h d) -> b h t d", h=self.n_heads, d=self.head_dim)
        k = rearrange(self.wk(x), "b t (h d) -> b h t d", h=self.n_kv_heads, d=self.head_dim)
        v = rearrange(self.wv(x), "b t (h d) -> b h t d", h=self.n_kv_heads, d=self.head_dim)

        q, k = rope(q, k, pos_offset=pos_offset)

        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # scaled_dot_product_attention supports an additive float mask (1e-4/-inf safe)
        # and computes softmax in fp32 internally when given fp16/bf16 inputs.
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=attn_mask is None,
        )
        y = rearrange(y, "b h t d -> b t (h d)")
        return self.resid_dropout(self.wo(y))


# --------------------------------------------------------------------------- feed-forward
class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (Shazeer, 2020): w2( silu(w1 x) * w3 x )."""

    def __init__(self, cfg: BlackHoleConfig) -> None:
        super().__init__()
        hidden = cfg.mlp_dim
        self.w1 = nn.Linear(cfg.dim, hidden, bias=False)   # gate
        self.w3 = nn.Linear(cfg.dim, hidden, bias=False)   # up
        self.w2 = nn.Linear(hidden, cfg.dim, bias=False)   # down
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# --------------------------------------------------------------------------- transformer block
class BlackHoleBlock(nn.Module):
    """Pre-norm residual block: x + proj(norm(x)) applied twice."""

    def __init__(self, cfg: BlackHoleConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)
        # GPT-2 scaled init: keep early residual branches small
        for module in (self.attn.wo, self.mlp.w2):
            nn.init.normal_(module.weight, mean=0.0, std=cfg.resid_scale)

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPE,
        attn_mask: Optional[torch.Tensor] = None,
        pos_offset: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope, attn_mask=attn_mask, pos_offset=pos_offset)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# --------------------------------------------------------------------------- full model
class BlackHole(nn.Module):
    """Autoregressive language model: token embedding -> N blocks -> RMSNorm -> LM head."""

    def __init__(self, cfg: BlackHoleConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([BlackHoleBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        self.rope = RoPE(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)

        self.apply(self._init_weights)
        # re-apply scaled init after the generic init (apply visits children last-to-first
        # ordering is not guaranteed, so be explicit)
        for block in self.blocks:
            for proj in (block.attn.wo, block.mlp.w2):
                nn.init.normal_(proj.weight, mean=0.0, std=cfg.resid_scale)

        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        # report non-embedding params
        self.n_params = sum(p.numel() for p in self.parameters())
        if cfg.tie_weights:
            self.n_params -= self.lm_head.weight.numel()

    # ------------------------------------------------------------------ init
    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        pos_offset: int = 0,
    ):
        """idx: (B, T) int64. targets: (B, T) int64 or None.

        Returns (logits, loss) where loss is cross-entropy if targets given.
        """
        if idx.shape[-1] > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {idx.shape[-1]} exceeds model max_seq_len {self.cfg.max_seq_len}"
            )
        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, self.rope, attn_mask=attn_mask, pos_offset=pos_offset)
        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                rearrange(logits.float(), "b t v -> (b t) v"),
                rearrange(targets, "b t -> (b t)"),
                ignore_index=-100,
            )
        else:
            # inference: only compute the last position's logits (memory friendly)
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    # ------------------------------------------------------------------ generation helpers
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        greedy: bool = False,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling with temperature / top-k / top-p."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.max_seq_len else idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if greedy:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-6)
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cumulative = torch.cumsum(probs, dim=-1)
                    remove = cumulative - probs > top_p          # keep first token above threshold
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_tok], dim=1)
            if eos_token_id is not None and bool((next_tok == eos_token_id).all()):
                break
        return idx

    # ------------------------------------------------------------------ serialization
    def num_params(self, non_embedding: bool = True) -> int:
        return self.n_params if non_embedding else sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def estimate_flops(self, seq_len: int) -> float:
        """Rough forward-pass FLOPs per token (6N + 2*n_layers*seq_len*dim)."""
        n = self.num_params(non_embedding=True)
        cfg = self.cfg
        attention = 2 * cfg.n_layers * seq_len * cfg.dim
        return 6 * n + 2 * attention
