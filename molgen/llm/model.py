"""Decoder-only (causal) Transformer language model over SMILES.

This is the "LLM" branch of the project: a GPT-style autoregressive model that
learns p(s_t | s_<t) over SMILES tokens.  It is deliberately small (a few
million parameters) because it is trained from scratch on QM9, but the
architecture is the standard pre-LayerNorm Transformer block with weight-tied
input/output embeddings, so the same code scales to larger depths/widths.

Optional conditional mode: a discrete property bucket (e.g. the decile of QED)
is embedded once and added to every token embedding, which turns the model into
p(s_t | s_<t, c).  This is the "prefix-free" way of conditioning an LLM, and it
is what ``generate(cond=...)`` uses at sampling time.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head scaled dot-product attention with a causal mask."""

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        # 1 x 1 x L x L lower-triangular mask, registered lazily in forward
        self.register_buffer("mask", torch.empty(0), persistent=False)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        if self.mask.numel() != length * length or self.mask.device != device:
            m = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
            self.mask = m.view(1, 1, length, length)
        return self.mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, length, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        # (B, L, D) -> (B, H, L, D/H)
        q = q.view(b, length, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, length, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, length, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(~self._causal_mask(length, x.device), float("-inf"))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, length, d)
        return self.proj(y)


class MLP(nn.Module):
    """Position-wise feed-forward network (GELU, 4x expansion)."""

    def __init__(self, d_model: int, ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(ratio * d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Pre-LN Transformer block: x + Attn(LN(x)); x + MLP(LN(x))."""

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MolGPT(nn.Module):
    """Causal Transformer that models SMILES token sequences.

    Args:
        vocab_size: number of SMILES tokens (including the 4 special tokens).
        max_len: maximum sequence length, also the position-embedding table size.
        d_model / n_head / n_layer: width, heads, depth.
        dropout: dropout on attention, MLP and embeddings.
        n_cond: number of discrete property buckets (0 disables conditioning).
        tie_embeddings: share the token embedding with the output projection.
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int = 64,
        d_model: int = 256,
        n_head: int = 8,
        n_layer: int = 6,
        dropout: float = 0.1,
        n_cond: int = 0,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_cond = int(n_cond)

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.cond_emb = nn.Embedding(self.n_cond, d_model) if self.n_cond > 0 else None
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # scale down residual projections (GPT-2 recipe) for stable deep stacks
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("net.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding: bool = False) -> int:
        """Parameter count (optionally excluding the position table)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n

    def forward(
        self,
        input_ids: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the model.

        Args:
            input_ids: [B, L] token ids (teacher-forcing inputs).
            cond: [B] property bucket ids, required when n_cond > 0.
            targets: [B, L] next-token labels; if given the loss is returned.
            loss_mask: [B, L] weights for the per-token loss (padding -> 0).
        Returns:
            (logits [B, L, V], loss scalar or None)
        """
        b, length = input_ids.shape
        if length > self.max_len:
            raise ValueError(f"sequence length {length} exceeds max_len {self.max_len}")

        pos = torch.arange(length, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        if self.cond_emb is not None and cond is not None:
            x = x + self.cond_emb(cond.long()).unsqueeze(1)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            flat = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                reduction="none",
            ).view(b, length)
            if loss_mask is not None:
                flat = flat * loss_mask
                loss = flat.sum() / loss_mask.sum().clamp_min(1.0)
            else:
                loss = flat.mean()
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        cond: Optional[torch.Tensor] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive sampling without a KV cache.

        Re-running the full prefix at every step is O(L^2) per token, which is
        acceptable for molecules (L <= 64) and keeps the code short; the report
        discusses the KV-cache variant as a straightforward optimisation.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]
            logits, _ = self.forward(idx_cond, cond=cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)[0][..., -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_token_id is not None and bool((nxt == eos_token_id).all()):
                break
        return idx
