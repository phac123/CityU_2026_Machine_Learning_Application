"""Batched SMILES generation on top of :class:`MolGPT`.

The model's ``generate`` returns token ids; this module owns everything around
it: seeding with <bos>, stopping at <eos>, decoding to strings, dropping invalid
strings, deduplicating, and (optionally) conditioning on a property bucket.
Keeping this logic outside the model makes it trivially reusable for the
optional HuggingFace adapter.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import torch

from .model import MolGPT
from .tokenizer import SmilesTokenizer


@torch.no_grad()
def sample_smiles(
    model: MolGPT,
    tokenizer: SmilesTokenizer,
    num_samples: int,
    device: torch.device,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    cond: Optional[Iterable[int]] = None,
    batch_size: int = 256,
    generator: Optional[torch.Generator] = None,
) -> List[str]:
    """Draw ``num_samples`` SMILES strings, batching the autoregressive loop.

    Args:
        cond: optional iterable of bucket ids, cycled over the requested samples.
    Returns:
        Decoded SMILES strings (may contain invalid chemistry; the caller
        validates them with RDKit, exactly as the paper does for any generator).
    """
    model.eval()
    cond_list: Optional[List[int]] = list(cond) if cond is not None else None
    if cond_list is not None and len(cond_list) == 0:
        cond_list = None

    out: List[str] = []
    done = 0
    while done < num_samples:
        cur = min(batch_size, num_samples - done)
        idx = torch.full((cur, 1), tokenizer.bos_id, dtype=torch.long, device=device)
        if cond_list is not None:
            c = torch.tensor(
                [cond_list[(done + i) % len(cond_list)] for i in range(cur)],
                dtype=torch.long, device=device,
            )
        else:
            c = None
        seq = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            cond=c if model.cond_emb is not None else None,
            eos_token_id=tokenizer.eos_id,
        )
        # strip the leading <bos> that seeded the sequence
        for row in seq.cpu().tolist():
            tokens = row[1:]
            if tokenizer.eos_id in tokens:
                tokens = tokens[: tokens.index(tokenizer.eos_id)]
            out.append(tokenizer.decode(tokens, skip_special=True))
        done += cur
    return out


def dedupe(smiles: Iterable[str]) -> List[str]:
    """Keep the first occurrence of each non-empty string, preserving order."""
    seen = set()
    out: List[str] = []
    for s in smiles:
        s = s.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


@torch.no_grad()
def sample_valid_smiles(
    model: MolGPT,
    tokenizer: SmilesTokenizer,
    num_samples: int,
    device: torch.device,
    batch_size: int = 256,
    max_rounds: int = 20,
    **kwargs,
) -> List[str]:
    """Sample until ``num_samples`` *RDKit-parseable, unique* molecules exist.

    Rejection sampling is the standard protocol for SMILES generators: validity
    is a property of the produced string, so invalid strings are simply
    discarded, and the reported validity rate counts them.
    """
    from .chem import is_valid_smiles  # local import: RDKit-dependent

    collected: List[str] = []
    seen = set()
    rounds = 0
    while len(collected) < num_samples and rounds < max_rounds:
        rounds += 1
        raw = sample_smiles(
            model, tokenizer, num_samples=num_samples, device=device,
            batch_size=batch_size, **kwargs,
        )
        for s in raw:
            if s in seen or not is_valid_smiles(s):
                continue
            seen.add(s)
            collected.append(s)
            if len(collected) >= num_samples:
                break
    return collected
