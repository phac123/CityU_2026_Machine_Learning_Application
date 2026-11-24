"""Sampling-based analysis, shared by the training loop and ``evaluate.py`` (LLM).

The language model produces strings, not coordinates, so the "analysis" is: sample N
SMILES, score them (validity / uniqueness / novelty on canonical strings, plus the 3D
metrics on the embedded conformers when asked), and place the accepted molecules on
the descriptor manifold of the training split (PCA).  The training loop calls
:func:`analyse` every ``train.analysis_every`` steps and at the last step, and
``evaluate.py`` calls the same function, so the two never disagree.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import torch


# --------------------------------------------------------------------------- #
# sampling + string metrics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample_strings(cfg, model, tokenizer, device: torch.device,
                   num: Optional[int] = None, temperature: Optional[float] = None,
                   top_k: Optional[int] = None, cond: Optional[List[int]] = None,
                   batch_size: Optional[int] = None) -> List[str]:
    """Draw ``num`` SMILES with the current weights."""
    from .sampler import sample_smiles

    model.eval()
    # the language model has no data.eval_num: fall back to train.sample_num
    default_num = int(getattr(cfg.train, "sample_num", 128))
    return sample_smiles(
        model,
        tokenizer,
        int(num) if num is not None else default_num,
        device,
        max_new_tokens=cfg.data.max_len,
        temperature=float(temperature if temperature is not None
                          else cfg.train.sample_temperature),
        top_k=int(top_k if top_k is not None else cfg.train.sample_top_k),
        cond=cond,
        batch_size=int(batch_size if batch_size is not None else cfg.data.eval_batch_size),
    )


def smiles_metrics(smiles: List[str], cfg) -> Dict[str, float]:
    """Validity / uniqueness / novelty computed on canonical SMILES."""
    from .chem import canonical_smiles, is_valid_smiles

    valid = [s for s in smiles if is_valid_smiles(s)]
    canon = [c for c in (canonical_smiles(s) for s in valid) if c]
    unique = set(canon)
    metrics = {
        "n_raw": float(len(smiles)),
        "n_valid": float(len(valid)),
        "validity": len(valid) / max(len(smiles), 1),
        "uniqueness": len(unique) / max(len(canon), 1),
    }
    try:
        from .metrics import collect_reference_smiles

        reference = set(collect_reference_smiles(cfg.data.root, "train"))
        metrics["novelty"] = (len([s for s in unique if s not in reference])
                              / max(len(unique), 1))
    except Exception:
        metrics["novelty"] = float("nan")
    return metrics


# --------------------------------------------------------------------------- #
# descriptor manifold
# --------------------------------------------------------------------------- #
def manifold(cfg, out_dir: Optional[str]):
    """Fit (once) and cache the descriptor PCA of the training split."""
    if not getattr(cfg.train, "analysis_pca", True):
        return None
    from molgen.preprocess.manifold import DescriptorManifold, fit_from_split

    path = os.path.join(out_dir, "manifold.npz") if out_dir else ""
    try:
        if path and os.path.exists(path):
            return DescriptorManifold.load(path)
        obj = fit_from_split(cfg.data.root,
                             limit=int(getattr(cfg.train, "analysis_pca_ref", 8000)))
        if path:
            obj.save(path)
        print(f"[analysis] descriptor manifold fitted on {cfg.data.root}/raw/train.smi")
        return obj
    except Exception as exc:
        print(f"[analysis] PCA disabled: {type(exc).__name__}: {exc}")
        return None


def analyse(cfg, model, tokenizer, device: torch.device, num: Optional[int] = None,
            out_dir: Optional[str] = None, with_pca: bool = True,
            temperature: Optional[float] = None, top_k: Optional[int] = None,
            cond: Optional[List[int]] = None,
            with_3d: bool = False) -> Tuple[Dict[str, float], List[str]]:
    """Sample strings, score them and (optionally) place them on the manifold."""
    from .chem import canonical_smiles, is_valid_smiles

    t0 = time.time()
    smiles = sample_strings(cfg, model, tokenizer, device, num=num,
                            temperature=temperature, top_k=top_k, cond=cond)
    metrics = smiles_metrics(smiles, cfg)

    canon = sorted({c for c in (canonical_smiles(s) for s in smiles
                                if is_valid_smiles(s)) if c})
    if with_pca:
        try:
            m = manifold(cfg, out_dir)
            if m is not None:
                metrics.update(m.project(canon))
        except Exception as exc:  # never let a diagnostic break the training run
            print(f"[analysis] manifold projection skipped: "
                  f"{type(exc).__name__}: {exc}")

    if with_3d and canon:
        try:
            from .conformer import smiles_list_to_3d

            samples, failed = smiles_list_to_3d(canon, seed=int(cfg.train.seed))
            if samples:
                from . import metrics as metrics_3d

                metrics.update({f"3d_{k}": v for k, v in
                                metrics_3d.evaluate(samples, reference_smiles=None).items()})
                print(f"[llm] embedded {len(samples)}/{len(canon)} molecules in 3D")
        except Exception as exc:
            print(f"[llm] 3D embedding skipped: {type(exc).__name__}: {exc}")

    metrics["eval_seconds"] = round(time.time() - t0, 1)
    return metrics, smiles


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #
def write_samples(cfg, smiles: List[str], path: str) -> None:
    """Write one SMILES per line."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(smiles) + "\n")


def append_log(path: str, record: Dict) -> None:
    """Append one JSON line (``eval_log.jsonl``)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
