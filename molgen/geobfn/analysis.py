"""Sampling-based analysis, shared by the training loop and ``evaluate.py``.

Everything that needs a *sample* is computed here: the molecules, the five chemical
metrics of the report, and the position of those molecules on the descriptor manifold
of the training split (PCA, ``molgen.preprocess.manifold``).  The training loop calls
:func:`analyse` every ``train.analysis_every`` steps and once more at the last step,
so a finished run already contains its own evaluation; ``evaluate.py`` calls exactly
the same function, so the two can never disagree.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import torch

from .data import EvalDataset, collate, to_device


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample_molecules(cfg, model, device: torch.device, num: Optional[int] = None,
                     steps: Optional[int] = None,
                     seed: Optional[int] = None) -> List[dict]:
    """Draw ``num`` molecules with the current weights and split them apart."""
    model.eval()
    dataset = EvalDataset(int(num if num is not None else cfg.data.eval_num),
                          cfg.data.n_node_histogram,
                          seed=int(seed if seed is not None else cfg.train.seed))
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=cfg.data.eval_batch_size, shuffle=False,
                        collate_fn=collate)
    n_steps = int(steps if steps is not None else cfg.train.sample_steps)
    samples: List[dict] = []
    for batch in loader:
        batch = to_device(batch, device)
        out = model.sample(
            n_nodes=int(batch["pos"].shape[0]),
            edge_index=batch["edge_index"],
            seg=batch["batch"],
            steps=n_steps,
        )
        pos, z = out["pos"].cpu(), out["z"].cpu()
        offset = 0
        for n in batch["sizes"].tolist():
            samples.append({"pos": pos[offset:offset + n], "z": z[offset:offset + n]})
            offset += n
    return samples


# --------------------------------------------------------------------------- #
# descriptor manifold
# --------------------------------------------------------------------------- #
def manifold(cfg, out_dir: Optional[str]):
    """Fit (once) and cache the descriptor PCA of the training split.

    Returns ``None`` when PCA is disabled or when RDKit / the training SMILES are not
    available: the periodic analysis then simply reports the chemical metrics.
    """
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
    except Exception as exc:  # RDKit missing, no .smi, synthetic data, ...
        print(f"[analysis] PCA disabled: {type(exc).__name__}: {exc}")
        return None


def _smiles_of(samples: List[dict]) -> List[str]:
    """Canonical SMILES of 3D samples (same bond perception as the metrics)."""
    from . import metrics as metrics_lib

    out: List[str] = []
    for s in samples:
        mol, _, _ = metrics_lib.build_mol(s["pos"].float().cpu(), s["z"].cpu())
        smi = metrics_lib.mol_to_smiles(mol)
        if smi:
            out.append(smi)
    return out


# --------------------------------------------------------------------------- #
# the analysis itself
# --------------------------------------------------------------------------- #
def analyse(cfg, model, device: torch.device, num: Optional[int] = None,
            steps: Optional[int] = None, out_dir: Optional[str] = None,
            with_pca: bool = True,
            seed: Optional[int] = None) -> Tuple[Dict[str, float], List[dict]]:
    """Sample, score and (optionally) place the samples on the training manifold."""
    t0 = time.time()
    samples = sample_molecules(cfg, model, device, num=num, steps=steps, seed=seed)
    try:
        from . import metrics as metrics_lib

        reference = None
        try:
            reference = metrics_lib.collect_reference_smiles(cfg.data.root, "train")
        except Exception:
            pass
        metrics = metrics_lib.evaluate(samples, reference_smiles=reference)
    except Exception as exc:  # RDKit missing: keep the samples, report NaN
        print(f"[analysis] chemistry metrics unavailable: {type(exc).__name__}: {exc}")
        nan = float("nan")
        metrics = {"validity": nan, "mol_stable": nan, "atm_stable": nan,
                   "uniqueness": nan, "novelty": nan}

    if with_pca:
        try:
            m = manifold(cfg, out_dir)
            if m is not None:
                metrics.update(m.project(_smiles_of(samples)))
        except Exception as exc:  # never let a diagnostic break the training run
            print(f"[analysis] manifold projection skipped: "
                  f"{type(exc).__name__}: {exc}")

    metrics["eval_seconds"] = round(time.time() - t0, 1)
    return metrics, samples


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #
def write_samples(cfg, samples: List[dict], path: str) -> None:
    """Write the samples as a multi-frame XYZ file."""
    from .metrics import save_xyz

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    save_xyz(path, samples)


def append_log(path: str, record: Dict) -> None:
    """Append one JSON line (``eval_log.jsonl``)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
