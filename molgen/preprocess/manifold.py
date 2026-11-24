"""Descriptor manifold: a PCA of the training split, used to place generated molecules.

The exploratory study of the report (§4.2) and the analysis inside the training loop
must use the *same* descriptors, the same standardisation and the same PCA, otherwise
the two numbers cannot be compared.  This module is that single definition:

* :meth:`DescriptorManifold.fit` fits it once on a SMILES file (typically
  ``raw/train.smi``) and caches it in the run directory (``manifold.npz``);
* :meth:`DescriptorManifold.project` maps an arbitrary set of SMILES onto it and
  returns three numbers --- how far the samples lie from the training cloud
  (Mahalanobis distance relative to the training median), how many of them fall
  outside the 95 % region of that cloud, and how far they are from their nearest
  reference molecule (a coverage / memorisation proxy).

Everything is numpy + scikit-learn, so it runs on the CPU while the GPU trains.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .features import DESCRIPTOR_NAMES, featurize


class DescriptorManifold:
    """Standardised descriptor space, its PCA basis and the reference scores."""

    def __init__(self, mu: np.ndarray, sd: np.ndarray, components: np.ndarray,
                 explained: np.ndarray, reference: np.ndarray):
        self.mu = mu                      # [1, D] descriptor mean
        self.sd = sd                      # [1, D] descriptor std
        self.components = components      # [K, D] PCA axes
        self.explained = explained        # [K] variance ratio
        self.reference = reference        # [R, K] whitened reference scores
        self._scale: Optional[np.ndarray] = None   # whitening scale (set by fit/load)
        self.ref_norm = np.linalg.norm(reference, axis=1)

    # ------------------------------------------------------------------ fit --
    @classmethod
    def fit(cls, smiles: Sequence[str], n_components: int = 10,
            reference_limit: int = 4000, seed: int = 0) -> "DescriptorManifold":
        """Fit on a SMILES corpus (training split) and keep a reference subsample."""
        from sklearn.decomposition import PCA

        x, kept = featurize(smiles)
        if len(kept) < n_components + 10:
            raise ValueError(f"only {len(kept)} molecules could be featurised")
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-12, 1.0, sd)
        z = (x - mu) / sd

        k = int(min(n_components, z.shape[1], max(2, z.shape[0] - 1)))
        pca = PCA(n_components=k, random_state=seed).fit(z)
        scale = np.sqrt(np.maximum(pca.explained_variance_, 1e-12))
        scores = pca.transform(z) / scale                     # whitened
        rng = np.random.default_rng(seed)
        take = min(reference_limit, len(scores))
        idx = rng.choice(len(scores), size=take, replace=False)

        obj = cls(mu, sd, pca.components_, pca.explained_variance_ratio_, scores[idx])
        obj._scale = scale
        return obj

    def scores(self, smiles: Sequence[str], n_components: Optional[int] = None
               ) -> np.ndarray:
        """Whitened PCA coordinates of ``smiles`` ([n, k]) for plotting.

        Rows that RDKit cannot featurise are dropped, so the returned array can be
        shorter than the input; ``project`` is the statistics-only counterpart.
        """
        x, kept = featurize(smiles)
        k = self.components.shape[0] if n_components is None else int(n_components)
        if len(kept) == 0:
            return np.zeros((0, k))
        scale = self._scale if self._scale is not None else np.ones(k)
        z = (x - self.mu) / self.sd
        return ((z @ self.components.T) / scale)[:, :k]

    # -------------------------------------------------------------- project --
    def project(self, smiles: Sequence[str]) -> Dict[str, float]:
        """Distance statistics of ``smiles`` with respect to the training manifold."""
        scale = self._scale if self._scale is not None else np.ones(
            self.components.shape[0])
        x, kept = featurize(smiles)
        if len(kept) == 0:
            nan = float("nan")
            return {"pca_n": 0.0, "pca_outside": nan, "pca_mahal_ratio": nan,
                    "pca_nn_mean": nan, "pca_var12": float(np.sum(self.explained[:2]))}

        z = (x - self.mu) / self.sd
        w = (z @ self.components.T) / scale
        norm = np.linalg.norm(w, axis=1)
        # nearest reference molecule, in the whitened PCA space
        diff = w[:, None, :] - self.reference[None, :, :]
        nn = np.sqrt((diff ** 2).sum(axis=2)).min(axis=1)

        d95 = float(np.percentile(self.ref_norm, 95))
        median_ref = float(np.median(self.ref_norm)) or 1.0
        return {
            "pca_n": float(len(kept)),
            "pca_outside": float((norm > d95).mean()),
            "pca_mahal_ratio": float(np.median(norm) / median_ref),
            "pca_nn_mean": float(nn.mean()),
            "pca_var12": float(np.sum(self.explained[:2])),
        }

    # ------------------------------------------------------------------ io ---
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        scale = self._scale if self._scale is not None else np.ones(
            self.components.shape[0])
        np.savez(path, mu=self.mu, sd=self.sd, components=self.components,
                 explained=self.explained, reference=self.reference, scale=scale)

    @classmethod
    def load(cls, path: str) -> "DescriptorManifold":
        data = np.load(path)
        obj = cls(data["mu"], data["sd"], data["components"], data["explained"],
                  data["reference"])
        obj._scale = data["scale"]
        return obj


def fit_from_split(root: str, split: str = "train", limit: int = 8000,
                   reference_limit: int = 4000, seed: int = 0) -> DescriptorManifold:
    """Convenience: fit the manifold from ``<root>/raw/<split>.smi``."""
    path = os.path.join(root, "raw", f"{split}.smi")
    with open(path, encoding="utf-8", errors="ignore") as f:
        smiles: List[str] = [line.split()[0] for line in f
                             if line.strip() and not line.startswith("#")]
    if limit and len(smiles) > limit:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(smiles), size=limit, replace=False)
        smiles = [smiles[i] for i in sorted(idx)]
    return DescriptorManifold.fit(smiles, reference_limit=reference_limit, seed=seed)


if __name__ == "__main__":                                   # tiny self-check
    import argparse

    p = argparse.ArgumentParser("Fit / inspect the descriptor manifold")
    p.add_argument("--data", required=True, help="SMILES file (training split)")
    p.add_argument("--limit", type=int, default=8000)
    p.add_argument("--out", default=None, help="optional manifold.npz")
    args = p.parse_args()
    m = fit_from_split(os.path.dirname(os.path.dirname(args.data)), limit=args.limit)
    print("descriptors:", DESCRIPTOR_NAMES)
    print("explained variance (PC1..PC3):", np.round(m.explained[:3], 4))
    print("reference distance: median %.3f, p95 %.3f"
          % (np.median(m.ref_norm), np.percentile(m.ref_norm, 95)))
    if args.out:
        m.save(args.out)
        print("wrote", args.out)
