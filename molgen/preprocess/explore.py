"""Dimensionality reduction and clustering of the descriptor matrix.

This is the "feature exploration" step of the project.  Two questions are asked:

1. Does the descriptor space have low intrinsic dimension?  PCA answers it, and
   the explained-variance curve tells us how many components are worth keeping.
2. Do the molecules fall into chemically meaningful groups?  KMeans + silhouette
   answers it, and the per-cluster descriptor profile tells us *what* each group
   is (small polar fragments, aromatic rings, fluorinated, ...).

The reduced coordinates are also useful for a sanity check of the generative
models later: samples that fall outside the training manifold in this 2-D map are
the ones that the chemical metrics tend to reject.

Usage:
    python explore.py --data dataset/qm9/raw/train.smi --out runs/explore
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# OpenBLAS / OpenMP threads: set before numpy is imported.
#
# A many-core node (8 GPUs usually means 64-128 cores) makes OpenBLAS default to
# nproc threads, which can exceed the thread metadata the wheel was built with:
# the result is "Bad memory unallocation" followed by a double free and a core
# dump inside KMeans / silhouette.  Capping the pool is the documented fix.
# Values from the environment (docker-compose sets OMP_NUM_THREADS) win.
# --------------------------------------------------------------------------- #
for _threads_var, _threads_value in (("OPENBLAS_NUM_THREADS", "8"),
                                     ("OMP_NUM_THREADS", "8"),
                                     ("MKL_NUM_THREADS", "8"),
                                     ("NUMEXPR_NUM_THREADS", "8")):
    os.environ.setdefault(_threads_var, _threads_value)

import numpy as np

from .features import DESCRIPTOR_NAMES, featurize, standardize


# --------------------------------------------------------------------------- #
# PCA (numpy only - no scikit-learn needed)
# --------------------------------------------------------------------------- #
def pca_fit(x: np.ndarray, n_components: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA through the SVD of the centred matrix.

    Args:
        x: [N, D] standardised features.
        n_components: number of components to keep (default: all).
    Returns:
        scores [N, k], components [k, D], explained_variance_ratio [k]
    """
    xc = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    k = n_components or min(xc.shape)
    scores = u[:, :k] * s[:k]
    components = vt[:k]
    var = (s ** 2) / max(len(xc) - 1, 1)
    ratio = var[:k] / var.sum()
    return scores, components, ratio


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #
def kmeans(x: np.ndarray, k: int, seed: int = 0, n_init: int = 10):
    """KMeans from scikit-learn (kept optional: it is the only sklearn use here)."""
    try:
        from sklearn.cluster import KMeans
    except Exception as exc:  # pragma: no cover
        raise ImportError("scikit-learn is required for KMeans / silhouette") from exc
    model = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(x)
    return model.labels_, model.cluster_centers_


def silhouette(x: np.ndarray, labels: np.ndarray, sample_size: int = 5000,
               seed: int = 0) -> float:
    """Mean silhouette coefficient of a labelling (computed on a subsample).

    The full pairwise matrix of 107 108 points would be 10^10 entries; the
    silhouette is an estimate anyway, so a seeded subsample is used.
    """
    try:
        from sklearn.metrics import silhouette_score
    except Exception as exc:  # pragma: no cover
        raise ImportError("scikit-learn is required for silhouette") from exc
    if len(set(labels.tolist())) < 2:
        return -1.0
    sample = int(min(sample_size, len(x)))
    return float(silhouette_score(x, labels, sample_size=sample, random_state=seed))


def choose_k(x: np.ndarray, k_min: int = 2, k_max: int = 10, seed: int = 0,
             sample_size: int = 5000) -> Tuple[int, Dict[int, float]]:
    """Pick the number of clusters that maximises the silhouette coefficient."""
    scores: Dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        labels, _ = kmeans(x, k, seed=seed)
        scores[k] = silhouette(x, labels, sample_size=sample_size, seed=seed)
    best = max(scores, key=scores.get)
    return int(best), scores


def assign_labels(x: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Nearest-centre label for every row (the model is fitted on a subsample)."""
    labels = np.zeros(len(x), dtype=np.int64)
    best = np.full(len(x), np.inf)
    for i, centre in enumerate(centres):
        d = ((x - centre) ** 2).sum(axis=1)
        closer = d < best
        labels[closer] = i
        best = np.where(closer, d, best)
    return labels


def cluster_profile(x: np.ndarray, labels: np.ndarray, names: Sequence[str] = DESCRIPTOR_NAMES) -> Dict[int, Dict[str, float]]:
    """Mean descriptor value per cluster (in standardised units)."""
    profile: Dict[int, Dict[str, float]] = {}
    for c in sorted(set(labels.tolist())):
        mask = labels == c
        profile[int(c)] = {n: float(v) for n, v in zip(names, x[mask].mean(axis=0))}
    return profile


# --------------------------------------------------------------------------- #
# optional 2-D projection for plotting
# --------------------------------------------------------------------------- #
def tsne_2d(x: np.ndarray, perplexity: float = 30.0, seed: int = 0,
            max_points: int = 5000) -> np.ndarray:
    """t-SNE embedding for visualisation only (not used by any model).

    t-SNE is O(n^2) in time and memory, so it is always computed on a seeded
    subsample of at most ``max_points`` rows, whatever the size of the split.
    """
    if len(x) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x), size=max_points, replace=False)
        x = x[np.sort(idx)]
    try:
        from sklearn.manifold import TSNE
    except Exception as exc:  # pragma: no cover
        raise ImportError("scikit-learn is required for t-SNE") from exc
    return TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                init="pca").fit_transform(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Descriptor PCA / clustering exploration")
    p.add_argument("--data", type=str, required=True, help="SMILES file or .npz")
    p.add_argument("--out", type=str, default="runs/explore")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--k-max", type=int, default=10)
    p.add_argument("--tsne", action="store_true", help="also compute a t-SNE map")
    p.add_argument("--tsne-max", type=int, default=5000,
                   help="points used for the t-SNE map (it is O(n^2))")
    p.add_argument("--cluster-max", type=int, default=20000,
                   help="points used for the KMeans / silhouette search")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    smiles = []
    if args.data.endswith(".npz"):
        raise SystemExit("pass a SMILES file: use build_dataset.py to export .smi "
                         "from the npz artefacts first")
    with open(args.data, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                smiles.append(line.split()[0])
    if args.limit:
        smiles = smiles[: args.limit]

    x_raw, kept = featurize(smiles, verbose=True)
    if len(kept) == 0:
        raise SystemExit("no molecule could be featurised")
    x = standardize(x_raw)

    scores, components, ratio = pca_fit(x)
    n_90 = int(np.searchsorted(np.cumsum(ratio), 0.90) + 1)
    print("[pca] explained variance (first 6):", np.round(ratio[:6], 4))
    print(f"[pca] components needed for 90% variance: {n_90}")

    rng = np.random.default_rng(args.seed)
    if len(x) > args.cluster_max:
        sub = np.sort(rng.choice(len(x), size=args.cluster_max, replace=False))
    else:
        sub = np.arange(len(x))
    xs = x[sub]
    best_k, sil = choose_k(xs, 2, args.k_max, seed=args.seed)
    _, centres = kmeans(xs, best_k, seed=args.seed)
    labels = assign_labels(x, centres)
    print(f"[kmeans] best k={best_k} (silhouette={sil[best_k]:.3f}) "
          f"fitted on {len(xs)} molecules, labelled {len(x)}")
    print("[kmeans] silhouette per k:", {k: round(v, 3) for k, v in sil.items()})

    profile = cluster_profile(x, labels)
    summary = {
        "n_molecules": len(kept),
        "n_descriptors": len(DESCRIPTOR_NAMES),
        "pca_explained_variance_ratio": [float(v) for v in ratio],
        "pca_components_for_90pct": n_90,
        "silhouette_by_k": {int(k): float(v) for k, v in sil.items()},
        "best_k": best_k,
        "cluster_profile_z": {str(k): v for k, v in profile.items()},
        "descriptor_means_raw": {n: float(v) for n, v in zip(DESCRIPTOR_NAMES, x_raw.mean(axis=0))},
    }

    if args.tsne:
        n_tsne = int(min(args.tsne_max, len(x)))
        idx = np.sort(rng.choice(len(x), size=n_tsne, replace=False))
        try:
            emb = tsne_2d(scores[idx, : min(10, scores.shape[1])], seed=args.seed,
                          max_points=n_tsne)
            summary["tsne_shape"] = list(emb.shape)
            np.savez(os.path.join(args.out, "tsne.npz"), emb=emb, labels=labels[idx],
                     index=idx)
            print(f"[tsne] wrote {args.out}/tsne.npz ({n_tsne} molecules)")
        except Exception as exc:  # a failed embedding must not lose the analysis
            print(f"[tsne] skipped: {type(exc).__name__}: {exc}")

    np.savez(os.path.join(args.out, "pca.npz"), scores=scores[:, :10],
             components=components[:10], ratio=ratio[:10], labels=labels)
    with open(os.path.join(args.out, "explore_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[done] wrote", os.path.join(args.out, "explore_summary.json"))


if __name__ == "__main__":
    main()
