"""Test the trained models on the *test* split: sample, score, and locate the samples.

    python tools/test_generation.py --method edm --ckpt runs/edm/<ts>/last.pt \
        --num 256 --out runs/edm/<ts>/test_eval

It runs ``evaluate.py`` for the checkpoint (so the sampling code is exactly the one the
report describes), then adds the part that is specific to this tool:

* the five chemical metrics (written by ``evaluate.py`` into ``metrics.json``);
* the position of the samples on the descriptor manifold fitted on the **test** split,
  either relative to the training cloud or to the test cloud (``--manifold``);
* a coverage / memorisation pair: how many samples are closer to the test set than the
  typical test molecule is to its neighbour, and how far the nearest test molecule is.

All numbers go to ``test_analysis.json`` next to the samples, and the whitened PCA
coordinates to ``test_scores.npz``; ``tools/make_figures.py`` turns them into the
figure of the report.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from molgen.preprocess.manifold import DescriptorManifold, fit_from_split  # noqa: E402

METHODS = ("geobfn", "edm", "llm")


# --------------------------------------------------------------------------- #
# sampling (delegated to evaluate.py, so there is one sampling path)
# --------------------------------------------------------------------------- #
def sample(cfg_root: str, method: str, ckpt: str, num: int, out_dir: str,
           seed: int, extra: List[str]) -> Tuple[str, str]:
    """Sample ``num`` molecules/strings and return (samples path, metrics path)."""
    ext = "smi" if method == "llm" else "xyz"
    samples = os.path.join(out_dir, f"samples.{ext}")
    metrics = os.path.join(out_dir, "metrics.json")
    command = [
        sys.executable, "evaluate.py",
        f"method={method}",
        f"eval.ckpt={ckpt}",
        f"eval.num={num}",
        "eval.use_ema=false",
        f"eval.seed={seed}",
        f"eval.out={samples}",
        f"eval.json_out={metrics}",
        *extra,
    ]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cfg_root, check=True)
    return samples, metrics


# --------------------------------------------------------------------------- #
# analysis on the test manifold
# --------------------------------------------------------------------------- #
def smiles_of(method: str, samples_path: str) -> List[str]:
    """SMILES of the generated samples (3D -> SMILES for the geometric branches)."""
    if samples_path.endswith(".smi"):
        with open(samples_path, encoding="utf-8") as f:
            return [line.split()[0] for line in f if line.strip()]

    import torch                     # only needed for the 3D -> SMILES path

    analysis = importlib.import_module(f"molgen.{method}.analysis")
    frames = []
    with open(samples_path, encoding="utf-8", errors="ignore") as f:
        while True:
            header = f.readline()
            if not header:
                break
            if not header.strip():
                continue
            n = int(header.strip())
            f.readline()
            z, pos = [], []
            for _ in range(n):
                parts = f.readline().split()
                z.append(int(parts[0]))
                pos.append([float(v) for v in parts[1:4]])
            frames.append({"z": torch.tensor(z), "pos": torch.tensor(pos)})
    return [s for s in analysis._smiles_of(frames) if s]  # noqa: SLF001


def test_manifold(cfg_root: str, cache: str, ref_limit: int = 8000) -> DescriptorManifold:
    """Fit (once) the descriptor manifold of the test split and cache it."""
    if os.path.exists(cache):
        return DescriptorManifold.load(cache)
    data_root = os.path.join(cfg_root, "data", "qm9")
    env_root = os.environ.get("DATA_DIR")
    if env_root:
        data_root = os.path.join(env_root, "qm9")
    obj = fit_from_split(data_root, split="test", limit=ref_limit,
                         reference_limit=min(ref_limit, 4000))
    obj.save(cache)
    return obj


def coverage(manifold: DescriptorManifold, generated: np.ndarray,
             test_scores: np.ndarray) -> Dict[str, float]:
    """How far the samples sit from the test cloud, in whitened PCA space."""
    if len(generated) == 0 or len(test_scores) == 0:
        nan = float("nan")
        return {"cov_n": 0.0, "cov_nn_median": nan, "cov_inside": nan}

    test_norm = np.linalg.norm(test_scores, axis=1)
    # distance from each test molecule to its nearest test neighbour (the baseline)
    sample = test_scores[: min(2000, len(test_scores))]
    d_test = np.sqrt(((sample[:, None, :] - test_scores[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d_test[:, : len(sample)], np.inf)
    baseline = float(np.median(d_test.min(axis=1)))

    d_gen = np.sqrt(((generated[:, None, :] - test_scores[None, :, :]) ** 2).sum(-1))
    nearest = d_gen.min(axis=1)
    return {
        "cov_n": float(len(generated)),
        "cov_nn_median": float(np.median(nearest)),
        "cov_baseline_median": baseline,
        "cov_inside": float((nearest <= baseline).mean()),
        "cov_test_norm_median": float(np.median(test_norm)),
    }


def build_result(method: str, ckpt: str, requested: int, smiles: list,
                 metrics: dict, manifold_stats: dict, coverage_stats: dict) -> dict:
    """Assemble ``test_analysis.json``: flat PCA keys, nested copies for clarity."""
    # DescriptorManifold.project() already returns pca_* keys; prefix only the raw ones
    pca = {(k if k.startswith("pca_") else f"pca_{k}"): v
           for k, v in manifold_stats.items()}
    return {
        "method": method,
        "ckpt": os.path.abspath(ckpt),
        "num_requested": int(requested),
        "num_samples": len(smiles),
        "metrics": metrics,
        # flat, so tools/make_figures.py and the log line can read them directly
        **pca,
        **coverage_stats,
        # and grouped, for a human reading the JSON
        "manifold": pca,
        "coverage": coverage_stats,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--num", type=int, default=256)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--manifold", default=None,
                   help="cached test manifold (.npz); shared by the three methods")
    p.add_argument("--no-sample", action="store_true",
                   help="analyse existing samples.<ext> instead of sampling again")
    p.add_argument("extra", nargs="*", help="extra Hydra overrides for evaluate.py")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg_root = PROJECT_ROOT
    manifold_path = args.manifold or os.path.join(args.out, "..", "test_manifold.npz")
    manifold_path = os.path.abspath(manifold_path)

    if args.no_sample:
        ext = "smi" if args.method == "llm" else "xyz"
        samples_path = os.path.join(args.out, f"samples.{ext}")
        metrics_path = os.path.join(args.out, "metrics.json")
    else:
        samples_path, metrics_path = sample(cfg_root, args.method, args.ckpt,
                                            args.num, args.out, args.seed, args.extra)

    smiles = smiles_of(args.method, samples_path)
    manifold = test_manifold(cfg_root, manifold_path)
    generated = manifold.scores(smiles)
    stats = manifold.project(smiles)

    # the test cloud itself, for the coverage baseline
    test_smiles_path = os.path.join(os.environ.get("DATA_DIR", os.path.join(cfg_root, "data")),
                                    "qm9", "raw", "test.smi")
    with open(test_smiles_path, encoding="utf-8") as f:
        test_smiles = [line.split()[0] for line in f
                       if line.strip() and not line.startswith("#")][:8000]
    test_scores = manifold.scores(test_smiles)

    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    result = build_result(method=args.method, ckpt=args.ckpt, requested=args.num,
                          smiles=smiles, metrics=metrics, manifold_stats=stats,
                          coverage_stats=coverage(manifold, generated, test_scores))
    with open(os.path.join(args.out, "test_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    np.savez(os.path.join(args.out, "test_scores.npz"), generated=generated,
             test=test_scores)
    def _fmt(value):
        value = float(value) if value is not None else float("nan")
        return "n/a" if value != value else f"{value:.3f}"

    print(f"[test] {args.method}: validity={_fmt(metrics.get('validity'))} "
          f"samples={len(smiles)} pca_outside={_fmt(result.get('pca_outside'))} "
          f"inside_test={_fmt(result.get('cov_inside'))}")
    print(f"[test] wrote {os.path.join(args.out, 'test_analysis.json')}")


if __name__ == "__main__":
    main()
