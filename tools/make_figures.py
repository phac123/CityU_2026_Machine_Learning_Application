"""Collect the intermediate and final results of a run as images.

    python tools/make_figures.py --log-root logs --out figures

It reads what the training loop and the evaluation stages wrote and produces the
figures the report needs, as PNG (plus PDF for the paper):

  loss_curves.png   loss against step, three models, with the validation points
  metrics.png       the five chemical metrics of the three models, grouped bars
  pca_test.png      the test-set descriptor manifold (PCA) with, per model, the
                    generated samples projected onto it, and the distance histogram

Every panel is tolerant of a missing input (a model that was not trained yet): it is
skipped, and the figure is written with whatever is available.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

METHODS = ("geobfn", "edm", "llm")
COLOURS = {"geobfn": "#2f6f4f", "edm": "#2f4f8f", "llm": "#8f4f2f"}
METRICS = ("validity", "mol_stable", "atm_stable", "uniqueness", "novelty")


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def latest_run(log_root: str, method: str, pattern: str = "train_log.jsonl") -> Optional[str]:
    files = sorted(glob.glob(os.path.join(log_root, method, "*", pattern)),
                   key=os.path.getmtime)
    return os.path.dirname(files[-1]) if files else None


def read_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def smooth(values: List[float], window: int = 5) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(np.asarray(values, dtype=float), kernel, mode="same")


def load_metrics(run_dir: Optional[str]) -> Dict[str, float]:
    """Metrics of the largest evaluation available for a run."""
    if not run_dir:
        return {}
    candidates = [p for p in glob.glob(os.path.join(run_dir, "eval_*", "metrics.json"))
                  if os.path.exists(p)]
    best: Dict[str, float] = {}
    for path in candidates:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("n_valid", payload.get("n_raw", 0)) >= best.get("n_raw", -1):
            best = payload
    if not best:
        single = os.path.join(run_dir, "metrics.json")
        if os.path.exists(single):
            with open(single, encoding="utf-8") as f:
                best = json.load(f)
    return best


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #
def panel_loss(log_root: str, ax) -> int:
    drawn = 0
    for method in METHODS:
        run = latest_run(log_root, method)
        if not run:
            continue
        train = read_jsonl(os.path.join(run, "train_log.jsonl"))
        valid = read_jsonl(os.path.join(run, "metrics.jsonl"))
        if not train:
            continue
        ax.plot([r["step"] for r in train], smooth([r["loss"] for r in train]),
                color=COLOURS[method], lw=1.5, label=f"{method}")
        if valid:
            ax.plot([r["step"] for r in valid], [r["valid_loss"] for r in valid],
                    "o--", ms=3, lw=1.0, color=COLOURS[method], alpha=0.8)
        drawn += 1
    if drawn:
        ax.set_yscale("log")
        ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.25, which="both")
    return drawn


def panel_metrics(log_root: str, ax) -> int:
    width = 0.8 / len(METHODS)
    drawn = 0
    for i, method in enumerate(METHODS):
        metrics = load_metrics(latest_run(log_root, method))
        if not metrics:
            continue
        values = [float(metrics.get(k, np.nan)) for k in METRICS]
        xs = np.arange(len(METRICS)) + i * width
        ax.bar(xs, values, width=width, color=COLOURS[method], label=method)
        drawn += 1
    ax.set_xticks(np.arange(len(METRICS)) + width * (len(METHODS) - 1) / 2)
    ax.set_xticklabels([m.replace("_", "\n") for m in METRICS], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction")
    ax.grid(axis="y", alpha=0.25)
    if drawn:
        ax.legend(fontsize=8, frameon=False)
    return drawn


def panel_pca(log_root: str, ax, ax_hist) -> int:
    drawn = 0
    test_scores = None
    for method in METHODS:
        run = latest_run(log_root, method, "test_analysis.json")
        if not run:
            continue
        path = os.path.join(run, "test_scores.npz")
        if not os.path.exists(path):
            continue
        data = np.load(path)
        if test_scores is None:
            test_scores = data["test"]
            ax.scatter(test_scores[:, 0], test_scores[:, 1], s=3, c="#c8c8c8",
                       alpha=0.5, label="test split", rasterized=True)
        gen = data["generated"]
        if len(gen) == 0:
            continue
        ax.scatter(gen[:, 0], gen[:, 1], s=12, c=COLOURS[method], alpha=0.85,
                   label=method, edgecolors="none")
        ax_hist.hist(np.linalg.norm(gen, axis=1), bins=40, color=COLOURS[method],
                     alpha=0.55, label=method, density=True)
        drawn += 1

    if test_scores is not None:
        ax_hist.hist(np.linalg.norm(test_scores, axis=1), bins=40, color="#999999",
                     alpha=0.45, label="test split", density=True)
        ax_hist.set_xlabel("Mahalanobis distance (whitened PCA)")
        ax_hist.set_ylabel("density")
        ax_hist.legend(fontsize=8, frameon=False)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.legend(fontsize=8, frameon=False, loc="best")
        ax.grid(alpha=0.2)
    return drawn


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def save(fig, out_dir: str, name: str) -> None:
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=200 if ext == "png" else None)
    print(f"[figures] {name}.png / .pdf")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-root", default=os.environ.get("LOG_DIR", "./logs"))
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    if panel_loss(args.log_root, ax):
        fig.tight_layout()
        save(fig, args.out, "loss_curves")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    if panel_metrics(args.log_root, ax):
        fig.tight_layout()
        save(fig, args.out, "metrics")
    plt.close(fig)

    fig, (ax, ax_hist) = plt.subplots(1, 2, figsize=(10.0, 4.2),
                                      gridspec_kw={"width_ratios": [1.4, 1.0]})
    if panel_pca(args.log_root, ax, ax_hist):
        fig.tight_layout()
        save(fig, args.out, "pca_test")
    plt.close(fig)

    print(f"[figures] done -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()

