"""Plot training loss against step for the three methods.

Reads the ``train_log.jsonl`` that every training run writes (one JSON record every
``train.log_every`` steps) and produces one figure with three curves plus the
validation points from ``metrics.jsonl``.

    python tools/plot_loss.py --log-root logs --out figures/loss_curves.pdf
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METHODS = ("geobfn", "edm", "llm")
COLOURS = {"geobfn": "#2f6f4f", "edm": "#2f4f8f", "llm": "#8f4f2f"}


def latest_run(log_root: str, method: str) -> str | None:
    """Newest run directory of ``method`` that contains a train_log.jsonl."""
    pattern = os.path.join(log_root, method, "*", "train_log.jsonl")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return os.path.dirname(files[-1]) if files else None


def read_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def curves(run_dir: str) -> Tuple[List[int], List[float], List[int], List[float]]:
    train = read_jsonl(os.path.join(run_dir, "train_log.jsonl"))
    valid = read_jsonl(os.path.join(run_dir, "metrics.jsonl"))
    return ([r["step"] for r in train], [r["loss"] for r in train],
            [r["step"] for r in valid], [r["valid_loss"] for r in valid])


def smooth(values: List[float], window: int = 5) -> List[float]:
    """Centred moving average, to make the three curves readable on one axis."""
    if window <= 1 or len(values) < window:
        return values
    out = []
    for i in range(len(values)):
        lo, hi = max(0, i - window // 2), min(len(values), i + window // 2 + 1)
        chunk = values[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log-root", default=os.environ.get("LOG_DIR", "./logs"))
    p.add_argument("--out", default="figures/loss_curves.pdf")
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--runs", nargs="*", default=None,
                   help="explicit run directories, in the order geobfn edm llm")
    args = p.parse_args()

    runs = args.runs or [latest_run(args.log_root, m) for m in METHODS]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
    plotted = 0
    for method, run_dir in zip(METHODS, runs):
        if not run_dir:
            print(f"[plot] no run for {method}, skipped")
            continue
        steps, losses, v_steps, v_losses = curves(run_dir)
        if not steps:
            print(f"[plot] empty train_log.jsonl in {run_dir}")
            continue
        ax.plot(steps, smooth(losses, args.window), color=COLOURS[method],
                lw=1.4, label=f"{method} (train)")
        if v_steps:
            ax.plot(v_steps, v_losses, "o--", color=COLOURS[method], ms=3, lw=1.0,
                    alpha=0.8, label=f"{method} (validation)")
        plotted += 1

    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.25, which="both")
    if plotted:
        ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print(f"[plot] wrote {args.out} ({plotted} methods)")


if __name__ == "__main__":
    main()
