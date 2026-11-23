"""Cross-validation driver: pick the hyper-parameters of one method.

    python cv.py method=geobfn +cv=geobfn
    python cv.py method=edm   +cv=edm   cv.folds=[0,1]

Each candidate configuration is trained once per fold, with a *small* budget
(``cv.budget.max_steps`` steps on ``cv.budget.limit`` training molecules, validated on
``cv.budget.valid_limit`` molecules of the held-out fold).  The candidate with the
lowest mean validation loss wins; the result is written to
``<LOG_DIR>/<method>/cv/cv_results.json`` and printed as a table, so the winning
overrides can be pasted into ``conf/method/<method>.yaml``.

The candidate grid lives in ``conf/cv/<method>.yaml``:

    cv:
      folds: [0, 1, 2]
      budget: {max_steps: 500, limit: 4000, valid_limit: 512}
      candidates:
        - {name: base, overrides: {}}
        - {name: lr_high, overrides: {train.lr: 0.0003}}
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Dict, List

import hydra
from omegaconf import DictConfig, OmegaConf

METHODS = ("geobfn", "edm", "llm")


def _run_dir(log_root: str, method: str, candidate: str, fold: int) -> str:
    return os.path.join(log_root, method, "cv", candidate, f"fold{fold}")


def _train_once(method: str, candidate: Dict, fold: int, budget: Dict,
                run_dir: str, extra: List[str], data_name: str = "qm9") -> Dict[str, float]:
    """Train one (candidate, fold) pair in a subprocess and read its final metrics."""
    cmd = [
        sys.executable, "train.py",
        f"method={method}",
        f"data.name={data_name}",
        f"data.fold={fold}",
        f"data.n_folds={budget['n_folds']}",
        f"data.limit={budget['limit']}",
        f"data.valid_limit={budget['valid_limit']}",
        f"data.valid_batches={budget['valid_batches']}",
        f"train.max_steps={budget['max_steps']}",
        f"train.lr={budget['lr']}",
        f"train.eval_every={budget['max_steps']}",
        f"train.ckpt_every={budget['max_steps']}",
        f"hydra.run.dir={run_dir}",
    ]
    # candidate overrides last, so they win over the CV defaults
    cmd += [f"{k}={v}" for k, v in (candidate.get("overrides") or {}).items()]
    cmd += extra
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    with open(os.path.join(run_dir, "final_metrics.json"), encoding="utf-8") as f:
        return json.load(f)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    method = str(cfg.name)
    if method not in METHODS:
        raise SystemExit(f"unknown method {method!r}")
    if "cv" not in cfg:
        raise SystemExit("pass a candidate grid, e.g. +cv=geobfn")

    cv = cfg.cv
    budget = {
        "max_steps": int(cv.budget.max_steps),
        "limit": int(cv.budget.limit),
        "valid_limit": int(cv.budget.valid_limit),
        "valid_batches": int(cv.budget.valid_batches),
        "n_folds": int(cv.get("n_folds", 3)),
        "lr": float(cv.budget.get("lr", cfg.train.lr)),
    }
    log_root = os.environ.get("LOG_DIR", "./logs")
    extra = [f"{k}={v}" for k, v in (cv.get("extra") or {}).items()]

    rows: List[Dict] = []
    for candidate in cv.candidates:
        name = str(candidate.name)
        losses = []
        for fold in cv.folds:
            metrics = _train_once(method, candidate, int(fold), budget,
                                  _run_dir(log_root, method, name, int(fold)), extra,
                                  data_name=str(cfg.data.name))
            loss = float(metrics.get("valid_loss", float("nan")))
            losses.append(loss)
            rows.append({"candidate": name, "fold": int(fold), "valid_loss": loss,
                         "overrides": OmegaConf.to_container(candidate.get("overrides") or {},
                                                             resolve=True)})
            print(f"[cv] {method} {name} fold {fold}: valid_loss={loss:.4f}")

    # aggregate: mean validation loss per candidate
    summary: Dict[str, Dict[str, float]] = {}
    for candidate in cv.candidates:
        name = str(candidate.name)
        vals = [r["valid_loss"] for r in rows if r["candidate"] == name]
        mean = sum(vals) / len(vals)
        summary[name] = {
            "mean_valid_loss": mean,
            "std_valid_loss": (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5,
            "overrides": OmegaConf.to_container(candidate.get("overrides") or {},
                                                resolve=True),
            "folds": [int(f) for f in cv.folds],
        }
    best = min(summary, key=lambda k: summary[k]["mean_valid_loss"])

    out_dir = os.path.join(log_root, method, "cv")
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "method": method,
        "budget": budget,
        "runs": rows,
        "summary": summary,
        "best": {"candidate": best, **summary[best]},
    }
    with open(os.path.join(out_dir, "cv_results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[cv] {method}: mean validation loss per candidate")
    for name, item in sorted(summary.items(), key=lambda kv: kv[1]["mean_valid_loss"]):
        mark = "  <-- best" if name == best else ""
        print(f"   {name:16s} {item['mean_valid_loss']:.4f} "
              f"+- {item['std_valid_loss']:.4f}{mark}")
    print(f"[cv] best candidate: {best}  overrides={summary[best]['overrides']}")
    print(f"[cv] results written to {os.path.join(out_dir, 'cv_results.json')}")
    print("[cv] paste the overrides into conf/method/"
          f"{method}.yaml (or pass them to the training script) and launch the "
          "10 000-step run.")


if __name__ == "__main__":
    main()
