"""Print the winning cross-validation configuration as Hydra overrides.

    python tools/cv_best.py logs/geobfn/cv/cv_results.json
    # -> train.lr=0.0003
    python tools/cv_best.py logs/geobfn/cv/cv_results.json --json
    # -> {"candidate": "lr_high", "mean_valid_loss": 0.123, "overrides": {...}, ...}

Used by ``run_experiments.sh`` to feed the cross-validation winner into the formal
training run, so nobody has to copy values by hand.
"""
from __future__ import annotations

import argparse
import json
from typing import Any


def format_value(value: Any) -> str:
    """Render a YAML/JSON scalar the way Hydra expects it on the command line."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="path to cv_results.json")
    p.add_argument("--json", action="store_true", help="dump the winning entry as JSON")
    args = p.parse_args()

    with open(args.path, encoding="utf-8") as f:
        payload = json.load(f)
    best = payload.get("best") or {}
    if args.json:
        print(json.dumps(best, indent=2))
        return
    overrides = best.get("overrides") or {}
    tokens = [f"{key}={format_value(value)}" for key, value in overrides.items()]
    print(" ".join(tokens))


if __name__ == "__main__":
    main()
