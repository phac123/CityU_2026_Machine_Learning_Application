"""Hydra entry point for evaluating / sampling one of the three methods.

    python evaluate.py method=geobfn eval.ckpt=/workspace/logs/geobfn/.../best.pt
    python evaluate.py method=edm eval.ckpt=... eval.num=512 eval.steps=100 \
                       eval.out=/workspace/logs/edm/samples.xyz
    python evaluate.py method=llm eval.ckpt=... eval.conformers=true \
                       eval.xyz_out=/workspace/logs/llm/conformers.xyz

Evaluation is a single process on purpose: there is nothing to synchronise, and a
single-process sampler gives a reproducible sample set for a given seed.
"""

from __future__ import annotations

import importlib
from typing import Dict

import hydra
from omegaconf import DictConfig

EVALUATORS: Dict[str, str] = {
    "geobfn": "molgen.geobfn.evaluate",
    "edm": "molgen.edm.evaluate",
    "llm": "molgen.llm.evaluate",
}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Compose the config and call the branch's evaluation entry point."""
    name = str(cfg.name)
    if name not in EVALUATORS:
        raise SystemExit(f"unknown method {name!r}; choose one of {sorted(EVALUATORS)}")
    if not cfg.eval.get("ckpt"):
        raise SystemExit(
            "pass the checkpoint to evaluate, e.g.\n"
            "  python evaluate.py method=edm eval.ckpt=/workspace/logs/edm/<run>/best.pt"
        )
    evaluate = importlib.import_module(EVALUATORS[name]).evaluate
    evaluate(cfg)


if __name__ == "__main__":
    main()
