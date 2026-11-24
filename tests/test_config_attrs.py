"""Every ``cfg.<section>.<attr>`` access must exist in that branch's dataclass.

This is the check that catches the crash

    AttributeError: 'TrainConfig' object has no attribute 'sample_steps'

which came from ``getattr(cfg.train, "analysis_steps", cfg.train.sample_steps)``: the
default argument is evaluated eagerly, so a fallback that the branch does not have
blows up even when the primary key exists.  Walking the AST finds *all* such accesses,
including ones hidden in defaults.

Run it with no arguments; it reads only the sources.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

METHODS = ("geobfn", "edm", "llm")
FILES = ("train.py", "analysis.py", "evaluate.py", "train_utils.py",
         "bfn.py", "diffusion.py", "egnn.py", "model.py", "data.py",
         "metrics.py", "sampler.py", "conformer.py", "dataset.py")
#: keys of the global config that are not method dataclasses
GLOBAL_KEYS = {"name", "seed", "smoke", "ddp", "eval", "cv", "budget", "hydra"}


def load_config_module(method: str):
    path = os.path.join(PROJECT_ROOT, "molgen", method, "config.py")
    name = f"_cfgattrs_{method}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)          # type: ignore[union-attr]
    return module


def sections(module) -> dict:
    """``{"train": {"lr", ...}, "data": {...}, ...}`` from the dataclasses."""
    out = {}
    for attr in dir(module):
        obj = getattr(module, attr)
        if dataclasses.is_dataclass(obj) and isinstance(obj, type):
            out[attr.lower().replace("config", "")] = {f.name for f in dataclasses.fields(obj)}
    return {k: v for k, v in out.items() if v}


def main() -> None:
    problems: list[str] = []
    for method in METHODS:
        module = load_config_module(method)
        known = sections(module)
        print(f"{method}: dataclasses {sorted(known)}")

        for filename in FILES:
            path = os.path.join(PROJECT_ROOT, "molgen", method, filename)
            if not os.path.exists(path):
                continue
            tree = ast.parse(open(path, encoding="utf-8").read())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Attribute):
                    continue
                outer = node
                middle = node.value
                if not isinstance(middle, ast.Attribute):
                    continue
                base = middle.value
                if not (isinstance(base, ast.Name) and base.id == "cfg"):
                    continue
                section, attr = middle.attr, outer.attr
                if section in GLOBAL_KEYS:
                    continue
                if section not in known:
                    problems.append(f"{method}/{filename}:{outer.lineno}: "
                                    f"cfg.{section}.{attr} -- '{section}' is not a "
                                    f"dataclass of this branch")
                elif attr not in known[section]:
                    problems.append(f"{method}/{filename}:{outer.lineno}: "
                                    f"cfg.{section}.{attr} -- '{attr}' is not a field of "
                                    f"{section} ({sorted(known[section])})")

    if problems:
        print("\nUNKNOWN CONFIG ATTRIBUTES:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("\nCONFIG ATTRIBUTE CHECKS PASSED")


if __name__ == "__main__":
    main()

