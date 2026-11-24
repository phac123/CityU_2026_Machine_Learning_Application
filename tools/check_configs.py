"""Check (and optionally repair) the Hydra configuration files.

    python tools/check_configs.py            # validate, exit 1 on any problem
    python tools/check_configs.py --fix      # normalise the files, then validate
    python tools/check_configs.py --verbose  # also print the composed root keys

What it verifies, in the order in which the failures actually happen:

1. no BOM and no CRLF in ``conf/**/*.yaml`` -- a BOM in front of the first line makes
   Hydra stop scanning for ``# @package``, so the group file lands under ``method.*``
   and every ``train.*`` override dies with "Key 'train' is not in struct";
2. ``# @package _global_`` is the **first** line of every ``conf/method/*.yaml``;
3. each method file defines ``data``, ``model`` and ``train`` (plus ``diffusion`` for
   EDM) and only uses keys the dataclasses declare;
4. when Hydra is importable, every ``method=<m>`` really composes a root config that
   contains those blocks -- the definitive check.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:            # running by path puts tools/ on sys.path
    sys.path.insert(0, PROJECT_ROOT)
CONF = os.path.join(PROJECT_ROOT, "conf")
METHODS = ("geobfn", "edm", "llm")
BOM = "\ufeff"
DIRECTIVE = "# @package _global_"


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def normalise(path: str) -> bool:
    """Strip BOM/CRLF and put the package directive on line 1; True if changed."""
    raw = open(path, "rb").read()
    text = raw.decode("utf-8-sig")                      # drops a UTF-8 BOM
    crlf = b"\r\n" in raw
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    if path.endswith(os.path.join("method", "llm.yaml")) or "/method/" in path.replace("\\", "/"):
        lines = [ln for ln in lines if ln.strip() != DIRECTIVE]
        while lines and not lines[0].strip():
            lines.pop(0)
        lines = [DIRECTIVE, ""] + lines

    new_text = "\n".join(lines).rstrip("\n") + "\n"
    changed = crlf or new_text != text or text != text.lstrip(BOM)
    if changed:
        io.open(path, "w", encoding="utf-8", newline="\n").write(new_text)
    return changed


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def check_files(fix: bool) -> list:
    problems = []
    for root, _dirs, files in os.walk(CONF):
        for name in sorted(files):
            if not name.endswith((".yaml", ".yml")):
                continue
            path = os.path.join(root, name)
            raw = open(path, "rb").read()
            if raw[:3] == b"\xef\xbb\xbf":
                problems.append(f"{path}: has a UTF-8 BOM (Hydra will ignore # @package)")
            if b"\r\n" in raw:
                problems.append(f"{path}: CRLF line endings")
            if fix:
                normalise(path)
    return problems


def check_method_files() -> list:
    problems = []
    for method in METHODS:
        path = os.path.join(CONF, "method", f"{method}.yaml")
        if not os.path.exists(path):
            problems.append(f"{path}: missing")
            continue
        lines = io.open(path, encoding="utf-8-sig").read().replace("\r\n", "\n").split("\n")
        if not lines or lines[0].strip() != DIRECTIVE:
            problems.append(f"{path}: first line must be exactly {DIRECTIVE!r}, "
                            f"found {lines[0][:40]!r}")
        text = "\n".join(lines)
        for block, why in (("data:", "data block"), ("model:", "model block"),
                           ("train:", "train block")):
            if not re.search(rf"^{block}", text, re.M):
                problems.append(f"{path}: no {why}")
        if method == "llm" and "ema_decay:" in text:
            problems.append(f"{path}: the language model has no EMA; drop ema_decay")
    return problems


def check_keys() -> list:
    """Every key in a method file must exist in that method's dataclasses."""
    problems = []
    try:
        import yaml
    except ImportError:
        return problems                                  # pyyaml absent: skip silently

    for method in METHODS:
        path = os.path.join(CONF, "method", f"{method}.yaml")
        if not os.path.exists(path):
            continue
        with io.open(path, encoding="utf-8-sig") as f:
            cfg = yaml.safe_load(f) or {}
        src = io.open(os.path.join(PROJECT_ROOT, "molgen", method, "config.py"),
                      encoding="utf-8").read()
        for section, values in cfg.items():
            if not isinstance(values, dict):
                continue
            for key in values:
                if not re.search(rf"^\s{{4}}{key}:", src, re.M):
                    problems.append(f"{method}: unknown field {section}.{key} "
                                    "(not in config.py)")
    return problems


def _probe(cfg):
    """A copy of the composed config with the Hydra-runtime interpolation replaced.

    ``train.out_dir`` is ``${hydra:runtime.output_dir}``; it only resolves inside a
    running Hydra job, so the probe uses a literal path instead.  Everything else
    (notably ``${oc.env:...}``) is resolved as usual by ``from_hydra``.
    """
    from omegaconf import OmegaConf

    container = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(container, dict):
        return cfg
    train = container.get("train")
    if isinstance(train, dict):
        train["out_dir"] = "/tmp/molgen-config-check"
    return OmegaConf.create(container)


def check_composition(verbose: bool) -> list:
    """Ask Hydra itself: does ``method=<m>`` produce data/model/train at the root?"""
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
    except ImportError:
        print("[check] hydra not importable here: composition check skipped")
        return []

    problems = []
    for method in METHODS:
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONF, version_base=None):
            cfg = compose(config_name="config", overrides=[f"method={method}"])
        keys = list(cfg.keys())
        if verbose:
            print(f"[check] method={method}: root keys {keys}")
        for block in ("data", "model", "train"):
            if block not in cfg:
                problems.append(
                    f"method={method}: '{block}' is not at the root of the composed "
                    "config (the # @package _global_ header of the method file is "
                    "not taking effect)")
        if method == "edm" and "diffusion" not in cfg:
            problems.append("method=edm: no 'diffusion' block")
        if str(cfg.get("name")) != method:
            problems.append(f"method={method}: cfg.name is {cfg.get('name')!r}")

        # the adapter the training loop actually calls: it must accept the composed
        # global config (name/seed/smoke/ddp/eval/cv) and only read data/model/train
        try:
            import importlib

            module = importlib.import_module(f"molgen.{method}.config")
            module.from_hydra(_probe(cfg))
        except ImportError as exc:          # torch/rdkit absent: not a config problem
            print(f"[check] {method}: from_hydra skipped ({exc})")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"method={method}: from_hydra failed -> "
                            f"{type(exc).__name__}: {exc}")
    return problems


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fix", action="store_true", help="normalise the files first")
    p.add_argument("--verbose", action="store_true", help="print the composed keys")
    args = p.parse_args()

    problems = check_files(args.fix)
    if args.fix and problems:
        problems = check_files(False)                    # re-check after repairing
    problems += check_method_files()
    problems += check_keys()
    problems += check_composition(args.verbose)

    if problems:
        print("CONFIG PROBLEMS:")
        for problem in problems:
            print("  -", problem)
        normalisable = [p for p in problems
                        if "BOM" in p or "CRLF" in p or "first line must be" in p]
        if normalisable:
            print("\nRun `python tools/check_configs.py --fix` to normalise the YAML files.")
        else:
            print("\nThese need a code/config edit, not --fix "
                  "(see project/README.md, troubleshooting).")
        sys.exit(1)
    print("configs OK" + (" (repaired)" if args.fix else ""))


if __name__ == "__main__":
    main()
