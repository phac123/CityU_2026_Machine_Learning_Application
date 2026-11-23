#!/usr/bin/env python3
"""docker.py - a thin, readable helper around ``docker compose`` for this project.

It is the implementation behind the ``Makefile`` (``make init`` -> ``docker.py
init``), and it can be used directly:

    ./docker.py init                 # configure the four host directories, build
    ./docker.py in                   # interactive shell in the container
    ./docker.py train -m edm -g 4    # torchrun inside the container
    ./docker.py eval -m edm -c runs/edm/best.pt
    ./docker.py smoke                # tiny end-to-end run of all three methods
    ./docker.py down                 # stop and remove the container

The four host directories are stored in ``.env`` next to this file:

    CODE_DIR       the repository                -> /workspace/code
    DATA_DIR       datasets (qm9, ...)           -> /workspace/data
    LOG_DIR        runs, checkpoints, samples    -> /workspace/logs
    CONTAINER_DIR  the container's home          -> /container
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"

PATH_KEYS = ("CODE_DIR", "DATA_DIR", "LOG_DIR", "CONTAINER_DIR")

DEFAULTS: Dict[str, str] = {
    "CODE_DIR": ".",
    "DATA_DIR": "./data",
    "LOG_DIR": "./logs",
    "CONTAINER_DIR": "./.container",
    "IMAGE_TAG": "latest",
    "CONTAINER_NAME": "molgen",
    "MASTER_PORT": "29500",
    "USER_ID": str(os.getuid() if hasattr(os, "getuid") else 1000),
    "GROUP_ID": str(os.getgid() if hasattr(os, "getgid") else 1000),
}

ENV_HEADER = """\
# Host directories mounted into the container (see docker-compose.yaml).
# Absolute paths are recommended; relative paths are resolved from this folder.
#
#   CODE_DIR       the repository               -> /workspace/code
#   DATA_DIR       datasets                     -> /workspace/data
#   LOG_DIR        runs / checkpoints / samples -> /workspace/logs
#   CONTAINER_DIR  the container's $HOME        -> /container
"""


# --------------------------------------------------------------------------- #
# .env handling
# --------------------------------------------------------------------------- #
def read_env() -> Dict[str, str]:
    """Parse .env into a dict (missing file -> empty dict)."""
    values: Dict[str, str] = {}
    if not ENV_FILE.exists():
        return values
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def write_env(values: Dict[str, str]) -> None:
    """Write .env, keeping the documented order and the header comment."""
    lines: List[str] = [ENV_HEADER]
    for key in PATH_KEYS:
        lines.append(f"{key}={values.get(key, DEFAULTS[key])}")
    lines.append("")
    for key in ("IMAGE_TAG", "CONTAINER_NAME", "MASTER_PORT", "USER_ID", "GROUP_ID"):
        lines.append(f"{key}={values.get(key, DEFAULTS[key])}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve(value: str) -> Path:
    """Absolute host path of a configured directory (relative to the project)."""
    p = Path(value)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()



def to_container_path(value: Optional[str]) -> Optional[str]:
    """Translate a host path that lives under a mount into its in-container path.

    ``make eval MODEL=edm CKPT=./logs/edm/run1/best.pt`` should work as well as
    ``CKPT=/workspace/logs/edm/run1/best.pt``; this is what makes both spellings
    equivalent.  Paths that do not exist on the host are returned unchanged (they
    are assumed to be container paths already).
    """
    if not value:
        return value
    candidate = Path(value)
    if not candidate.exists():
        return value
    candidate = candidate.resolve()
    values = {**DEFAULTS, **read_env()}
    for key, mount in (("CODE_DIR", "/workspace/code"), ("DATA_DIR", "/workspace/data"),
                       ("LOG_DIR", "/workspace/logs")):
        root = resolve(values[key])
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            continue
        return f"{mount}/{relative.as_posix()}"
    return value

# --------------------------------------------------------------------------- #
# docker plumbing
# --------------------------------------------------------------------------- #
def ensure_docker() -> None:
    """Fail early with an actionable message when docker is missing."""
    if shutil_which("docker") is None:
        sys.exit("docker not found in PATH; install Docker Desktop / Docker Engine first")


def shutil_which(name: str) -> Optional[str]:
    from shutil import which

    return which(name)


def ensure_env() -> None:
    """Create .env from the defaults if it does not exist yet.

    Every command except ``init`` then works out of the box; ``init`` is still the
    one that asks for the paths and builds the image.
    """
    if not ENV_FILE.exists():
        write_env(DEFAULTS)
        print(f"note: wrote a default {ENV_FILE.name} (edit it or run `init`)")


def compose(*args: str, check: bool = True) -> int:
    """Run ``docker compose <args>`` in the project directory."""
    ensure_env()
    cmd = ["docker", "compose", *args]
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=PROJECT_ROOT, check=check).returncode


def compose_run(service: str, *command: str, env: Optional[Dict[str, str]] = None,
                check: bool = True) -> int:
    """``docker compose run --rm [-e K=V ...] <service> <command>``."""
    args: List[str] = ["run", "--rm"]
    for key, value in (env or {}).items():
        if value is None or value == "":
            continue
        args += ["-e", f"{key}={value}"]
    args += [service, *command]
    return compose(*args, check=check)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_init(args: argparse.Namespace) -> None:
    """Configure .env, create the host directories and build the image."""
    ensure_docker()
    values = {**DEFAULTS, **read_env()}

    prompted = {k: getattr(args, k.lower(), None) for k in PATH_KEYS}
    interactive = not args.yes and any(v is None for v in prompted.values())
    if interactive:
        print("Where should the container look for its data? (Enter = default)\n")
        for key in PATH_KEYS:
            default = values.get(key, DEFAULTS[key])
            answer = input(f"  {key:14s} [{default}]: ").strip()
            values[key] = answer or default
    else:
        for key in PATH_KEYS:
            if getattr(args, key.lower(), None):
                values[key] = getattr(args, key.lower())
    if args.tag:
        values["IMAGE_TAG"] = args.tag

    for key in PATH_KEYS:
        path = resolve(values[key])
        path.mkdir(parents=True, exist_ok=True)
        print(f"  {key:14s} -> {path}")
    if not (resolve(values["CODE_DIR"]) / "train.py").exists():
        print(f"warning: {resolve(values['CODE_DIR'])} does not contain train.py; "
              f"CODE_DIR should point at the project folder")

    write_env(values)
    print(f"\nwrote {ENV_FILE}")
    print("building the image (first build takes a few minutes)\n")
    compose("build")
    print("\ndone. next steps:")
    print("  make in                     # shell inside the container")
    print("  make train MODEL=geobfn     # torchrun on all visible GPUs")
    print("  make smoke                  # 1-minute end-to-end check")
    print("  make eval MODEL=geobfn CKPT=<path/to/best.pt>")


def cmd_in(args: argparse.Namespace) -> None:
    """Interactive shell inside the container."""
    ensure_docker()
    compose_run("dev", args.shell if args.shell else "bash")


def cmd_exec(args: argparse.Namespace) -> None:
    """Run an arbitrary command inside the container."""
    ensure_docker()
    compose_run("dev", *args.command)


def cmd_build(args: argparse.Namespace) -> None:
    compose("build", *(["--no-cache"] if args.no_cache else []))


def cmd_up(args: argparse.Namespace) -> None:
    compose("up", "-d")


def cmd_down(args: argparse.Namespace) -> None:
    compose("down", "--remove-orphans")


def cmd_status(args: argparse.Namespace) -> None:
    compose("ps")
    values = {**DEFAULTS, **read_env()}
    print()
    for key in PATH_KEYS:
        print(f"  {key:14s} {resolve(values[key])}")


def cmd_logs(args: argparse.Namespace) -> None:
    compose("logs", "-f", "--tail", str(args.tail))


def cmd_train(args: argparse.Namespace) -> None:
    """torchrun inside the container (single node, all visible GPUs by default)."""
    ensure_docker()
    env = {"METHOD": args.model, "GPUS": args.gpus or ""}
    if args.nodes and args.nodes > 1:
        env.update({"NNODES": str(args.nodes), "NODE_RANK": str(args.rank),
                    "MASTER_ADDR": args.master_addr or "", "STAMP": args.stamp or ""})
        script = "scripts/train_multi_node.sh"
    else:
        script = "scripts/train.sh"
    compose_run("dev", "bash", script, *args.overrides, env=env)


def cmd_eval(args: argparse.Namespace) -> None:
    """Sample / score a checkpoint inside the container."""
    ensure_docker()
    if not args.ckpt:
        sys.exit("pass the checkpoint with -c/--ckpt")
    env = {"METHOD": args.model, "CKPT": to_container_path(args.ckpt)}
    compose_run("dev", "bash", "scripts/evaluate.sh", *args.overrides, env=env)


def cmd_smoke(args: argparse.Namespace) -> None:
    """Tiny end-to-end run of every method inside the container."""
    ensure_docker()
    env = {"METHOD": args.model or ""}
    compose_run("dev", "bash", "scripts/smoke.sh", env=env)


def cmd_all(args: argparse.Namespace) -> None:
    """Train then evaluate all three methods, one after the other."""
    ensure_docker()
    env = {"GPUS": args.gpus or ""}
    compose_run("dev", "bash", "scripts/run_all.sh", *args.overrides, env=env)


def cmd_test(args: argparse.Namespace) -> None:
    """Run the unit checks (no GPU, no data needed)."""
    ensure_docker()
    compose_run("test")


def cmd_preprocess(args: argparse.Namespace) -> None:
    """Build the dataset artefacts (npz + smi) inside the container."""
    ensure_docker()
    command = ["python", "-m", "molgen.preprocess.build_dataset", *args.overrides]
    compose_run("dev", *command)


def cmd_doctor(args: argparse.Namespace) -> None:
    """Report what the host provides: docker, compose, GPU, configured paths."""
    print(f"project root : {PROJECT_ROOT}")
    print(f"python       : {sys.version.split()[0]}")
    for tool in ("docker", "nvidia-smi"):
        path = shutil_which(tool)
        print(f"{tool:13s}: {path or 'NOT FOUND'}")
    if shutil_which("docker"):
        subprocess.run(["docker", "compose", "version"], check=False)
        subprocess.run(["docker", "info", "--format",
                        "{{.ServerVersion}} / runtimes: {{.Runtimes}}"], check=False)
    if shutil_which("nvidia-smi"):
        subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total",
                        "--format=csv,noheader"], check=False)
    values = {**DEFAULTS, **read_env()}
    print()
    for key in PATH_KEYS:
        path = resolve(values[key])
        print(f"{key:13s}: {path} {'(ok)' if path.exists() else '(missing)'}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="docker helpers for the three-method molecule-generation project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="configure .env, create directories, build image")
    for key in PATH_KEYS:
        init.add_argument(f"--{key.lower().replace('_', '-')}", dest=key.lower(),
                          default=None, help=f"host path for {key}")
    init.add_argument("-y", "--yes", action="store_true", help="accept the defaults")
    init.add_argument("--tag", default=None, help="image tag (default: latest)")
    init.set_defaults(func=cmd_init)

    shell = sub.add_parser("in", help="interactive shell in the container")
    shell.add_argument("--shell", default=None, help="shell binary (default: bash)")
    shell.set_defaults(func=cmd_in)

    run = sub.add_parser("exec", help="run a command in the container")
    run.add_argument("command", nargs=argparse.REMAINDER)
    run.set_defaults(func=cmd_exec)

    build = sub.add_parser("build", help="build the image")
    build.add_argument("--no-cache", action="store_true")
    build.set_defaults(func=cmd_build)

    sub.add_parser("up", help="start the container in the background").set_defaults(func=cmd_up)
    sub.add_parser("down", help="stop and remove the container").set_defaults(func=cmd_down)
    sub.add_parser("status", help="show container status and configured paths").set_defaults(func=cmd_status)

    logs = sub.add_parser("logs", help="follow the container logs")
    logs.add_argument("--tail", type=int, default=200)
    logs.set_defaults(func=cmd_logs)

    train = sub.add_parser("train", help="train one method (torchrun)")
    train.add_argument("-m", "--model", default="geobfn", choices=["geobfn", "edm", "llm"])
    train.add_argument("-g", "--gpus", default=None, help="GPUs per node (default: all)")
    train.add_argument("--nodes", type=int, default=1, help=">1 for multi-node")
    train.add_argument("--rank", type=int, default=0, help="node rank for multi-node")
    train.add_argument("--master-addr", default=None)
    train.add_argument("--stamp", default=None, help="shared run stamp across nodes")
    train.add_argument("overrides", nargs=argparse.REMAINDER, help="Hydra overrides")
    train.set_defaults(func=cmd_train)

    ev = sub.add_parser("eval", help="sample / score a checkpoint")
    ev.add_argument("-m", "--model", default="geobfn", choices=["geobfn", "edm", "llm"])
    ev.add_argument("-c", "--ckpt", default=None)
    ev.add_argument("overrides", nargs=argparse.REMAINDER, help="Hydra overrides")
    ev.set_defaults(func=cmd_eval)

    smoke = sub.add_parser("smoke", help="1-minute end-to-end check of all methods")
    smoke.add_argument("-m", "--model", default=None, choices=["geobfn", "edm", "llm"])
    smoke.set_defaults(func=cmd_smoke)

    everything = sub.add_parser("all", help="train + evaluate all three methods")
    everything.add_argument("-g", "--gpus", default=None)
    everything.add_argument("overrides", nargs=argparse.REMAINDER)
    everything.set_defaults(func=cmd_all)

    sub.add_parser("test", help="run the unit checks").set_defaults(func=cmd_test)

    pre = sub.add_parser("preprocess", help="build the dataset artefacts")
    pre.add_argument("overrides", nargs=argparse.REMAINDER)
    pre.set_defaults(func=cmd_preprocess)

    sub.add_parser("doctor", help="check docker / GPU / configured paths").set_defaults(func=cmd_doctor)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    # argparse.REMAINDER keeps a leading "--" as a literal argument
    if hasattr(args, "overrides"):
        args.overrides = [a for a in args.overrides if a != "--"]
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
