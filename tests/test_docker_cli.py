"""Dry-run test of docker.py: capture the docker commands it would run.

No docker is required; ``subprocess.run`` is replaced by a recorder, so this checks
the argument wiring of the CLI (the part that is easy to get wrong).
"""
import importlib.util
import io
import os
import sys
from pathlib import Path

ROOT = Path("project").resolve()
spec = importlib.util.spec_from_file_location("docker_cli", ROOT / "docker.py")
cli = importlib.util.module_from_spec(spec)
sys.modules["docker_cli"] = cli
spec.loader.exec_module(cli)

calls = []


class Recorder:
    def __init__(self, *a, **k):
        self.returncode = 0

    def run(self, cmd, **kwargs):  # noqa: A003
        calls.append(cmd)
        return type("R", (), {"returncode": 0})()


def fake_run(cmd, *a, **k):
    calls.append(cmd)
    return type("R", (), {"returncode": 0})()


cli.subprocess.run = fake_run
cli.shutil_which = lambda name: "/usr/bin/docker"          # pretend docker exists
cwd = os.getcwd()
os.chdir(ROOT)                                             # .env and paths resolve here

try:
    cli.main(["train", "-m", "edm", "-g", "4", "data.batch_size=32"])

    assert calls, "no docker command was produced"
    cmd = calls[-1]
    print("train ->", " ".join(cmd))
    assert cmd[:3] == ["docker", "compose", "run"], cmd[:3]
    assert "METHOD=edm" in cmd and "GPUS=4" in cmd
    assert cmd[-2:] == ["scripts/train.sh", "data.batch_size=32"], cmd[-2:]

    calls.clear()
    cli.main(["eval", "-m", "llm", "-c", "/workspace/logs/llm/run/last.pt",
              "eval.num=128"])
    cmd = calls[-1]
    print("eval  ->", " ".join(cmd))
    assert "METHOD=llm" in cmd
    assert "CKPT=/workspace/logs/llm/run/last.pt" in cmd
    assert cmd[-2:] == ["scripts/evaluate.sh", "eval.num=128"]

    calls.clear()
    cli.main(["train", "-m", "geobfn", "--nodes", "2", "--rank", "1",
              "--master-addr", "10.0.0.1", "--stamp", "run42"])
    cmd = calls[-1]
    print("multi ->", " ".join(cmd))
    assert "scripts/train_multi_node.sh" in cmd
    for needle in ("NNODES=2", "NODE_RANK=1", "MASTER_ADDR=10.0.0.1", "STAMP=run42"):
        assert needle in cmd, (needle, cmd)

    calls.clear()
    cli.main(["smoke"])
    print("smoke ->", " ".join(calls[-1]))
    assert "scripts/smoke.sh" in calls[-1]

    calls.clear()
    cli.main(["test"])
    print("test  ->", " ".join(calls[-1]))
    assert calls[-1][-1] == "test"

    calls.clear()
    cli.main(["preprocess", "--", "--source", "/data/gdb9.sdf"])
    print("pre   ->", " ".join(calls[-1]))
    assert "molgen.preprocess.build_dataset" in " ".join(calls[-1])
    assert "--source" in calls[-1], calls[-1]

    print("\nALL DOCKER CLI CHECKS PASSED")
finally:
    os.chdir(cwd)
