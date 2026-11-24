"""Regression test: ``from_hydra`` must accept the *global* Hydra config.

The composed config contains the run-level keys (``name``, ``seed``, ``smoke``,
``ddp``, ``eval``, and ``cv`` when cv.py runs) next to the method sections.  An
earlier version treated all of them as unknown sections and every launch died with

    KeyError: unknown config sections for this method: ['ddp', 'eval', 'name', ...]

This test builds the same dict by hand, so it needs neither Hydra nor PyYAML.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# from_hydra imports omegaconf; stub it when the test runs outside the image
try:                                                    # pragma: no cover
    import omegaconf  # noqa: F401
except ImportError:                                     # pragma: no cover
    import copy

    def _has_hydra_interp(value, depth=0):
        if depth > 6:
            return False
        if isinstance(value, dict):
            return any(_has_hydra_interp(k, depth + 1) or _has_hydra_interp(v, depth + 1)
                       for k, v in value.items())
        if isinstance(value, (list, tuple)):
            return any(_has_hydra_interp(v, depth + 1) for v in value)
        return isinstance(value, str) and "${hydra:" in value

    def _to_container(node, resolve=True):
        # emulate OmegaConf: resolving ${hydra:...} needs a running Hydra job
        if resolve and _has_hydra_interp(node):
            raise RuntimeError("InterpolationResolutionError: ValueError raised while "
                               "resolving interpolation: HydraConfig was not set "
                               "(full_key: train.out_dir)")
        return copy.deepcopy(node)

    stub = types.ModuleType("omegaconf")
    stub.OmegaConf = types.SimpleNamespace(          # type: ignore[attr-defined]
        to_container=_to_container,
        create=lambda data: data,
    )
    stub.to_container = _to_container                    # type: ignore[attr-defined]
    stub.DictConfig = dict                               # type: ignore[attr-defined]
    sys.modules["omegaconf"] = stub

METHODS = ("geobfn", "edm", "llm")

#: what `conf/config.yaml` adds around every method block
GLOBAL_KEYS = {
    "name": "llm",
    "seed": 0,
    "smoke": False,
    "ddp": {"backend": None, "timeout_minutes": 30, "find_unused_parameters": False},
    "eval": {"ckpt": None, "num": 1024, "use_ema": False},
    "cv": {"folds": [0, 1, 2], "budget": {"max_steps": 500}},
}

#: one minimal method block per branch (the dataclass fills everything else)
METHOD_BLOCKS = {
    "geobfn": {
        "data": {"name": "qm9", "root": "data/qm9", "batch_size": 64, "bins": 9},
        "model": {"hidden_nf": 64, "n_layers": 2},
        "train": {"max_steps": 10, "lr": 1e-4, "ema_decay": 0.0},
    },
    "edm": {
        "data": {"name": "qm9", "root": "data/qm9", "batch_size": 64},
        "model": {"hidden_nf": 64, "n_layers": 2},
        "diffusion": {"n_timesteps": 100, "sample_steps": 10},
        "train": {"max_steps": 10, "lr": 2e-4, "ema_decay": 0.0},
    },
    "llm": {
        "data": {"name": "qm9", "root": "data/qm9", "batch_size": 64, "max_len": 32},
        "model": {"d_model": 32, "n_layer": 1},
        "train": {"max_steps": 10, "lr": 5e-4},
    },
}


def load_config_module(method: str):
    """Load ``molgen/<method>/config.py`` without importing the package.

    The package ``__init__`` pulls in torch and the model, which a configuration test
    must not require; ``config.py`` has no relative imports, so loading it by path is
    enough.
    """
    path = os.path.join(PROJECT_ROOT, "molgen", method, "config.py")
    name = f"_cfg_{method}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module        # dataclasses look the module up by name
    spec.loader.exec_module(module)   # type: ignore[union-attr]
    return module


def main() -> None:
    for method in METHODS:
        module = load_config_module(method)
        node = dict(GLOBAL_KEYS)
        node.update(METHOD_BLOCKS[method])
        node["name"] = method

        # 1. the global keys must be ignored, not rejected
        cfg = module.from_hydra(node)
        assert cfg.train.max_steps == 10, (method, cfg.train.max_steps)
        assert cfg.data.batch_size == 64, (method, cfg.data.batch_size)
        print(f"{method}: from_hydra accepted {sorted(node)}")

        # 2. a genuinely unknown section must still fail loudly
        broken = dict(node)
        broken["nonsense"] = {"a": 1}
        try:
            module.from_hydra(broken)
        except KeyError as exc:
            assert "nonsense" in str(exc), exc
            print(f"{method}: unknown section rejected -> {exc}")
        else:
            raise AssertionError(f"{method}: unknown section was silently accepted")

        # 3. a wrong type inside a section must still fail (dataclass validation)
        broken = dict(node)
        broken["train"] = {"lr": 1e-4, "not_a_field": 1}
        try:
            module.from_hydra(broken)
        except KeyError as exc:
            assert "not_a_field" in str(exc), exc
            print(f"{method}: unknown field rejected -> {exc}")
        else:
            raise AssertionError(f"{method}: unknown train field was accepted")

    # 4. the safety net used by tools/check_configs.py: an unresolved
    #    ${hydra:runtime.output_dir} must be replaced before from_hydra is called,
    #    otherwise the probe dies with the error the user saw in the container.
    sys.path.insert(0, PROJECT_ROOT)
    from tools.check_configs import _probe

    for method in METHODS:
        module = load_config_module(method)
        node = dict(GLOBAL_KEYS)
        node.update(METHOD_BLOCKS[method])
        node["name"] = method
        node["train"] = dict(node["train"])
        node["train"]["out_dir"] = "${hydra:runtime.output_dir}"

        try:                       # without the probe this is exactly the old failure
            module.from_hydra(node)
        except RuntimeError as exc:
            assert "HydraConfig was not set" in str(exc), exc
        else:
            raise AssertionError(f"{method}: the stub should have refused to resolve")

        probe = _probe(node)
        assert probe["train"]["out_dir"] == "/tmp/molgen-config-check", probe["train"]
        cfg = module.from_hydra(probe)
        assert cfg.train.out_dir == "/tmp/molgen-config-check"
        print(f"{method}: probe neutralised the hydra interpolation")

    print("\nCONFIG ADAPTER CHECKS PASSED")


if __name__ == "__main__":
    main()
