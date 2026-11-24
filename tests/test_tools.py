"""Unit tests for the analysis tools that do not need torch, RDKit or a dataset.

The bug this pins: ``tools/test_generation.py`` printed ``result['pca_outside']`` while
the PCA statistics were nested under ``result["manifold"]`` -> ``KeyError`` at the very
end of a four-minute evaluation.
"""

from __future__ import annotations

import importlib.util
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def load_tool(name: str):
    path = os.path.join(PROJECT_ROOT, "tools", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_tool_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"_tool_{name}"] = module
    spec.loader.exec_module(module)          # type: ignore[union-attr]
    return module


def test_build_result() -> None:
    tool = load_tool("test_generation")
    result = tool.build_result(
        method="edm",
        ckpt="/tmp/run/last.pt",
        requested=256,
        smiles=["C", "CC", "CCC"],
        metrics={"validity": 0.62, "uniqueness": 1.0},
        manifold_stats={"n": 3.0, "outside": 0.08, "mahal_ratio": 1.00, "nn_mean": 1.07},
        coverage_stats={"cov_n": 3.0, "cov_nn_median": 1.1, "cov_inside": 0.9},
    )
    # DescriptorManifold.project() already prefixes its keys: they must not be doubled
    result2 = tool.build_result(
        method="geobfn", ckpt="/tmp/run/last.pt", requested=256, smiles=["C"],
        metrics={"validity": 1.0},
        manifold_stats={"pca_n": 1.0, "pca_outside": 0.48, "pca_mahal_ratio": 1.6,
                        "pca_nn_mean": 2.1, "pca_var12": 0.53},
        coverage_stats={"cov_n": 1.0, "cov_inside": 0.08},
    )
    assert "pca_outside" in result2 and result2["pca_outside"] == 0.48, sorted(result2)
    assert not [k for k in result2 if k.startswith("pca_pca_")], sorted(result2)
    print("build_result: pre-prefixed keys are not doubled")
    print("keys:", sorted(result))
    # the flat keys the log line and the figures read
    for key in ("pca_n", "pca_outside", "pca_mahal_ratio", "pca_nn_mean",
                "cov_inside", "cov_nn_median", "num_samples", "metrics"):
        assert key in result, key
    # ... and the grouped copies
    assert result["manifold"]["pca_outside"] == 0.08
    assert result["coverage"]["cov_inside"] == 0.9
    assert result["num_samples"] == 3 and result["num_requested"] == 256
    assert result["ckpt"].endswith("last.pt")
    print("build_result: flat + grouped keys present")


def test_kill_script_patterns() -> None:
    """The kill script must match the launchers and must not match this test."""
    path = os.path.join(PROJECT_ROOT, "scripts", "kill_training.sh")
    text = open(path, encoding="utf-8").read()
    for token in ("torchrun", "train\\.py", "evaluate\\.py", "run_experiments\\.sh",
                  "kill_training\\.sh"):
        assert token in text, token
    print("kill_training.sh: patterns present")


def main() -> None:
    test_build_result()
    test_kill_script_patterns()
    print("\nTOOL CHECKS PASSED")


if __name__ == "__main__":
    main()

