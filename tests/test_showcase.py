"""Checks for tools/showcase.py that need neither torch nor RDKit.

The script imports numpy and Pillow at module level and defers ``torch``/``rdkit`` into the
functions that use them, so its pure helpers can be exercised here: the formula builder, the
geometry statistics, the LaTeX table writer and the index reader.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def load_showcase():
    path = os.path.join(PROJECT_ROOT, "tools", "showcase.py")
    spec = importlib.util.spec_from_file_location("_showcase", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_showcase"] = module
    spec.loader.exec_module(module)          # type: ignore[union-attr]
    return module


def main() -> None:
    sc = load_showcase()
    print("imported tools/showcase.py")

    # formula in Hill order with element symbols
    assert sc.formula([6, 6, 1, 1, 1, 8]) == "C2H3O", sc.formula([6, 6, 1, 1, 1, 8])
    assert sc.formula([1, 1]) == "H2"
    assert sc.formula([6]) == "C"
    print("formula():", sc.formula([6, 6, 1, 1, 1, 8]), sc.formula([1, 1]))

    # geometry: a collapsed pair must show a tiny minimum distance and clashes
    collapsed = np.zeros((6, 3))
    collapsed[:, 0] = np.linspace(0.0, 0.05, 6)
    geo = sc.geometry(collapsed)
    print("geometry(collapsed):", {k: round(v, 4) for k, v in geo.items()})
    assert geo["min_d"] < 0.06 and geo["clashes"] >= 5, geo

    # geometry: a stretched chain should have no clashes
    chain = np.zeros((6, 3))
    chain[:, 0] = np.arange(6) * 1.5
    geo2 = sc.geometry(chain)
    print("geometry(chain):", {k: round(v, 4) for k, v in geo2.items()})
    assert geo2["clashes"] == 0 and 1.4 < geo2["min_d"] < 1.6, geo2

    # the LaTeX summary writer must emit booktabs rows for the three methods
    rows = {
        "llm": [{"status": "sanitised", "atoms": 9, "novel": True, "index": 4}],
        "edm": [{"status": "unsanitised", "saturated": False, "min_d": 0.04,
                 "clashes": 17, "rg": 0.14, "index": 19}],
        "geobfn": [{"status": "sanitised", "saturated": True, "min_d": 1.02,
                    "clashes": 0, "rg": 2.04, "index": 19}],
    }
    body = sc.summary_rows(rows)
    for token in ("\\midrule", "\\bottomrule", "LLM", "EDM", "GeoBFN"):
        assert token in body, token
    print("summary_rows() emits", len(body.splitlines()), "lines")

    # indices: falls back to the documented defaults
    picks = sc.indices_from_report()
    print("indices:", picks)
    assert len(picks) == 10 and picks[0] == 4, picks

    print("\nSHOWCASE TOOL CHECKS PASSED")


if __name__ == "__main__":
    main()

