"""Summarise the 10 sampled molecules of one model (qualitative analysis).

    python tools/analyse_samples.py --method geobfn --input runs/.../samples.xyz
    python tools/analyse_samples.py --method llm   --input runs/.../samples.smi

For a 3D sample (``.xyz``) it reuses the branch's own bond perception to report the
canonical SMILES, whether the sample is valid and stable, how many fragments it has
and the distribution of its bond lengths; for a SMILES list it reports the canonical
form and whether RDKit accepts it.  The markdown output is meant to be pasted into the
"Molecular generation" section of the report, one block per model.
"""
from __future__ import annotations

import argparse
import importlib
import os
from typing import Dict, List, Optional

import torch

METHOD_MODULES = {
    "geobfn": "molgen.geobfn.metrics",
    "edm": "molgen.edm.metrics",
    "llm": "molgen.llm.metrics",
}
ELEMENT = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}


# --------------------------------------------------------------------------- #
# readers
# --------------------------------------------------------------------------- #
def read_xyz(path: str) -> List[Dict[str, torch.Tensor]]:
    """Read a multi-frame XYZ file into (pos, z) records."""
    frames: List[Dict[str, torch.Tensor]] = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        while True:
            header = f.readline()
            if not header:
                break
            if not header.strip():
                continue
            n = int(header.strip())
            f.readline()                       # comment line
            z, pos = [], []
            for _ in range(n):
                parts = f.readline().split()
                z.append(int(parts[0]))
                pos.append([float(v) for v in parts[1:4]])
            frames.append({"z": torch.tensor(z), "pos": torch.tensor(pos)})
    return frames


def read_smiles(path: str) -> List[str]:
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [line.split()[0] for line in f
                if line.strip() and not line.startswith("#")]


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def n_fragments(bonds, n_atoms: int) -> int:
    """Number of connected components of the inferred bond graph."""
    parent = list(range(n_atoms))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j, _ in bonds:
        ra, rb = find(i), find(j)
        if ra != rb:
            parent[ra] = rb
    return len({find(i) for i in range(n_atoms)})


def formula(z: torch.Tensor) -> str:
    counts: Dict[int, int] = {}
    for zi in z.tolist():
        counts[int(zi)] = counts.get(int(zi), 0) + 1
    order = [6, 1, 7, 8, 9]
    parts = [f"{ELEMENT.get(k, k)}{counts[k]}" for k in order if k in counts]
    parts += [f"{ELEMENT.get(k, k)}{v}" for k, v in counts.items() if k not in order]
    return "".join(parts)


def analyse_3d(method: str, frames: List[Dict[str, torch.Tensor]]) -> List[Dict]:
    metrics = importlib.import_module(METHOD_MODULES[method])
    rows = []
    for idx, frame in enumerate(frames, 1):
        pos, z = frame["pos"], frame["z"]
        bonds, remaining = metrics.infer_bonds(pos, z)
        mol, _, _ = metrics.build_mol(pos, z)
        smiles = metrics.mol_to_smiles(mol)
        lengths = [float(torch.norm(pos[i] - pos[j])) for i, j, _ in bonds]
        rows.append({
            "index": idx,
            "atoms": int(z.numel()),
            "formula": formula(z),
            "smiles": smiles or "-",
            "valid": smiles is not None,
            "stable": bool(remaining) and all(r == 0 for r in remaining),
            "fragments": n_fragments(bonds, int(z.numel())),
            "bonds": len(bonds),
            "mean_bond": sum(lengths) / len(lengths) if lengths else float("nan"),
            "min_bond": min(lengths) if lengths else float("nan"),
            "max_bond": max(lengths) if lengths else float("nan"),
        })
    return rows


def analyse_smiles(smiles: List[str]) -> List[Dict]:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    rows = []
    for idx, smi in enumerate(smiles, 1):
        mol = Chem.MolFromSmiles(smi) if smi else None
        rows.append({
            "index": idx,
            "atoms": mol.GetNumAtoms() if mol else 0,
            "formula": Chem.rdMolDescriptors.CalcMolFormula(mol) if mol else "-",
            "smiles": Chem.MolToSmiles(mol) if mol else "-",
            "valid": mol is not None,
            "stable": mol is not None,
            "fragments": len(Chem.GetMolFrags(mol)) if mol else 0,
            "bonds": mol.GetNumBonds() if mol else 0,
            "mean_bond": float("nan"),
            "min_bond": float("nan"),
            "max_bond": float("nan"),
        })
    return rows


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def to_markdown(method: str, rows: List[Dict], source: str) -> str:
    lines = [f"### {method}: 10 randomly drawn samples", "",
             f"Source: `{source}`", "",
             "| # | atoms | formula | SMILES | valid | stable | frag. | bonds | mean bond (A) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        mean_bond = "-" if r["mean_bond"] != r["mean_bond"] else f"{r['mean_bond']:.2f}"
        lines.append(f"| {r['index']} | {r['atoms']} | {r['formula']} | "
                     f"`{r['smiles']}` | {'yes' if r['valid'] else 'no'} | "
                     f"{'yes' if r['stable'] else 'no'} | {r['fragments']} | "
                     f"{r['bonds']} | {mean_bond} |")
    n = len(rows)
    valid = sum(r["valid"] for r in rows)
    stable = sum(r["stable"] for r in rows)
    frag = sum(r["fragments"] > 1 for r in rows)
    single = sum(r["atoms"] <= 2 for r in rows)
    unique = len({r["smiles"] for r in rows if r["valid"]})
    lines += ["", f"- valid: **{valid}/{n}**, stable: **{stable}/{n}**, "
                  f"fragmented: **{frag}/{n}**, degenerate (<= 2 atoms): **{single}/{n}**, "
                  f"distinct: **{unique}/{n}**"]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=sorted(METHOD_MODULES))
    p.add_argument("--input", required=True, help=".xyz (3D samples) or .smi (strings)")
    p.add_argument("--out", default=None, help="markdown file to write")
    args = p.parse_args()

    if args.input.endswith(".smi"):
        rows = analyse_smiles(read_smiles(args.input))
    else:
        rows = analyse_3d(args.method, read_xyz(args.input))

    report = to_markdown(args.method, rows, args.input)
    print(report)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[analyse] wrote {args.out}")


if __name__ == "__main__":
    main()
