"""Build the shared dataset artefacts used by all three methods.

The three generative models read *different* projections of the same molecules:

* EDM and GeoBFN read ``dataset/qm9/raw/{split}.npz`` (dense arrays of atomic
  numbers and coordinates),
* the LLM branch reads ``dataset/qm9/raw/{split}.smi`` (one SMILES per line).

This script produces both from a single raw source (QM9 ``gdb9.sdf`` +
``gdb9.sdf.csv``, or any SDF/SMILES file), so the models are guaranteed to see the
same molecules and the same split - a requirement for a fair comparison.

It also implements the two split protocols used in the report:

* ``random``    - uniformly shuffled, the protocol used by the original GeoBFN and
                  EDM papers;
* ``scaffold``  - Bemis-Murcko scaffold split, which measures generalisation to
                  unseen molecular frameworks instead of memorised ones.

Usage:
    python build_dataset.py --source gdb9.sdf --out dataset/qm9 --split random
    python build_dataset.py --source gdb9.sdf --out dataset/qm9 --split scaffold
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ELEMENTS = (1, 6, 7, 8, 9)          # H, C, N, O, F - the chemistry of QM9


def require_rdkit():
    """Import RDKit or fail with an actionable message."""
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        return Chem
    except Exception as exc:  # pragma: no cover
        raise ImportError("RDKit is required to read SDF files and compute SMILES") from exc


# --------------------------------------------------------------------------- #
# reading raw QM9
# --------------------------------------------------------------------------- #
def read_sdf(path: str, limit: Optional[int] = None) -> List[dict]:
    """Read a multi-molecule SDF into ``{"z", "pos", "smiles"}`` records."""
    Chem = require_rdkit()
    supplier = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
    records: List[dict] = []
    for i, mol in enumerate(supplier):
        if mol is None:
            continue
        if limit is not None and len(records) >= limit:
            break
        z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
        if any(int(v) not in ELEMENTS for v in z):
            continue                      # keep only the five QM9 elements
        conf = mol.GetConformer()
        pos = np.array(conf.GetPositions(), dtype=np.float32)
        # SMILES with explicit hydrogens removed (the graph is implicit in SMILES)
        try:
            smi = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)), canonical=True)
        except Exception:
            smi = ""
        records.append({"z": z, "pos": pos, "smiles": smi, "index": i})
    return records


def read_smiles_source(path: str) -> List[dict]:
    """Read ``SMILES`` lines (or a ``smiles`` column of a CSV) into records.

    A 3D structure is generated with ETKDGv3 so that the npz arrays exist, which
    makes this path usable for corpora that ship without coordinates.
    """
    Chem = require_rdkit()
    from rdkit.Chem import AllChem

    smiles: List[str] = []
    if path.lower().endswith(".csv"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            key = "smiles" if "smiles" in {k.lower() for k in reader.fieldnames or []} else None
            if key is None:
                raise KeyError("CSV needs a 'smiles' column")
            for row in reader:
                val = row.get(key) or row.get(key.upper())
                if val:
                    smiles.append(val.strip())
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    smiles.append(line.split()[0])

    records: List[dict] = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE + i) < 0:
            continue
        z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
        if any(int(v) not in ELEMENTS for v in z):
            continue
        records.append({
            "z": z,
            "pos": np.array(mol.GetConformer().GetPositions(), dtype=np.float32),
            "smiles": Chem.MolToSmiles(mol, canonical=True),
            "index": i,
        })
    return records


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def scaffold_of(smiles: str) -> str:
    """Bemis-Murcko scaffold (the ring systems plus linkers) of a SMILES."""
    Chem = require_rdkit()
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def random_split(n: int, ratios: Sequence[float], seed: int = 0) -> Dict[str, List[int]]:
    """Shuffle indices and cut them into train/valid/test."""
    idx = np.random.default_rng(seed).permutation(n)
    n_train = int(round(ratios[0] * n))
    n_valid = int(round(ratios[1] * n))
    return {
        "train": idx[:n_train].tolist(),
        "valid": idx[n_train:n_train + n_valid].tolist(),
        "test": idx[n_train + n_valid:].tolist(),
    }


def scaffold_split(records: Sequence[dict], ratios: Sequence[float]) -> Dict[str, List[int]]:
    """Group molecules by scaffold, then fill each split largest-scaffold first.

    This is the standard cheap approximation of a scaffold split: scaffolds are
    never shared between splits, and every split gets a comparable share of the
    data (deterministic, no seed needed).
    """
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        groups[scaffold_of(rec["smiles"])].append(i)
    # large scaffolds first, so that the test set is not dominated by singletons
    ordered = sorted(groups.values(), key=len, reverse=True)
    n = sum(len(g) for g in ordered)
    targets = [ratios[0] * n, ratios[1] * n]
    splits: Dict[str, List[int]] = {"train": [], "valid": [], "test": []}
    names = ["train", "valid", "test"]
    for group in ordered:
        # greedy: put the scaffold in the split that is furthest below its target
        deficits = [targets[0] - len(splits["train"]),
                    targets[1] - len(splits["valid"]),
                    (n - targets[0] - targets[1]) - len(splits["test"])]
        k = int(np.argmax(deficits))
        splits[names[k]].extend(group)
    return splits


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
def to_arrays(records: Sequence[dict]) -> Dict[str, np.ndarray]:
    """Pad variable-size molecules into the dense npz layout.

    Shapes: index [N], num_atoms [N], charges [N, K_max], positions [N, K_max, 3]
    with zero padding, exactly what the EDM and GeoBFN loaders expect.
    """
    n = len(records)
    k_max = max(int(len(r["z"])) for r in records) if n else 0
    charges = np.zeros((n, k_max), dtype=np.int64)
    positions = np.zeros((n, k_max, 3), dtype=np.float32)
    num_atoms = np.zeros(n, dtype=np.int64)
    index = np.zeros(n, dtype=np.int64)
    for i, rec in enumerate(records):
        m = int(len(rec["z"]))
        num_atoms[i] = m
        index[i] = rec.get("index", i)
        charges[i, :m] = rec["z"]
        positions[i, :m] = rec["pos"]
    return {"index": index, "num_atoms": num_atoms, "charges": charges, "positions": positions}


def write_split(out_root: str, name: str, records: Sequence[dict]) -> dict:
    """Write ``{name}.npz`` (dense arrays) and ``{name}.smi`` (SMILES) plus stats."""
    raw = os.path.join(out_root, "raw")
    os.makedirs(raw, exist_ok=True)
    arrays = to_arrays(records)
    np.savez(os.path.join(raw, f"{name}.npz"), **arrays)
    with open(os.path.join(raw, f"{name}.smi"), "w", encoding="utf-8") as f:
        for rec in records:
            if rec["smiles"]:
                f.write(rec["smiles"] + "\n")
    return {
        "split": name,
        "n_molecules": len(records),
        "n_atoms_mean": float(arrays["num_atoms"].mean()) if len(records) else 0.0,
        "n_atoms_max": int(arrays["num_atoms"].max()) if len(records) else 0,
        "symbols": Counter(
            int(z) for rec in records for z in rec["z"]
        ).most_common(),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Build QM9 npz/smi artefacts for all three methods")
    p.add_argument("--source", type=str, required=True,
                   help="gdb9.sdf (with coordinates) or a .smi/.csv of SMILES")
    p.add_argument("--out", type=str, default="dataset/qm9")
    p.add_argument("--split", type=str, default="random", choices=["random", "scaffold"])
    p.add_argument("--ratios", type=str, default="0.8,0.1,0.1",
                   help="train,valid,test fractions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ratios = [float(x) for x in args.ratios.split(",")]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("--ratios must be three fractions summing to 1")

    if args.source.lower().endswith(".sdf"):
        records = read_sdf(args.source, limit=args.limit)
    else:
        records = read_smiles_source(args.source)
        if args.limit:
            records = records[: args.limit]
    print(f"[read] {len(records)} molecules from {args.source}")

    if args.split == "random":
        idx = random_split(len(records), ratios, seed=args.seed)
    else:
        idx = scaffold_split(records, ratios)

    stats = []
    for name in ("train", "valid", "test"):
        subset = [records[i] for i in idx[name]]
        stats.append(write_split(args.out, name, subset))
        print(f"[write] {name}: {len(subset)} molecules -> {args.out}/raw/{name}.npz")

    with open(os.path.join(args.out, "dataset_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"source": args.source, "split": args.split, "ratios": ratios,
                   "seed": args.seed, "splits": stats}, f, indent=2)
    print(f"[done] stats -> {args.out}/dataset_stats.json")


if __name__ == "__main__":
    main()
