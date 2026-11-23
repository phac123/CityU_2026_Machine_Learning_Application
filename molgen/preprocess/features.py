"""Molecular feature extraction for the exploratory part of the report.

The course project asks for a feature-exploration / dimensionality-reduction step
before modelling.  This module computes, for every molecule, a small set of
interpretable physicochemical descriptors with RDKit; ``explore.py`` then runs
PCA / t-SNE and KMeans on the standardised matrix.

Descriptors (all RDKit built-ins, so the numbers are reproducible):
    mw        molecular weight (Da)
    logp      Crippen logP
    qed       quantitative estimate of drug-likeness in [0,1]
    tpsa      topological polar surface area (A^2)
    hbd/hba   hydrogen-bond donors / acceptors
    rotb      rotatable bonds
    rings     ring count, arom   aromatic ring count
    fsp3      fraction of sp3 carbons
    n_atoms / n_heavy   atom counts
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

DESCRIPTOR_NAMES = [
    "mw", "logp", "qed", "tpsa", "hbd", "hba", "rotb", "rings", "arom", "fsp3",
    "n_atoms", "n_heavy",
]


def require_rdkit():
    """Import RDKit or fail with a clear message."""
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        return Chem
    except Exception as exc:  # pragma: no cover
        raise ImportError("RDKit is required for descriptor computation") from exc


def descriptors_of(smiles: str) -> Optional[List[float]]:
    """Descriptor vector of one SMILES, or ``None`` if RDKit cannot parse it."""
    Chem = require_rdkit()
    from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    try:
        return [
            float(Descriptors.MolWt(mol)),
            float(Crippen.MolLogP(mol)),
            float(QED.qed(mol)),
            float(rdMolDescriptors.CalcTPSA(mol)),
            float(Lipinski.NumHDonors(mol)),
            float(Lipinski.NumHAcceptors(mol)),
            float(Lipinski.NumRotatableBonds(mol)),
            float(rdMolDescriptors.CalcNumRings(mol)),
            float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            float(rdMolDescriptors.CalcFractionCSP3(mol)),
            float(mol.GetNumAtoms()),
            float(mol.GetNumHeavyAtoms()),
        ]
    except Exception:
        return None


def featurize(smiles: Sequence[str], verbose: bool = False):
    """Descriptor matrix [N, D] plus the SMILES that produced each row."""
    feats: List[List[float]] = []
    kept: List[str] = []
    failures = 0
    for smi in smiles:
        vec = descriptors_of(smi)
        if vec is None:
            failures += 1
            continue
        feats.append(vec)
        kept.append(smi)
    if verbose:
        print(f"[features] kept {len(kept)}/{len(smiles)} molecules "
              f"({failures} unparsable), D={len(DESCRIPTOR_NAMES)}")
    return np.asarray(feats, dtype=np.float64), kept


def standardize(x: np.ndarray) -> np.ndarray:
    """Zero-mean / unit-variance columns (descriptors live on very different scales)."""
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / np.where(sd < 1e-12, 1.0, sd)


def descriptor_table(smiles: Sequence[str]) -> Dict[str, np.ndarray]:
    """Convenience wrapper returning ``{"X": ..., "names": ..., "smiles": ...}``."""
    x, kept = featurize(smiles, verbose=True)
    return {"X": x, "names": DESCRIPTOR_NAMES, "smiles": kept}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser("Compute RDKit descriptors for a SMILES file")
    p.add_argument("path", help=".smi/.txt with one SMILES per line")
    p.add_argument("--out", default=None, help="optional .npz output")
    args = p.parse_args()

    with open(args.path, "r", encoding="utf-8", errors="ignore") as f:
        smis = [line.split()[0] for line in f if line.strip() and not line.startswith("#")]
    table = descriptor_table(smis)
    print("descriptor means:")
    for name, mean in zip(table["names"], table["X"].mean(axis=0)):
        print(f"  {name:10s} {mean:10.4f}")
    if args.out:
        np.savez(args.out, X=table["X"], names=np.array(table["names"]),
                 smiles=np.array(table["smiles"]))
        print(f"wrote {args.out}")
