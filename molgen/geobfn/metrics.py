# =============================================================================
# FILE ROLE
#   Evaluation layer: turns "atom coordinates + atomic numbers" into
#   chemically decidable molecules and computes the paper's metrics:
#     validity (can RDKit parse it), uniqueness, novelty,
#     mol_stable / atm_stable (is the valence reasonable).
#
# IMPLEMENTATION
#   1) infer_bonds(): infers bond orders from the distance matrix. The method is
#      "distance table + valence-constrained greedy assignment":
#        - look up a single-bond length table L1(element, element) for each pair;
#        - judge the candidate bond order from the ratio d/L1 (<=0.85 triple,
#          <=0.95 double, <=1.30 single);
#        - assign from nearest to farthest, never exceeding the remaining
#          valence of either endpoint (min(candidate, remaining, remaining)).
#      This is more robust than a pure threshold: the 1.39 A bond in a benzene
#      ring becomes one double + one single (a Kekule structure), instead of two
#      doubles blowing up the carbon valence.
#   2) Build an RDKit RWMol from the inferred bonds; a successful Sanitize means
#      valid, then take the canonical SMILES.
#   3) stability is judged by "is the remaining valence 0": a molecule is stable
#      iff every atom's valence is used up exactly.
#
# RELATION TO THE PAPER / ORIGINAL IMPLEMENTATION
#   The paper's evaluation follows the GeoDiff/EDM protocol (bond-order table +
#   allowed_bonds valence table).
#   Here one compact single-bond length table + valence greedy assignment
#   replaces the original repository's 280-line three-bond table; the metric
#   semantics are the same and the implementation is shorter. If strict
#   agreement with the official numbers is needed, replace infer_bonds with the
#   original bond_analyze.get_bond_order (same interface).
#
#   Note: this file is imported only during evaluation; the training path does
#   not depend on RDKit.
# =============================================================================
"""Molecular metrics: bond orders, RDKit molecules, validity/stability/uniqueness/novelty."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

# --------------------------------------------------------------------------- #
# Chemical constants (covering only H, C, N, O, F)
# --------------------------------------------------------------------------- #
# Single-bond lengths (Angstrom). Used to convert a measured distance into a relative
# length, which determines the bond order.
SINGLE_BOND_LEN: Dict[Tuple[int, int], float] = {
    (1, 1): 0.74, (1, 6): 1.09, (1, 7): 1.01, (1, 8): 0.96, (1, 9): 0.92,
    (6, 6): 1.54, (6, 7): 1.47, (6, 8): 1.43, (6, 9): 1.35,
    (7, 7): 1.45, (7, 8): 1.40, (7, 9): 1.36,
    (8, 8): 1.48, (8, 9): 1.42, (9, 9): 1.42,
}
# Allowed valences (the paper's stability criterion)
ALLOWED_VALENCE: Dict[int, int] = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1}
# Bond-order thresholds (ratios relative to the single-bond length)
TRIPLE_RATIO = 0.85
DOUBLE_RATIO = 0.95
BOND_RATIO = 1.30   # above this, no bond is assumed


def _l1(z1: int, z2: int) -> float:
    """Look up the single-bond length; None if the pair is missing.

    Only 5 elements can occur in this repository, so the table is complete for
    our purposes.
    """
    return SINGLE_BOND_LEN.get((min(z1, z2), max(z1, z2)))


def infer_bonds(
    pos: torch.Tensor, z: torch.Tensor
) -> Tuple[List[Tuple[int, int, int]], List[int]]:
    """Infer bonds (with orders) from coordinates and atomic numbers.

    Also returns the remaining valence of every atom.

    Args:
        pos: [n,3] coordinates (Angstrom)
        z:   [n]   atomic numbers (1/6/7/8/9)
    Returns:
        bonds: [(i, j, order)], order in {1,2,3}
        remaining: valence left on each atom after bond assignment (all 0 =
                   valence saturated)

    Complexity O(n^2) (n <= 29, negligible).
    """
    n = int(z.numel())
    remaining = [ALLOWED_VALENCE.get(int(zi), 4) for zi in z.tolist()]
    if n <= 1:
        return [], remaining

    dist = torch.cdist(pos.float(), pos.float())  # [n,n]
    zs = z.tolist()

    # collect candidate pairs and sort by distance (near pairs bond first)
    cand: List[Tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(dist[i, j])
            l1 = _l1(zs[i], zs[j])
            if l1 is None or d > l1 * BOND_RATIO:
                continue
            cand.append((d, i, j))
    cand.sort(key=lambda t: t[0])

    bonds: List[Tuple[int, int, int]] = []
    for d, i, j in cand:
        l1 = _l1(zs[i], zs[j])
        ratio = d / l1
        if ratio <= TRIPLE_RATIO:
            order = 3
        elif ratio <= DOUBLE_RATIO:
            order = 2
        else:
            order = 1
        # key point: cannot exceed the remaining valence of either endpoint
        order = min(order, remaining[i], remaining[j])
        if order <= 0:
            continue
        bonds.append((i, j, order))
        remaining[i] -= order
        remaining[j] -= order
    return bonds, remaining


def build_mol(pos: torch.Tensor, z: torch.Tensor):
    """Build an RDKit molecule (without Sanitize). Returns (mol, bonds, remaining)."""
    from rdkit import Chem

    bonds, remaining = infer_bonds(pos, z)
    mol = Chem.RWMol()
    for zi in z.tolist():
        mol.AddAtom(Chem.Atom(int(zi)))
    bt = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    for i, j, order in bonds:
        mol.AddBond(int(i), int(j), bt[order])
    # write 3D coordinates, used for visualization / exporting the mol
    conf = Chem.Conformer(int(z.numel()))
    for i in range(int(z.numel())):
        conf.SetAtomPosition(i, pos[i].tolist())
    mol.AddConformer(conf, assignId=True)
    return mol, bonds, remaining


def mol_to_smiles(mol) -> Optional[str]:
    """Sanitize, then return the canonical SMILES of the largest fragment.

    Returns None on failure (= invalid molecule).
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")   # silence RDKit's C++-level warnings
    try:
        m = Chem.Mol(mol)
        Chem.SanitizeMol(m)
        frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=True)
        if not frags:
            return None
        largest = max(frags, key=lambda x: x.GetNumAtoms())
        smi = Chem.MolToSmiles(largest, canonical=True)
        return smi if smi else None
    except Exception:
        return None


def evaluate(
    samples: Sequence[Dict[str, torch.Tensor]],
    reference_smiles: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Compute metrics for a batch of generated samples.

    Args:
        samples: list of {"pos": [n,3], "z": [n]}
        reference_smiles: set of training-set SMILES (for novelty); when None,
                          novelty is recorded as 0
    Returns:
        dict: validity / uniqueness / novelty / mol_stable / atm_stable / n_valid
    """
    ref = set(reference_smiles) if reference_smiles is not None else None
    n_mol = len(samples)
    n_stable_mol = 0
    n_stable_atom = 0
    n_atom = 0
    smiles_list: List[str] = []

    for s in samples:
        pos, z = s["pos"].float().cpu(), s["z"].cpu()
        n_atom += int(z.numel())

        # ---- stability: is the valence used up exactly ----
        _, _, remaining = build_mol(pos, z)
        if all(r == 0 for r in remaining) and int(z.numel()) > 1:
            n_stable_mol += 1
        n_stable_atom += sum(1 for r in remaining if r == 0)

        # ---- validity: can RDKit parse it ----
        mol, _, _ = build_mol(pos, z)
        smi = mol_to_smiles(mol)
        if smi is not None:
            smiles_list.append(smi)

    valid = len(smiles_list)
    unique = len(set(smiles_list))
    out = {
        "validity": valid / max(n_mol, 1),
        "uniqueness": unique / max(valid, 1),
        "novelty": (
            len([s for s in set(smiles_list) if s not in ref]) / max(unique, 1)
            if ref is not None else 0.0
        ),
        "mol_stable": n_stable_mol / max(n_mol, 1),
        "atm_stable": n_stable_atom / max(n_atom, 1),
        "n_valid": float(valid),
    }
    return out

def collect_reference_smiles(root: str, split: str = "train", limit: Optional[int] = None) -> List[str]:
    """Read the training set and compute its SMILES set (the novelty reference).

    The result is cached in `{root}/reference_smiles.txt` to avoid recomputing it
    at every evaluation.
    """
    cache = os.path.join(root, f"reference_smiles_{split}.txt")
    if os.path.exists(cache):
        with open(cache, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    from .data import QM9Dataset

    ds = QM9Dataset(root, split=split, limit=limit)
    smis: List[str] = []
    for i in range(len(ds)):
        item = ds[i]
        # the data stores "bin-center values"; invert them back to atomic numbers
        z = (item["charges"].view(-1) + 1.0) * (ds.bins / 2.0) + 0.5
        z = z.round().long()
        mol, _, _ = build_mol(item["pos"] * ds.pos_scale, z)
        smi = mol_to_smiles(mol)
        if smi:
            smis.append(smi)
    with open(cache, "w", encoding="utf-8") as f:
        f.write("\n".join(smis))
    return smis

def save_xyz(path: str, samples: Sequence[Dict[str, torch.Tensor]]) -> None:
    """Write the samples as a multi-frame XYZ file (for visualization / downstream analysis)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            pos, znum = s["pos"].cpu(), s["z"].cpu()
            f.write(f"{len(znum)}\n\n")
            for (cx, cy, cz), zi in zip(pos.tolist(), znum.tolist()):
                f.write(f"{int(zi):2d} {cx:12.6f} {cy:12.6f} {cz:12.6f}\n")
