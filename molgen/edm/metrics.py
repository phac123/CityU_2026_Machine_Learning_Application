"""Chemical evaluation metrics for the EDM branch.

A generated sample is a set of atoms (coordinates + atomic numbers) with no bonds.
To score it chemically we must first *infer* the bonds, then hand the molecule to
RDKit.  The procedure used here (and by the other two branches) is:

1. look up the single-bond length L1(z_i, z_j) for the element pair;
2. collect candidate pairs with d_ij <= 1.30 * L1, sorted by distance;
3. assign a bond order from the ratio d_ij / L1 (<=0.85 triple, <=0.95 double,
   otherwise single), capped by the remaining valence of both atoms;
4. sanitise with RDKit: a molecule that survives is *valid*; a molecule in which
   every atom ends with zero remaining valence is *stable*.

The greedy valence-aware assignment matters in practice: an aromatic ring with all
bond lengths near 1.39 A is turned into one double plus one single bond (a Kekule
structure) instead of two double bonds, which would exceed carbon's valence and
falsely mark the molecule unstable.

Metrics reported (the same protocol as GeoBFN / EDM papers):
    validity    fraction of samples RDKit can sanitise
    mol_stable  fraction where every atom's valence is exactly saturated
    atm_stable  fraction of atoms whose valence is exactly saturated
    uniqueness  1 - duplicate fraction among valid SMILES
    novelty     fraction of valid SMILES absent from the training set
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

# Single-bond lengths in Angstrom, only the element pairs QM9 can contain.
SINGLE_BOND_LEN: Dict[Tuple[int, int], float] = {
    (1, 1): 0.74, (1, 6): 1.09, (1, 7): 1.01, (1, 8): 0.96, (1, 9): 0.92,
    (6, 6): 1.54, (6, 7): 1.47, (6, 8): 1.43, (6, 9): 1.35,
    (7, 7): 1.45, (7, 8): 1.40, (7, 9): 1.36,
    (8, 8): 1.48, (8, 9): 1.42, (9, 9): 1.42,
}
# Valence that a neutral atom of each element can carry.
ALLOWED_VALENCE: Dict[int, int] = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1}
# Bond-order thresholds, relative to the single-bond length.
TRIPLE_RATIO = 0.85
DOUBLE_RATIO = 0.95
BOND_RATIO = 1.30          # beyond this ratio the pair is not considered bonded


def _l1(z1: int, z2: int) -> Optional[float]:
    """Single-bond length of an element pair (order-insensitive)."""
    return SINGLE_BOND_LEN.get((min(z1, z2), max(z1, z2)))


def infer_bonds(pos: torch.Tensor, z: torch.Tensor) -> Tuple[List[Tuple[int, int, int]], List[int]]:
    """Infer bonds and bond orders from a distance matrix.

    Args:
        pos: [n,3] coordinates in Angstrom.
        z:   [n] atomic numbers.
    Returns:
        bonds: [(i, j, order)] with order in {1,2,3}.
        remaining: leftover valence of every atom (all zero means saturated).

    Complexity O(n^2); n <= 29 for QM9, so this is negligible.
    """
    n = int(z.numel())
    remaining = [ALLOWED_VALENCE.get(int(zi), 4) for zi in z.tolist()]
    if n <= 1:
        return [], remaining

    dist = torch.cdist(pos.float(), pos.float())
    zs = z.tolist()
    candidates: List[Tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(dist[i, j])
            l1 = _l1(zs[i], zs[j])
            if l1 is None or d > l1 * BOND_RATIO:
                continue
            candidates.append((d, i, j))
    candidates.sort(key=lambda t: t[0])          # short contacts bond first

    bonds: List[Tuple[int, int, int]] = []
    for d, i, j in candidates:
        l1 = _l1(zs[i], zs[j])
        ratio = d / l1
        if ratio <= TRIPLE_RATIO:
            order = 3
        elif ratio <= DOUBLE_RATIO:
            order = 2
        else:
            order = 1
        order = min(order, remaining[i], remaining[j])   # never exceed valence
        if order <= 0:
            continue
        bonds.append((i, j, order))
        remaining[i] -= order
        remaining[j] -= order
    return bonds, remaining


def build_mol(pos: torch.Tensor, z: torch.Tensor):
    """Build an RDKit molecule (unsanitised) from coordinates and atomic numbers."""
    from rdkit import Chem

    bonds, remaining = infer_bonds(pos, z)
    mol = Chem.RWMol()
    for zi in z.tolist():
        mol.AddAtom(Chem.Atom(int(zi)))
    bond_types = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    for i, j, order in bonds:
        mol.AddBond(int(i), int(j), bond_types[order])
    conf = Chem.Conformer(int(z.numel()))
    for i in range(int(z.numel())):
        conf.SetAtomPosition(i, pos[i].tolist())
    mol.AddConformer(conf, assignId=True)
    return mol, bonds, remaining


def mol_to_smiles(mol) -> Optional[str]:
    """Canonical SMILES of the largest fragment after sanitisation, else ``None``."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
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


def evaluate(samples: Sequence[Dict[str, torch.Tensor]],
             reference_smiles: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """Compute validity / stability / uniqueness / novelty for generated samples.

    Args:
        samples: list of {"pos": [n,3], "z": [n]}.
        reference_smiles: training-set SMILES, needed for novelty.
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

        mol, _, remaining = build_mol(pos, z)
        # stability: every atom's valence exactly used up
        if all(r == 0 for r in remaining) and int(z.numel()) > 1:
            n_stable_mol += 1
        n_stable_atom += sum(1 for r in remaining if r == 0)

        smi = mol_to_smiles(mol)
        if smi is not None:
            smiles_list.append(smi)

    valid = len(smiles_list)
    unique_smiles = set(smiles_list)
    return {
        "validity": valid / max(n_mol, 1),
        "uniqueness": len(unique_smiles) / max(valid, 1),
        "novelty": (
            len([s for s in unique_smiles if s not in ref]) / max(len(unique_smiles), 1)
            if ref is not None else 0.0
        ),
        "mol_stable": n_stable_mol / max(n_mol, 1),
        "atm_stable": n_stable_atom / max(n_atom, 1),
        "n_valid": float(valid),
    }


def collect_reference_smiles(root: str, split: str = "train",
                             limit: Optional[int] = None) -> List[str]:
    """SMILES of the training molecules, cached on disk for the novelty metric.

    The npz stores atomic numbers, so the molecule is rebuilt with the same
    distance-based bond inference used at evaluation time - the reference set and
    the samples are therefore scored by an identical code path.
    """
    cache = os.path.join(root, f"reference_smiles_{split}.txt")
    if os.path.exists(cache):
        with open(cache, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    from .data import QM9Dataset

    ds = QM9Dataset(root, split=split, limit=limit)
    atomic_nb = list(ds.atomic_nb)
    smis: List[str] = []
    for i in range(len(ds)):
        item = ds[i]
        z = torch.tensor([atomic_nb[int(c)] if int(c) < len(atomic_nb) else 6
                          for c in item["h"].tolist()], dtype=torch.long)
        mol, _, _ = build_mol(item["pos"] * ds.pos_scale, z)
        smi = mol_to_smiles(mol)
        if smi:
            smis.append(smi)
    with open(cache, "w", encoding="utf-8") as f:
        f.write("\n".join(smis))
    return smis


def save_xyz(path: str, samples: Sequence[Dict[str, torch.Tensor]]) -> None:
    """Write a multi-frame XYZ file (one molecule per block)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            pos, znum = s["pos"].cpu(), s["z"].cpu()
            f.write(f"{len(znum)}\n\n")
            for (cx, cy, cz), zi in zip(pos.tolist(), znum.tolist()):
                f.write(f"{int(zi):2d} {cx:12.6f} {cy:12.6f} {cz:12.6f}\n")
