"""SMILES -> 3D conformer bridge for the LLM branch.

An autoregressive SMILES model knows nothing about geometry, so the LLM method
generates a molecular graph first and then *embeds* it in 3D with RDKit's
distance-geometry code (ETKDGv3) followed by a force-field relaxation.  This is
the standard "1D generator + deterministic 3D decoder" baseline, and it lets us
run the very same 3D evaluation harness that the diffusion and Bayesian-flow
models are scored with, which is what makes the three-method comparison fair.

Note the asymmetry that the report highlights: the geometry here is produced by
a *heuristic* (ETKDG) and is not learned, so the quality of the 3D structure is
capped by the toolkit rather than by the model.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    _HAS_RDKIT = True
except Exception:  # pragma: no cover
    _HAS_RDKIT = False

# Element symbol -> atomic number, for the elements that appear in QM9.
_SYMBOL_TO_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "B": 5, "P": 15, "S": 16, "Cl": 17, "Br": 35}


def smiles_to_mol(smiles: str, add_hydrogens: bool = True):
    """Parse SMILES and (optionally) make every hydrogen explicit.

    Explicit hydrogens are required for parity with QM9, whose molecules are
    stored with all atoms (``num_atoms`` counts hydrogen) - the diffusion and
    Bayesian-flow models both generate hydrogen atoms explicitly.
    """
    if not _HAS_RDKIT:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if add_hydrogens:
        mol = Chem.AddHs(mol)
    return mol


def smiles_to_3d(
    smiles: str,
    seed: int = 0xC0FFEE,
    optimize: Optional[str] = "MMFF",
    max_attempts: int = 4,
) -> Optional[Dict[str, torch.Tensor]]:
    """Embed one molecule in 3D.

    Args:
        smiles: input SMILES (conformer generated for the same molecule).
        seed: ETKDG random seed; different seeds give different conformers.
        optimize: "MMFF", "UFF" or None for no force-field relaxation.
        max_attempts: retries with a perturbed seed if embedding fails.
    Returns:
        dict with ``pos`` [n,3] float32 coordinates in Angstrom, ``z`` [n] atomic
        numbers, and ``smiles``; ``None`` if RDKit cannot embed the molecule.
    """
    mol = smiles_to_mol(smiles, add_hydrogens=True)
    if mol is None:
        return None
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed) & 0x7FFFFFFF
    params.useSmallRingTorsions = True
    conf_id = -1
    for attempt in range(max_attempts):
        params.randomSeed = int(seed + 7919 * attempt) & 0x7FFFFFFF
        conf_id = AllChem.EmbedMolecule(mol, params)
        if conf_id >= 0:
            break
    if conf_id < 0:
        # last resort: random coordinates (still a valid graph, worse geometry)
        conf_id = AllChem.EmbedMolecule(mol, randomSeed=int(seed) & 0x7FFFFFFF,
                                        useRandomCoords=True)
        if conf_id < 0:
            return None
    if optimize == "MMFF" and AllChem.MMFFHasAllMoleculeParams(mol):
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            pass
    elif optimize == "UFF":
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            pass

    conf = mol.GetConformer()
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)          # [n,3]
    z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
    # centre of mass free coordinates, exactly as the 3D models are trained
    pos = pos - pos.mean(dim=0, keepdim=True)
    return {"pos": pos, "z": z, "smiles": smiles}


def smiles_list_to_3d(
    smiles: Sequence[str],
    seed: int = 0xC0FFEE,
    optimize: Optional[str] = "MMFF",
) -> Tuple[List[Dict[str, torch.Tensor]], List[str]]:
    """Embed a batch of SMILES; returns (successes, failed SMILES).

    A per-molecule seed makes the embedding reproducible while still decorrelating
    the conformers of different molecules.
    """
    samples: List[Dict[str, torch.Tensor]] = []
    failed: List[str] = []
    for i, smi in enumerate(smiles):
        out = smiles_to_3d(smi, seed=seed + i, optimize=optimize)
        if out is None:
            failed.append(smi)
        else:
            samples.append(out)
    return samples, failed


def save_xyz(path: str, samples: Sequence[Dict[str, torch.Tensor]]) -> None:
    """Write multi-frame XYZ (one block per molecule) for visual inspection."""
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            pos, z = s["pos"].cpu(), s["z"].cpu()
            f.write(f"{len(z)}\n\n")
            for (cx, cy, cz), zi in zip(pos.tolist(), z.tolist()):
                f.write(f"{int(zi):2d} {cx:12.6f} {cy:12.6f} {cz:12.6f}\n")
