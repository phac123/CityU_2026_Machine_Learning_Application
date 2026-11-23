"""RDKit helpers shared by the LLM branch (validity, canonicalisation, properties).

Imported lazily: training an language model does not strictly need RDKit, but
every reported number in the paper does, because validity is defined by the
chemistry toolkit and not by the model.
"""

from __future__ import annotations

from typing import Dict, Optional

try:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    _HAS_RDKIT = True
except Exception:  # pragma: no cover
    _HAS_RDKIT = False

_DESCRIPTORS_READY = False


def has_rdkit() -> bool:
    """Whether RDKit could be imported."""
    return _HAS_RDKIT


def mol_from_smiles(smiles: str):
    """Parse a SMILES string, returning ``None`` when it is chemically invalid."""
    if not _HAS_RDKIT or not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def is_valid_smiles(smiles: str) -> bool:
    """RDKit-parseable and non-degenerate (at least two atoms)."""
    mol = mol_from_smiles(smiles)
    return mol is not None and mol.GetNumAtoms() >= 2


def canonical_smiles(smiles: str) -> Optional[str]:
    """Canonical SMILES, or ``None`` if RDKit cannot parse the string."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _load_descriptors() -> None:
    global _DESCRIPTORS_READY, QED, Crippen
    if _DESCRIPTORS_READY:
        return
    from rdkit.Chem import Crippen, QED  # noqa: F401  (imported for globals)

    _DESCRIPTORS_READY = True


def molecular_properties(smiles: str) -> Optional[Dict[str, float]]:
    """QED and Crippen logP for a SMILES string.

    These are the two cheap, standard drug-likeness descriptors used for the
    *conditional* experiments: the conditioning bucket at generation time is a
    decile of one of them, and the target values are then compared against the
    properties of the generated molecules.
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    _load_descriptors()
    try:
        return {
            "qed": float(QED.qed(mol)),
            "logp": float(Crippen.MolLogP(mol)),
            "n_heavy": float(mol.GetNumHeavyAtoms()),
        }
    except Exception:
        return None
