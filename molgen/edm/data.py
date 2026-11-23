"""QM9 data pipeline for the EDM branch.

Differences to the GeoBFN loader are deliberate and are discussed in the report:

* atom types are *class indices* ``0..K-1`` (a categorical variable), because the
  diffusion model needs a discrete state space, whereas GeoBFN works with a
  continuous charge value placed at a bin centre;
* coordinates are centred on the centre of mass (zero COM) per molecule, which is
  the translation-invariance constraint that the equivariant network relies on;
* a batch is one big disjoint union graph (``edge_index`` + ``batch`` segment ids),
  so variable-size molecules need neither padding nor masks.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_ATOMIC_NB = (1, 6, 7, 8, 9)


def zero_com(pos: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Subtract the per-molecule mean so that every molecule has zero COM."""
    if batch is None:
        return pos - pos.mean(dim=0, keepdim=True)
    n_mol = int(batch.max().item()) + 1
    cnt = torch.zeros(n_mol, device=pos.device, dtype=pos.dtype)
    cnt = cnt.index_add_(0, batch, torch.ones_like(batch, dtype=pos.dtype))
    s = torch.zeros(n_mol, pos.size(-1), device=pos.device, dtype=pos.dtype)
    s = s.index_add_(0, batch, pos)
    mean = s / cnt.clamp_min(1.0).unsqueeze(-1)
    return pos - mean[batch]


def build_full_edge_index(sizes: Sequence[int], device: torch.device) -> torch.Tensor:
    """Fully connected, self-loop free edges for a batch of molecules.

    EDM (like GeoBFN) does not use chemical bonds as input: the graph is the
    complete graph inside each molecule, so the model has to learn valence rules
    from geometry alone.
    """
    rows: List[torch.Tensor] = []
    cols: List[torch.Tensor] = []
    offset = 0
    for n in sizes:
        n = int(n)
        if n <= 1:
            offset += n
            continue
        idx = torch.arange(n, device=device)
        r = idx.repeat_interleave(n)
        c = idx.repeat(n)
        keep = r != c
        rows.append(r[keep] + offset)
        cols.append(c[keep] + offset)
        offset += n
    if not rows:
        return torch.zeros(2, 0, dtype=torch.long, device=device)
    return torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)


class QM9Dataset(Dataset):
    """QM9 stored as ``{root}/{split}.npz`` (or under ``raw/``).

    Expected arrays (the layout used by the original GeoBFN/EDM releases):
        index      [N_mol]
        num_atoms  [N_mol]            heavy + hydrogen atoms
        charges    [N_mol, K_max]     atomic numbers, zero padded
        positions  [N_mol, K_max, 3]  Angstrom, zero padded
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        pos_scale: float = 1.0,
        atomic_nb: Sequence[int] = DEFAULT_ATOMIC_NB,
        limit: Optional[int] = None,
    ):
        self.root = root
        self.split = split
        self.pos_scale = float(pos_scale)
        self.atomic_nb = tuple(int(z) for z in atomic_nb)
        self.z_to_class = {z: i for i, z in enumerate(self.atomic_nb)}

        path = self._find_file(root, split)
        data = np.load(path)
        self.num_atoms = np.asarray(data["num_atoms"])
        self.charges = np.asarray(data["charges"])
        self.positions = np.asarray(data["positions"])
        self.index = np.asarray(data["index"]) if "index" in data else np.arange(len(self.num_atoms))
        if limit is not None:
            self.num_atoms = self.num_atoms[:limit]
            self.charges = self.charges[:limit]
            self.positions = self.positions[:limit]
            self.index = self.index[:limit]

    @staticmethod
    def _find_file(root: str, split: str) -> str:
        for cand in (
            os.path.join(root, f"{split}.npz"),
            os.path.join(root, "raw", f"{split}.npz"),
            os.path.join(root, "processed", f"{split}.npz"),
        ):
            if os.path.exists(cand):
                return cand
        raise FileNotFoundError(
            f"cannot find {split} data under {root} "
            f"(looked in {split}.npz, raw/{split}.npz, processed/{split}.npz)"
        )

    def __len__(self) -> int:
        return len(self.num_atoms)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        n = int(self.num_atoms[i])
        z = torch.as_tensor(self.charges[i][:n], dtype=torch.long)
        # atomic number -> class index; unknown elements are clamped to carbon
        h = torch.tensor([self.z_to_class.get(int(v), 1) for v in z.tolist()], dtype=torch.long)
        pos = torch.as_tensor(self.positions[i][:n], dtype=torch.float32)
        pos = zero_com(pos) / self.pos_scale
        return {"pos": pos, "h": h, "n": n}


class SyntheticDataset(Dataset):
    """Random molecules, for smoke tests only (no chemical meaning)."""

    def __init__(self, size: int = 512, min_atoms: int = 3, max_atoms: int = 12,
                 n_types: int = 5, seed: int = 0):
        self.size = size
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.n_types = n_types
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(self.seed * 100003 + i)
        n = int(torch.randint(self.min_atoms, self.max_atoms + 1, (1,), generator=g))
        pos = zero_com(torch.randn(n, 3, generator=g))
        h = torch.randint(0, self.n_types, (n,), generator=g)
        return {"pos": pos, "h": h, "n": n}


class EvalDataset(Dataset):
    """Molecule sizes only: generation needs n, not chemistry."""

    def __init__(self, num: int, n_node_histogram: Sequence[int], seed: int = 1234):
        self.num = num
        hist = np.asarray(n_node_histogram, dtype=np.float64)
        self.probs = hist / hist.sum()
        self.seed = seed

    def __len__(self) -> int:
        return self.num

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        g = np.random.default_rng(self.seed + i)
        n = max(int(g.choice(len(self.probs), p=self.probs)), 1)
        return {"n": n}


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Concatenate molecules into one disjoint graph."""
    sizes = [int(b["n"]) for b in batch]
    total = sum(sizes)
    if "pos" in batch[0]:
        pos = torch.cat([b["pos"] for b in batch], dim=0)
        h = torch.cat([b["h"] for b in batch], dim=0)
    else:
        pos = torch.zeros(total, 3)
        h = torch.zeros(total, dtype=torch.long)
    batch_idx = torch.repeat_interleave(
        torch.arange(len(batch), dtype=torch.long), torch.tensor(sizes, dtype=torch.long)
    )
    return {
        "pos": pos,
        "h": h,
        "batch": batch_idx,
        "edge_index": build_full_edge_index(sizes, torch.device("cpu")),
        "sizes": torch.tensor(sizes, dtype=torch.long),
        "n_mol": len(batch),
    }


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move every tensor of a batch to ``device``."""
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
