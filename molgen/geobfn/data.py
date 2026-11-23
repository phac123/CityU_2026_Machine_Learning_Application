# =============================================================================
# FILE ROLE
#   Data layer: turns the QM9 npz files into batches the BFN can consume
#   directly, plus a distributed sampler and a graph builder for evaluation.
#   This file does not depend on torch_geometric -- we assemble batches
#   ourselves (lighter and easier to follow).
#
# IMPLEMENTATION
#   1) QM9Dataset.__getitem__ returns a single molecule:
#        pos     : [n,3]  subtract the centroid (zero centroid), then divide by
#                         pos_scale
#        charges : [n,1]  atomic number Z mapped to bin center
#                         k_c(Z) = (2Z-1)/K - 1
#        n       : number of atoms
#      -- these two steps correspond to "zero centroid" in Theorem 3.1 and the
#      "K-bins definition" of Eq.15.
#   2) collate() assembles several molecules into "one big graph":
#        pos       [N,3]        N = sum n_b
#        charges   [N,1]
#        batch     [N]          which molecule each node belongs to
#                               (= segment_ids in the original repo)
#        edge_index[2,E]        fully connected within each molecule, no
#                               self-loops (E = sum n_b(n_b-1))
#      Benefit: no padding, no masks; every operator supports variable-size
#      molecules natively.
#   3) EvalDataset does not read real data: it samples n from the QM9 node-count
#      histogram and builds an "empty graph" for the sampler (generation only
#      needs node counts and topology).
#
# PAPER MAPPING
#   - zero centroid: Theorem 3.1 (translation invariance), Sec.3.1;
#   - fully connected graph: EGNN does all-pair message passing within a
#     molecule (the paper does not rely on chemical-bond priors);
#   - charge encoding: Eq.15's k_c(k)=(2k-1)/K-1; here Z lands exactly on the
#     Z-th bin center.
# =============================================================================
"""QM9 data, dynamic batch assembly, distributed sampler."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

# allowed atomic numbers -> bin-center index (Z is used directly as the Z-th bin)
DEFAULT_ATOMIC_NB = (1, 6, 7, 8, 9)


def k_centers(bins: int) -> torch.Tensor:
    """Bin centers k_c(k) = (2k-1)/K - 1 for K bins, k = 1..K. Returns [K]."""
    k = torch.arange(1, bins + 1, dtype=torch.float32)
    return (2 * k - 1) / bins - 1


def atomic_nb_to_bucket(z: torch.Tensor, bins: int) -> torch.Tensor:
    """Atomic number Z -> bin center value.

    For Z=6, K=9 this gives (2*6-1)/9-1 = 0.2222, which is exactly k_c(6).
    Such a "direct mapping" works because we choose K >= max(atomic_nb), so that
    every allowed Z has its own bin center (see docs section 2.2).
    """
    return (2.0 * z - 1.0) / bins - 1.0


def zero_com(pos: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Remove the centroid per molecule.

    Args:
        pos: [N,3]
        batch: [N], molecule id of each node; None means the whole tensor is
               one molecule.
    Returns:
        [N,3] coordinates after centering (each molecule's centroid is moved to
        the origin).
    """
    if batch is None:
        return pos - pos.mean(dim=0, keepdim=True)
    # hand-written equivalent of scatter_mean: mean per batch, then index back
    n_mol = int(batch.max().item()) + 1
    cnt = torch.zeros(n_mol, device=pos.device).index_add_(
        0, batch, torch.ones_like(batch, dtype=pos.dtype)
    )                                                        # [M] atoms per molecule
    s = torch.zeros(n_mol, pos.size(-1), device=pos.device, dtype=pos.dtype).index_add_(0, batch, pos)
    mean = s / cnt.clamp_min(1.0).unsqueeze(-1)              # [M,3]
    return pos - mean[batch]


def build_full_edge_index(sizes: Sequence[int], device: torch.device) -> torch.Tensor:
    """Build a fully connected (no self-loops) edge_index for a batch of molecules.

    For each molecule n: generate the n x n ordered pairs (i,j), drop i==j, then
    add the molecule's node offset.

    Args:
        sizes: number of nodes per molecule, e.g. [3, 4].
        device: target device.
    Returns:
        [2, sum n_b(n_b-1)], edge_index[0]=source(row), edge_index[1]=target(col).
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
        r = idx.repeat_interleave(n)          # [n*n] 0,0,...,1,1,...
        c = idx.repeat(n)                     # [n*n] 0,1,...,0,1,...
        keep = r != c
        rows.append(r[keep] + offset)
        cols.append(c[keep] + offset)
        offset += n
    if not rows:
        return torch.zeros(2, 0, dtype=torch.long, device=device)
    return torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)


class QM9Dataset(Dataset):
    """QM9 npz dataset.

    Expected files: `{root}/{split}.npz` (or `{root}/raw/{split}.npz`), fields:
        index      [N_mol]          molecule id
        num_atoms  [N_mol]          true atom count
        charges    [N_mol, Kmax]    atomic numbers (zero-padded)
        positions  [N_mol, Kmax, 3] coordinates (padded positions are 0)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        bins: int = 9,
        pos_scale: float = 1.0,
        atomic_nb: Sequence[int] = DEFAULT_ATOMIC_NB,
        limit: Optional[int] = None,
    ) -> None:
        self.root = root
        self.split = split
        self.bins = bins
        self.pos_scale = pos_scale
        self.atomic_nb = tuple(atomic_nb)

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

        # set of legal atomic numbers: used for validation (no other elements)
        self.allowed = set(self.atomic_nb)

    @staticmethod
    def _find_file(root: str, split: str) -> str:
        """Look for the npz in a few common locations.

        Compatible with the original repo's dataset/qm9/raw layout.
        """
        for cand in (
            os.path.join(root, f"{split}.npz"),
            os.path.join(root, "raw", f"{split}.npz"),
            os.path.join(root, "processed", f"{split}.npz"),
        ):
            if os.path.exists(cand):
                return cand
        raise FileNotFoundError(
            f"cannot find {split} data: tried {root}/{split}.npz, raw/, processed/.\n"
            f"See 'Data preparation' in the README; put the QM9 npz under {root}/raw/."
        )

    def __len__(self) -> int:
        return len(self.num_atoms)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        n = int(self.num_atoms[i])
        z = torch.as_tensor(self.charges[i][:n], dtype=torch.float32).view(-1, 1)  # [n,1]
        pos = torch.as_tensor(self.positions[i][:n], dtype=torch.float32)          # [n,3]
        pos = zero_com(pos) / self.pos_scale                                      # zero-centroid
        ch = atomic_nb_to_bucket(z, self.bins)                                     # Z -> k_c(Z)
        return {"pos": pos, "charges": ch, "n": n}


class SyntheticDataset(Dataset):
    """Synthetic data for debugging/smoke tests: random atoms, random coordinates.

    It has no chemical meaning. Only verifies that "the code runs, shapes are
    right, loss goes down"; it is not used for any experimental conclusion.
    """

    def __init__(self, size: int = 512, min_atoms: int = 3, max_atoms: int = 12,
                 bins: int = 9, atomic_nb: Sequence[int] = DEFAULT_ATOMIC_NB, seed: int = 0):
        self.size = size
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.bins = bins
        self.atomic_nb = list(atomic_nb)
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(self.seed * 100003 + i)
        n = int(torch.randint(self.min_atoms, self.max_atoms + 1, (1,), generator=g))
        pos = torch.randn(n, 3, generator=g)
        pos = zero_com(pos)
        z = torch.tensor(self.atomic_nb, dtype=torch.float32)[
            torch.randint(0, len(self.atomic_nb), (n,), generator=g)
        ].view(-1, 1)
        return {"pos": pos, "charges": atomic_nb_to_bucket(z, self.bins), "n": n}


class EvalDataset(Dataset):
    """Evaluation dataset: sample n from the node-count histogram, build empty graph.

    Generation tasks need no ground truth; the sampler only needs N, edge_index
    and batch.
    """

    def __init__(self, num: int, n_node_histogram: Sequence[int], seed: int = 1234):
        self.num = num
        hist = np.asarray(n_node_histogram, dtype=np.float64)
        self.probs = hist / hist.sum()
        self.seed = seed

    def __len__(self) -> int:
        return self.num

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        g = np.random.default_rng(self.seed + i)
        n = int(g.choice(len(self.probs), p=self.probs))
        n = max(n, 1)
        return {"n": n}


# --------------------------------------------------------------------------- #
# collate: one molecule dict -> one big graph
# --------------------------------------------------------------------------- #
def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Assemble several molecules into a PyG-style batch (but using plain torch).

    Returns:
        pos        [N,3]
        charges    [N,1]
        batch      [N]        node -> molecule id
        edge_index [2,E]      fully connected within each molecule, no self-loops
        sizes      [B]        atom count per molecule
        n_mol      int
    Note: the evaluation set has no pos/charges, so zeros are used as
    placeholders (the sampling process never reads them).
    """
    device = torch.device("cpu")
    sizes = [int(b["n"]) for b in batch]
    total = sum(sizes)

    if "pos" in batch[0]:
        pos = torch.cat([b["pos"] for b in batch], dim=0)
        charges = torch.cat([b["charges"] for b in batch], dim=0)
    else:
        pos = torch.zeros(total, 3)
        charges = torch.zeros(total, 1)

    batch_idx = torch.repeat_interleave(
        torch.arange(len(batch), dtype=torch.long), torch.tensor(sizes, dtype=torch.long)
    )
    edge_index = build_full_edge_index(sizes, device)
    return {
        "pos": pos,
        "charges": charges,
        "batch": batch_idx,
        "edge_index": edge_index,
        "sizes": torch.tensor(sizes, dtype=torch.long),
        "n_mol": len(batch),
    }


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move all tensors in the batch to the given device (non-tensors unchanged)."""
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
