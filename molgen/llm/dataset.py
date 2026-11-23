"""Dataset utilities for the SMILES language model.

A molecule is represented as the token sequence of one SMILES string, so the
generative model is a plain next-token predictor:

    p(S) = prod_t p(s_t | s_<t)

Two details matter for chemistry:

* Random SMILES augmentation.  One molecule has many valid SMILES encodings.
  Canonicalising every training string fixes one arbitrary ordering, which
  makes the model over-confident about that ordering.  Drawing a random
  encoding per epoch (MolToSmiles(doRandom=True)) instead teaches the model the
  equivalence class, which markedly improves sample diversity.
* Optional property conditioning.  If the corpus file carries a numeric
  property in its second column (e.g. QED or logP), we quantise it into deciles
  and expose the bucket id as a conditioning label, giving p(S | c).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import SmilesTokenizer

try:  # RDKit is only needed for augmentation / property computation
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    _HAS_RDKIT = True
except Exception:  # pragma: no cover - RDKit is optional for training
    _HAS_RDKIT = False


# A small, hand-written corpus used by ``--smoke`` so that the whole pipeline
# (tokenize -> train -> sample) can be exercised without QM9 on disk.
SMOKE_SMILES: Tuple[str, ...] = (
    "C", "N", "O", "CC", "CN", "CO", "CCO", "CCN", "CCC", "CNC",
    "c1ccccc1", "c1ccncc1", "c1ccccc1O", "CC(=O)O", "CC(=O)N", "CC(=O)C",
    "CCOCC", "CC(C)C", "CC(C)O", "NCCO", "OCCO", "NCC(=O)O", "C1CCCCC1",
    "c1ccc2ccccc2c1", "CCN(CC)CC", "CC(C)(C)O", "O=C=O", "N#N", "C#C",
    "CS(=O)(=O)C", "c1ccc(O)cc1", "CCCl", "CCBr", "FC(F)(F)C", "CCOC(=O)C",
    "CC(=O)OC", "NC(=O)N", "c1cnc[nH]1", "c1ccoc1", "c1ccsc1", "CCS",
    "CCNCC", "CCCN", "CCCO", "CCCC", "CCCCCC", "OCC(O)CO", "NCCN",
    "c1ccc(cc1)N", "c1ccc(cc1)C", "CC(=O)Nc1ccccc1", "OC(=O)c1ccccc1",
)


def read_smiles_file(path: str, limit: Optional[int] = None) -> Tuple[List[str], Optional[List[float]]]:
    """Read ``smiles[<sep>property]`` lines; returns (smiles, properties or None).

    The separator may be whitespace or a comma.  Lines whose first field is not
    a non-empty string are skipped.  Properties are returned only if *every*
    kept line has a parseable number in its second field.
    """
    smiles: List[str] = []
    props: List[float] = []
    prop_is_complete = True
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if not parts:
                continue
            smiles.append(parts[0])
            if len(parts) > 1:
                try:
                    props.append(float(parts[1]))
                except ValueError:
                    prop_is_complete = False
            else:
                prop_is_complete = False
            if limit is not None and len(smiles) >= limit:
                break
    if not prop_is_complete or len(props) != len(smiles):
        return smiles, None
    return smiles, props


def find_smiles_file(root: str, split: str) -> str:
    """Locate a SMILES file for a split inside ``root``.

    Accepts ``{split}.smi``, ``{split}.txt``, ``{split}.smiles`` either directly
    in ``root`` or under ``root/raw`` / ``root/processed``.
    """
    for folder in (root, os.path.join(root, "raw"), os.path.join(root, "processed")):
        for ext in ("smi", "txt", "smiles", "csv"):
            cand = os.path.join(folder, f"{split}.{ext}")
            if os.path.exists(cand):
                return cand
    raise FileNotFoundError(
        f"no SMILES file for split '{split}' under {root} "
        f"(looked for {split}.smi/.txt/.smiles/.csv in root, raw/ and processed/)"
    )


def randomize_smiles(smiles: str) -> str:
    """Return a random valid encoding of ``smiles`` (identity without RDKit)."""
    if not _HAS_RDKIT:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        return Chem.MolToSmiles(mol, doRandom=True, canonical=False)
    except Exception:
        return smiles


def quantile_bins(values: Sequence[float], n_bins: int) -> List[float]:
    """Internal bin edges of an equal-frequency discretisation."""
    sorted_vals = sorted(values)
    edges: List[float] = []
    for b in range(1, n_bins):
        idx = int(round(b * (len(sorted_vals) - 1) / n_bins))
        edges.append(sorted_vals[idx])
    return edges


def to_bucket(value: float, edges: Sequence[float]) -> int:
    """Bucket id of ``value`` given ascending internal edges."""
    bucket = 0
    for e in edges:
        if value > e:
            bucket += 1
        else:
            break
    return bucket


class SmilesDataset(Dataset):
    """Tokenised SMILES corpus.

    Args:
        smiles: SMILES strings.
        tokenizer: fitted :class:`SmilesTokenizer`.
        max_len: maximum number of tokens *including* <bos>/<eos>.
        props: optional numeric property per molecule (enables conditioning).
        n_cond_bins: number of property buckets; 0 disables conditioning.
        randomize: draw a random SMILES encoding on every ``__getitem__``.
    """

    def __init__(
        self,
        smiles: Sequence[str],
        tokenizer: SmilesTokenizer,
        max_len: int = 64,
        props: Optional[Sequence[float]] = None,
        n_cond_bins: int = 0,
        randomize: bool = True,
    ):
        self.smiles = list(smiles)
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.randomize = bool(randomize)
        self.n_cond_bins = int(n_cond_bins)
        self.edges: List[float] = []
        self.cond: Optional[List[int]] = None
        if self.n_cond_bins > 0 and props is not None and len(props) == len(self.smiles):
            self.edges = quantile_bins(props, self.n_cond_bins)
            self.cond = [to_bucket(float(v), self.edges) for v in props]

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        smi = self.smiles[i]
        if self.randomize:
            smi = randomize_smiles(smi)
        ids = self.tokenizer.encode(smi, add_special=True, max_len=self.max_len)
        ids = torch.tensor(ids, dtype=torch.long)
        item = {
            "input_ids": ids[:-1],          # <bos> ... s_{L-1}
            "labels": ids[1:],              # s_1   ... <eos>
        }
        if self.cond is not None:
            item["cond"] = torch.tensor(self.cond[i], dtype=torch.long)
        return item


def collate_batch(batch: List[Dict[str, torch.Tensor]], pad_id: int = 0):
    """Right-pad a batch to its longest sequence and build the loss mask."""
    lengths = torch.tensor([len(b["input_ids"]) for b in batch], dtype=torch.long)
    max_len = int(lengths.max())
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.float32)
    for row, b in enumerate(batch):
        n = len(b["input_ids"])
        input_ids[row, :n] = b["input_ids"]
        labels[row, :n] = b["labels"]
        # the final <eos> must contribute to the loss, so mask by position and
        # only zero out the padding region
        mask[row, :n] = 1.0
    out = {"input_ids": input_ids, "labels": labels, "loss_mask": mask, "lengths": lengths}
    if "cond" in batch[0]:
        out["cond"] = torch.stack([b["cond"] for b in batch])
    return out


def build_datasets(cfg, tokenizer: Optional[SmilesTokenizer] = None):
    """Construct (train, valid, tokenizer) from a data config section.

    The tokenizer is fitted on the training split only, so that validation and
    test SMILES never leak vocabulary statistics.
    """
    train_path = find_smiles_file(cfg.root, "train")
    train_smiles, train_props = read_smiles_file(train_path, limit=cfg.limit)
    if tokenizer is None:
        tokenizer = SmilesTokenizer.build(train_smiles, min_count=cfg.min_token_count,
                                          max_vocab=cfg.max_vocab)
    train_ds = SmilesDataset(
        train_smiles, tokenizer, max_len=cfg.max_len, props=train_props,
        n_cond_bins=cfg.n_cond_bins, randomize=cfg.randomize,
    )
    try:
        valid_path = find_smiles_file(cfg.root, "valid")
        valid_smiles, valid_props = read_smiles_file(valid_path, limit=cfg.limit)
    except FileNotFoundError:
        valid_smiles, valid_props = train_smiles, train_props
    valid_ds = SmilesDataset(
        valid_smiles, tokenizer, max_len=cfg.max_len, props=valid_props,
        n_cond_bins=cfg.n_cond_bins, randomize=False,
    )
    return train_ds, valid_ds, tokenizer
