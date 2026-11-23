"""SMILES tokenizer for the LLM branch.

The tokenizer splits a SMILES string with a chemistry-aware regular expression
instead of characters, so that two-character atoms (``Cl``, ``Br``), bracket
atoms (``[nH]``, ``[C@@H]``), ring-closure labels and stereo bonds stay single
tokens.  The vocabulary is built from the training corpus and serialised to
JSON so that inference can rebuild exactly the same mapping.

Reference for the token grammar: Schwaller et al., "Molecular Transformer"
(2019); the same split is used by SMILES-pair-encoding style models.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

# Order matters: longer patterns must come before shorter ones that prefix them.
SMILES_PATTERN = (
    r"(\[[^\]]+]"                                  # bracket atom, e.g. [nH], [C@@H]
    r"|Br|Cl"                                      # two-letter organic atoms
    r"|B|C|N|O|P|S|F|I"                            # one-letter organic atoms
    r"|b|c|n|o|p|s"                                # aromatic atoms
    r"|%\d{2}"                                     # ring closure >= 10, e.g. %10
    r"|\d"                                         # ring closure < 10
    r"|\.|=|#|-|\+|\\|/|:|~"                       # bonds and disconnect
    r"|\(|\)|\[|\]"                                # branches
    r"|\*|@@|@|H)")                                # wildcard / chirality / hydrogen
_SMILES_RE = re.compile(SMILES_PATTERN)

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS = (PAD, BOS, EOS, UNK)


def split_smiles(smiles: str) -> List[str]:
    """Split one SMILES string into chemistry-aware tokens.

    Anything the regular expression does not match is emitted as a single
    character so that encoding never silently drops information.
    """
    tokens: List[str] = []
    pos = 0
    while pos < len(smiles):
        match = _SMILES_RE.match(smiles, pos)
        if match:
            tokens.append(match.group(0))
            pos = match.end()
        else:
            tokens.append(smiles[pos])
            pos += 1
    return tokens


class SmilesTokenizer:
    """Bijective token <-> id mapping for SMILES strings.

    Args:
        itos: token strings, index position equals the token id.
    """

    def __init__(self, itos: Sequence[str]):
        self.itos: List[str] = list(itos)
        self.stoi: Dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]
        self.unk_id = self.stoi[UNK]

    # ---------------------------------------------------------------- #
    # construction
    # ---------------------------------------------------------------- #
    @classmethod
    def build(cls, smiles: Iterable[str], min_count: int = 1,
              max_vocab: Optional[int] = None) -> "SmilesTokenizer":
        """Build a vocabulary from a SMILES corpus.

        Tokens seen fewer than ``min_count`` times are mapped to ``<unk>``;
        ``max_vocab`` truncates to the most frequent tokens (special tokens are
        always kept at ids 0..3).
        """
        counter: Counter = Counter()
        for smi in smiles:
            counter.update(split_smiles(smi.strip()))
        # sort by frequency, then alphabetically for a deterministic vocab
        ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        ordered = [(t, c) for t, c in ordered if c >= min_count]
        if max_vocab is not None:
            ordered = ordered[: max(0, max_vocab - len(SPECIAL_TOKENS))]
        itos = list(SPECIAL_TOKENS) + [t for t, _ in ordered]
        return cls(itos)

    # ---------------------------------------------------------------- #
    # conversion
    # ---------------------------------------------------------------- #
    def encode(self, smiles: str, add_special: bool = True,
               max_len: Optional[int] = None) -> List[int]:
        """SMILES -> token ids (optionally wrapped in <bos>/<eos> and truncated)."""
        ids = [self.stoi.get(t, self.unk_id) for t in split_smiles(smiles.strip())]
        if max_len is not None:
            budget = max_len - (2 if add_special else 0)
            ids = ids[: max(budget, 0)]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        """token ids -> SMILES (special tokens dropped by default)."""
        out: List[str] = []
        for i in ids:
            i = int(i)
            if skip_special and i in (self.pad_id, self.bos_id, self.eos_id):
                continue
            if i < 0 or i >= len(self.itos):
                continue
            out.append(self.itos[i])
        return "".join(out)

    # ---------------------------------------------------------------- #
    # persistence
    # ---------------------------------------------------------------- #
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f, ensure_ascii=True, indent=2)

    @classmethod
    def load(cls, path: str) -> "SmilesTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f)["itos"])

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)
