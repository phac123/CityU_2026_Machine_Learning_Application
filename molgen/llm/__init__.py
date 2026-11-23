"""LLM branch of the project: a small GPT-style SMILES language model.

Public API:
    SmilesTokenizer   chemistry-aware tokenizer with JSON persistence
    SmilesDataset     tokenised corpus with random-SMILES augmentation
    MolGPT            decoder-only Transformer over SMILES
    sample_smiles     batched autoregressive sampling

The package is intentionally self-contained: it does not import anything from
the EDM or GeoBFN packages, so the three methods can be trained, evaluated and
shipped independently.
"""

from .config import Config, load_config, save_config
from .dataset import SmilesDataset, collate_batch
from .model import MolGPT
from .sampler import dedupe, sample_smiles
from .tokenizer import SmilesTokenizer

__all__ = [
    "Config",
    "load_config",
    "save_config",
    "SmilesDataset",
    "collate_batch",
    "MolGPT",
    "SmilesTokenizer",
    "sample_smiles",
    "dedupe",
]
