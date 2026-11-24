"""Configuration objects for the LLM branch.

Three dataclasses (data / model / train) are filled from a YAML (or JSON) file
and may be overridden on the command line with ``--set train.lr=5e-5``.  Unknown
keys raise an error instead of being silently ignored, which keeps the YAML and
the code in sync while the project evolves.
"""

from __future__ import annotations

import ast
import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DataConfig:
    """SMILES corpus settings."""

    name: str = "qm9"
    root: str = "dataset/qm9"          # expects raw/{train,valid,test}.smi
    limit: Optional[int] = None        # cap the number of molecules (debug)
    max_len: int = 64                  # tokens per SMILES, including <bos>/<eos>
    min_token_count: int = 1           # prune hapax tokens into <unk>
    max_vocab: Optional[int] = 256
    n_cond_bins: int = 0               # >0 enables p(S | property bucket)
    randomize: bool = True             # random SMILES augmentation
    num_workers: int = 4
    fold: int = -1                     # >=0: use fold k of n_folds as validation
    n_folds: int = 3
    valid_limit: Optional[int] = 512   # cap on the validation split (CV is cheap)
    valid_batches: int = 8             # batches used for one validation estimate
    batch_size: int = 64             # per GPU, identical for the three methods
    eval_batch_size: int = 256


@dataclass
class ModelConfig:
    """Transformer size."""

    d_model: int = 256
    n_head: int = 8
    n_layer: int = 6
    dropout: float = 0.1
    tie_embeddings: bool = True


@dataclass
class TrainConfig:
    """Optimisation and bookkeeping."""

    max_steps: int = 10000        # optimisation budget (fair comparison)
    lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    amp: str = "bf16"                  # none | bf16 | fp16
    seed: int = 0
    heartbeat_seconds: int = 30      # print a stall warning after N s (0 = off)
    progress_every: int = 1          # steps between progress lines (0 = off)
    eta_window: int = 50             # steps in the ETA window (rolling)
    log_every: int = 50
    eval_every: int = 1000           # steps between validation passes
    ckpt_every: int = 1000           # steps between checkpoints
    ckpt_keep_all: bool = False       # also keep ckpt/step%06d.pt, not just last.pt
    analysis_every: int = 2000        # steps between sampling analyses (0 = off)
    analysis_num: int = 128           # molecules per periodic analysis
    analysis_steps: int = 100         # reverse steps used by the periodic analysis
    analysis_pca: bool = True         # place the samples on the training manifold
    analysis_pca_ref: int = 8000      # reference molecules used to fit that manifold
    analysis_save_samples: bool = True  # write samples/analysis_<step>.xyz
    skip_training: bool = False       # evaluate eval.ckpt and stop (no training)
    sample_num: int = 128              # molecules drawn for the periodic sanity check
    sample_temperature: float = 0.9
    sample_top_k: int = 50
    out_dir: str = "runs/molgpt_qm9"
    resume: bool = False


@dataclass
class Config:
    """Top-level bundle."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _build(cls, value: Any):
    """Recursively fill a dataclass from a dict, rejecting unknown fields."""
    if value is None:
        return cls()
    if not isinstance(value, dict):
        raise TypeError(f"{cls.__name__} expects a dict, got {type(value)}")
    known = {f.name for f in fields(cls)}
    unknown = set(value) - known
    if unknown:
        raise KeyError(f"unknown fields for {cls.__name__}: {sorted(unknown)}")
    obj = cls()
    for k, v in value.items():
        cur = getattr(obj, k)
        if is_dataclass(cur) and isinstance(v, dict):
            setattr(obj, k, _build(type(cur), v))
        else:
            setattr(obj, k, v)
    return obj


def _read_file(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml"):
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"unsupported config format: {path}")


def apply_overrides(cfg: Config, overrides: List[str]) -> Config:
    """Apply ``section.key=value`` overrides (values parsed with literal_eval)."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item}")
        key, raw = item.split("=", 1)
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            if not hasattr(node, p):
                raise KeyError(f"no such config path: {key}")
            node = getattr(node, p)
        leaf = parts[-1]
        if not hasattr(node, leaf):
            raise KeyError(f"no such config path: {key}")
        try:
            val = ast.literal_eval(raw)
        except Exception:
            val = raw
        setattr(node, leaf, val)
    return cfg


SECTIONS = ("data", "model", "train")

#: Keys of the global Hydra config that belong to the run, not to the
#: method: the dispatcher and the launchers read them directly.
SHARED_KEYS = ("name", "seed", "smoke", "ddp", "eval", "cv", "budget", "hydra")


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> Config:
    """Defaults -> YAML -> command-line overrides."""
    cfg = Config()
    if path:
        raw = _read_file(path)
        for section in ("data", "model", "train"):
            if section in raw and isinstance(raw[section], dict):
                setattr(cfg, section, _build(type(getattr(cfg, section)), raw[section]))
        unknown = set(raw) - set(SECTIONS) - set(SHARED_KEYS)
        if unknown:
            raise KeyError(f"unknown config sections: {sorted(unknown)}")
    return apply_overrides(cfg, overrides or [])


def save_config(cfg: Config, path: str) -> None:
    """Persist the resolved config next to the checkpoint for reproducibility."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)


# --------------------------------------------------------------------------- #
# Hydra adapter
# --------------------------------------------------------------------------- #
def from_hydra(node) -> Config:
    """Build the dataclass config from a Hydra/OmegaConf node.

    Only the sections declared by this branch (see ``SECTIONS``) are read; everything
    else in the global Hydra config (the method switch, the eval block, Hydra's own
    bookkeeping) is ignored here.  Values must already be resolvable, so the caller
    passes the composed config before any interpolation is deferred.
    """
    from omegaconf import OmegaConf

    raw = OmegaConf.to_container(node, resolve=True)
    cfg = Config()
    for section in SECTIONS:
        if section in raw and isinstance(raw[section], dict):
            setattr(cfg, section, _build(type(getattr(cfg, section)), raw[section]))
    unknown = set(raw) - set(SECTIONS) - set(SHARED_KEYS)
    if unknown:
        raise KeyError(f"unknown config sections for this method: {sorted(unknown)}")
    return cfg
