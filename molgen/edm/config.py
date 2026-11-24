"""Configuration objects for the EDM branch (E(3)-equivariant diffusion).

Descriptions of the fields are also the notation used in the report:
``n_timesteps`` is the number of discrete diffusion steps T, ``cosine_s`` is the
small offset of the cosine noise schedule, and ``lambda_h`` weights the discrete
(atom type) term against the continuous (coordinate) term.
"""

from __future__ import annotations

import ast
import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DataConfig:
    """QM9 (npz) settings."""

    name: str = "qm9"
    root: str = "dataset/qm9"          # expects raw/{train,valid,test}.npz
    atomic_nb: List[int] = field(default_factory=lambda: [1, 6, 7, 8, 9])
    pos_scale: float = 1.0             # coordinates are divided by this factor
    num_workers: int = 4
    limit: Optional[int] = None        # cap the training split (CV / debug)
    fold: int = -1                     # >=0: use fold k of n_folds as validation
    n_folds: int = 3
    valid_limit: Optional[int] = 512   # cap on the validation split (CV is cheap)
    valid_batches: int = 8             # batches used for one validation estimate
    batch_size: int = 64             # per GPU, identical for the three methods
    eval_num: int = 1024
    eval_batch_size: int = 64
    n_node_histogram: List[int] = field(
        default_factory=lambda: [
            0, 0, 0, 1, 4, 5, 9, 16, 49, 124, 362, 807, 1689, 3060, 5136, 7796,
            10644, 13025, 13364, 13832, 9482, 9970, 3393, 4848, 539, 1506, 48, 266, 0, 25,
        ]
    )


@dataclass
class ModelConfig:
    """EGNN backbone."""

    hidden_nf: int = 256
    n_layers: int = 9
    attention: bool = False            # gated edge attention inside the EGNN
    tanh: bool = True                  # bound the coordinate update (EDM recipe)
    coords_range: float = 10.0
    norm_constant: float = 1.0         # stabilises 1/r for near-coincident atoms
    time_embed_dim: int = 128


@dataclass
class DiffusionConfig:
    """Forward/reverse process hyper-parameters."""

    n_timesteps: int = 1000            # T
    cosine_s: float = 0.008            # cosine schedule offset
    loss_weight: str = "snr"           # "none" | "snr"  (weight is a_t^2 / s_t^2)
    w_clip: float = 100.0              # upper clamp on the SNR weight (numerical)
    lambda_h: float = 0.1              # weight of the discrete cross-entropy term
    sample_steps: int = 1000
    sample_eta: float = 0.0            # 0 = DDIM (deterministic), 1 = DDPM ancestral
    clip_denoise: float = 1.0e3        # clamp on the implied x0 during sampling


@dataclass
class TrainConfig:
    """Optimisation / bookkeeping."""

    max_steps: int = 10000        # optimisation budget (fair comparison)
    lr: float = 2.0e-4
    warmup_ratio: float = 0.05    # linear warm-up, then a constant lr
    weight_decay: float = 1.0e-12
    grad_clip: float = 5.0
    ema_decay: float = 0.0           # 0 disables EMA: raw weights are evaluated
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
    out_dir: str = "runs/edm_qm9"
    resume: bool = False


@dataclass
class Config:
    """Top-level bundle."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


SECTIONS = ("data", "model", "diffusion", "train")

#: Keys of the global Hydra config that belong to the run, not to the
#: method: the dispatcher and the launchers read them directly.
SHARED_KEYS = ("name", "seed", "smoke", "ddp", "eval", "cv", "budget", "hydra")


def _build(cls, value: Any):
    """Recursively fill a dataclass, rejecting unknown keys."""
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
    """Apply ``section.key=value`` overrides."""
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


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> Config:
    """Defaults -> YAML -> command-line overrides."""
    cfg = Config()
    if path:
        raw = _read_file(path)
        for section in SECTIONS:
            if section in raw and isinstance(raw[section], dict):
                setattr(cfg, section, _build(type(getattr(cfg, section)), raw[section]))
        unknown = set(raw) - set(SECTIONS) - set(SHARED_KEYS)
        if unknown:
            raise KeyError(f"unknown config sections: {sorted(unknown)}")
    return apply_overrides(cfg, overrides or [])


def save_config(cfg: Config, path: str) -> None:
    """Persist the resolved config next to the checkpoint."""
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
