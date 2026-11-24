# =============================================================================
# FILE ROLE
#   The single configuration entry point for true_geobfn: dataclasses describe
#   the data/model/training hyperparameters, and this module loads them from
#   YAML / JSON, merges command-line overrides, and serializes back to a dict.
#
# IMPLEMENTATION
#   1) Three dataclasses (DataConfig / ModelConfig / TrainConfig) + a top-level
#      Config;
#   2) `load_config(path, overrides)`:
#        - reads yaml (if PyYAML is installed) or json;
#        - accepts only fields declared on the dataclass (a typo raises instead
#          of being silently ignored);
#        - supports dotted overrides such as `--set train.lr=1e-4` (see
#          apply_overrides).
#   3) `Config.to_dict()` is used for logging / checkpointing, so that "the
#      config used for training" is reproducible.
#
# WHY DATACLASS INSTEAD OF A PLAIN DICT
#   The training script is full of accesses like cfg.model.sigma1_coord;
#   dataclasses give types, defaults and IDE completion, and avoid the original
#   repository's "you only find the typo when cfg.dynamics.xxx crashes" issue.
#
# PAPER MAPPING
#   The hyperparameters here map directly to quantities in the paper/code:
#     sigma1_coord / sigma1_charges  -> the paper's sigma1 (residual noise scale
#                                       at t=1, which sets the accuracy schedule
#                                       rho(t)=sigma1^(-2t))
#     bins                           -> K in paper Eq.15 (number of discrete bins)
#     t_min                          -> not in the paper, but required by the
#                                       implementation as a time lower bound
#                                       (prevents division by zero when gamma->0)
# =============================================================================
"""Configuration definition and loading."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List


@dataclass
class DataConfig:
    """Data-related configuration.

    Attributes:
        name: dataset name (only qm9 and synthetic are implemented).
        root: data root; expects raw/train.npz, raw/valid.npz, raw/test.npz.
        n_node_histogram: histogram of QM9 node counts (including hydrogens),
                          used to sample molecule sizes during evaluation.
        atomic_nb: allowed atomic numbers (H, C, N, O, F).
        bins: number of charge discretization bins K; must be >= max(atomic_nb)
              so that every Z has a bin center.
        pos_scale: coordinate scale factor (the original repo uses 2.0; the
                   default here is 1.0, see docs section 2.2).
        num_workers: number of DataLoader worker processes (one set per rank).
        batch_size: **per-GPU** batch size; global batch = batch_size x world_size.
        eval_num: number of molecules generated during evaluation.
        eval_batch_size: evaluation batch size.
    """

    name: str = "qm9"
    root: str = "dataset/qm9"
    n_node_histogram: List[int] = field(
        default_factory=lambda: [
            0, 0, 0, 1, 4, 5, 9, 16, 49, 124, 362, 807, 1689, 3060, 5136, 7796,
            10644, 13025, 13364, 13832, 9482, 9970, 3393, 4848, 539, 1506, 48, 266, 0, 25,
        ]
    )
    atomic_nb: List[int] = field(default_factory=lambda: [1, 6, 7, 8, 9])
    bins: int = 9
    pos_scale: float = 1.0
    num_workers: int = 4
    limit: Optional[int] = None        # cap the training split (CV / debug)
    fold: int = -1                     # >=0: use fold k of n_folds as validation
    n_folds: int = 3
    valid_limit: Optional[int] = 512   # cap on the validation split (CV is cheap)
    valid_batches: int = 8             # batches used for one validation estimate
    batch_size: int = 64             # per GPU, identical for the three methods
    eval_num: int = 1024
    eval_batch_size: int = 64


@dataclass
class ModelConfig:
    """Model (EGNN backbone + BFN head) configuration.

    Attributes:
        in_node_nf: dimension of the Gaussian basis embedding of the charge
                    (width of each atom input, excluding the time column).
        hidden_nf: EGNN hidden dimension.
        n_layers: number of equivariant blocks.
        attention: whether to enable edge attention (off by default, matching
                   the paper).
        sigma1_coord: sigma1 of the coordinate modality (a core hyperparameter
                      of the paper).
        sigma1_charges: sigma1 of the charge modality.
        t_min: time lower bound; every t is clamped to >= t_min.
        charge_clamp: whether to clamp the reconstructed mu_hc/sigma_hc for
                      numerical stability.
        mu_clamp: clamp range for theta (mu_x, mu_c).
        loss_space: "eps" or "x".
            - "x"  : exactly paper Eq.19, in data space: w(t)*||x - x_hat||^2;
            - "eps": mathematically equivalent but numerically more stable,
                     in noise space: w_eps(t)*||eps - eps_hat||^2.
            The two are identical (see docs section 5.3), default eps.
    """

    in_node_nf: int = 16
    hidden_nf: int = 256
    n_layers: int = 9
    attention: bool = False
    sigma1_coord: float = 1e-3
    sigma1_charges: float = 0.15
    t_min: float = 1e-4
    charge_clamp: bool = True
    mu_clamp: float = 10.0
    loss_space: str = "eps"


@dataclass
class TrainConfig:
    """Training/distributed/logging configuration.

    Attributes:
        epochs: number of training epochs.
        lr / weight_decay: AdamW hyperparameters.
        grad_clip: global gradient-norm clipping threshold (<=0 disables).
        skip_bad_step: whether to drop a step when the gradient is non-finite
                       (a safety net together with gradient clipping).
        accum_steps: gradient accumulation steps (effectively enlarges batch).
        ema_decay: decay rate of the parameter EMA.
        amp: "none" | "bf16" | "fp16"; bf16 does not need GradScaler.
        loss_scale: initial loss scale for fp16 (used only when amp=fp16).
        seed: random seed (each rank uses seed+rank).
        log_every: print the training loss every this many steps.
        eval_every: run sampling evaluation every this many epochs (rank0 only).
        ckpt_every: save a checkpoint every this many epochs (rank0 only).
        sample_steps: number of BFN sampling steps at evaluation.
        out_dir: output directory (checkpoints, logs, generated molecules).
        resume: whether to resume from out_dir/last.pt.
    """

    max_steps: int = 10000        # optimisation budget (fair comparison)
    lr: float = 1e-4
    warmup_ratio: float = 0.05    # linear warm-up, then a constant lr
    weight_decay: float = 1e-12
    grad_clip: float = 5.0
    skip_bad_step: bool = True
    accum_steps: int = 1
    ema_decay: float = 0.0           # 0 disables EMA: raw weights are evaluated
    amp: str = "bf16"
    loss_scale: float = 1024.0
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
    sample_steps: int = 1000
    out_dir: str = "runs/geobfn_qm9"
    resume: bool = False


@dataclass
class Config:
    """Top-level configuration: data / model / train sections."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict (for logging and checkpoints)."""
        return asdict(self)


# --------------------------------------------------------------------------- #
# Loading and merging
# --------------------------------------------------------------------------- #
def _build(cls, value: Any):
    """Recursively put a dict into a dataclass and reject unknown fields."""
    if value is None:
        return cls()
    if not isinstance(value, dict):
        raise TypeError(f"{cls.__name__} expects a dict, got {type(value)}")
    known = {f.name: f.type for f in fields(cls)}
    unknown = set(value) - set(known)
    if unknown:
        raise KeyError(f"{cls.__name__} has unknown fields: {sorted(unknown)}")
    obj = cls()
    for k, v in value.items():
        cur = getattr(obj, k)
        if is_dataclass(cur) and isinstance(v, dict):
            setattr(obj, k, _build(type(cur), v))
        else:
            setattr(obj, k, v)
    return obj


def _read_file(path: str) -> Dict[str, Any]:
    """Read yaml or json."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml"):
        import yaml  # lazy import: without PyYAML, only yaml configs raise an error

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"unsupported config format: {path}")


def apply_overrides(cfg: Config, overrides: List[str]) -> Config:
    """Support overrides of the form `--set train.lr=1e-4 data.batch_size=32`.

    Implementation: walk down the '.'-separated path, then convert the string
    to a number/bool/list with ast.literal_eval.
    """
    import ast

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be in key=value form: {item}")
        key, raw = item.split("=", 1)
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            if not hasattr(node, p):
                raise KeyError(f"config path does not exist: {key}")
            node = getattr(node, p)
        leaf = parts[-1]
        if not hasattr(node, leaf):
            raise KeyError(f"config path does not exist: {key}")
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


def load_config(path: str | None = None, overrides: List[str] | None = None) -> Config:
    """Load configuration: defaults -> file -> command-line overrides."""
    cfg = Config()
    if path:
        raw = _read_file(path)
        for section in ("data", "model", "train"):
            if section in raw and isinstance(raw[section], dict):
                setattr(cfg, section, _build(type(getattr(cfg, section)), raw[section]))
        unknown = set(raw) - set(SECTIONS) - set(SHARED_KEYS)
        if unknown:
            raise KeyError(f"config file has unknown sections: {sorted(unknown)}")
    return apply_overrides(cfg, overrides or [])


def save_config(cfg: Config, path: str) -> None:
    """Write the configuration to json (next to the checkpoint, for reproducibility)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)


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
