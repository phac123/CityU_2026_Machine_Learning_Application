"""edm - E(3)-equivariant diffusion model for 3D molecule generation.

Public API:
    EDMDiffusion   the model: noise schedule, forward process, loss, sampler
    EGNN           the equivariant backbone
    QM9Dataset     npz loader producing (pos, h) pairs
    EMA            parameter averaging used at sampling time

Self-contained by design: nothing here imports from the GeoBFN or LLM packages,
so each of the three methods can be trained and reviewed on its own.
"""

from .config import Config, DiffusionConfig, ModelConfig, load_config, save_config
from .data import QM9Dataset, collate, zero_com
from .diffusion import EDMDiffusion
from .egnn import EGNN
from .ema import EMA

__all__ = [
    "Config",
    "ModelConfig",
    "DiffusionConfig",
    "load_config",
    "save_config",
    "QM9Dataset",
    "collate",
    "zero_com",
    "EDMDiffusion",
    "EGNN",
    "EMA",
]
