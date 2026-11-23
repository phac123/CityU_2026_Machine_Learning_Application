# =============================================================================
# FILE ROLE
#   Package entry point for molgen.geobfn: hoists the common classes/functions to
#   package level so that `from geobfn import GeoBFN` works.
#   It also states what the package actually does, so that a newcomer can grasp
#   the repository boundary from this single file.
#
# IMPLEMENTATION
#   Re-exports only, no logic; this keeps the import order clear and avoids
#   circular dependencies.
#   Note: metrics depends on RDKit, so it is **not** imported at package level
#   (training does not need RDKit).
# =============================================================================
"""molgen.geobfn: a compact, distributed-training-capable reproduction of GeoBFN.

Module layout (one-to-one mapping with the paper):
    config      : hyperparameters (sigma1, K, number of layers, ...)
    data        : QM9 -> dynamic batches (zero centroid + charge bin encoding)
    egnn        : equivariant backbone Phi (paper Eq.9)
    bfn         : Bayesian flow p_F, output distribution p_O, Eq.19 loss,
                  Fig.2 sampling
    ema         : exponential moving average of parameters
    metrics     : RDKit metrics (imported only during evaluation)
    distributed : DDP initialization and cross-GPU communication
"""

from .bfn import GeoBFN
from .config import Config, DataConfig, ModelConfig, TrainConfig, load_config
from .data import EvalDataset, QM9Dataset, SyntheticDataset, collate, to_device
from .ema import EMA
from .egnn import EGNN

__all__ = [
    "Config",
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "load_config",
    "QM9Dataset",
    "SyntheticDataset",
    "EvalDataset",
    "collate",
    "to_device",
    "EGNN",
    "GeoBFN",
    "EMA",
]
