# =============================================================================
# FILE ROLE
#   Exponential moving average (EMA) of parameters. Training keeps a "shadow"
#   copy of the weights; evaluation/export uses the shadow weights instead of
#   the training weights, which usually improves generation quality markedly.
#
# IMPLEMENTATION
#   shadow <- decay*shadow + (1-decay)*theta, with decay=0.9999 (roughly the
#   average over the last 10k steps). This implementation handles several
#   common pitfalls:
#     1) EMA is applied only to **floating-point** parameters (integer buffers
#        such as k_c do not need it);
#     2) shadow is always stored as float32: under AMP the parameters may be
#        bf16, and averaging them directly loses precision;
#     3) state_dict()/load_state_dict() are provided separately so that the
#        checkpoint can store the EMA weights;
#     4) copy_to() uses no_grad + dtype conversion and restores the weights
#        right after evaluation, so training weights are never polluted.
# =============================================================================
"""Exponential moving average (EMA)."""

from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn


class EMA:
    """Maintains a moving average of the model parameters.

    Args:
        model: model to average (pass model.module under DDP)
        decay: decay rate, larger is smoother
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self._init_from(model)

    @torch.no_grad()
    def _init_from(self, model: nn.Module) -> None:
        """Initialize the shadow weights from the current parameters (cast to float32)."""
        self.shadow = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if torch.is_floating_point(v)
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """One EMA update step."""
        d = self.decay
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                continue
            s = self.shadow[k]
            if s.shape != v.shape:      # shape changed (e.g. new head) -> reset
                self.shadow[k] = v.detach().clone().float()
                continue
            s.mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Write the shadow weights into the model (call before evaluation)."""
        state = model.state_dict()
        new_state = {}
        for k, v in state.items():
            if k in self.shadow:
                new_state[k] = self.shadow[k].to(dtype=v.dtype, device=v.device)
            else:
                new_state[k] = v
        model.load_state_dict(new_state, strict=True)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """State for the checkpoint (deep copy, so later updates do not mutate it)."""
        return {k: v.detach().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        """Restore from a checkpoint."""
        self.shadow = {k: v.detach().clone().float() for k, v in state.items()}
