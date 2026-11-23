"""Exponential moving average of model parameters (EDM branch).

Sampling uses the averaged weights, which is standard for diffusion models: the
averaged network is smoother and produces noticeably better molecules.  The
shadow copy is kept in float32 so that bf16/fp16 training does not lose precision
in the average, and integer buffers (none here, but kept for safety) are ignored.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class EMA:
    """Maintains ``shadow <- decay * shadow + (1 - decay) * param``."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self._init_from(model)

    @torch.no_grad()
    def _init_from(self, model: nn.Module) -> None:
        self.shadow = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if torch.is_floating_point(v)
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """One EMA step; keys whose shape changed are re-initialised."""
        d = self.decay
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                continue
            s = self.shadow[k]
            if s.shape != v.shape:
                self.shadow[k] = v.detach().clone().float()
                continue
            s.mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Overwrite the live parameters with the averaged ones (for evaluation)."""
        state = model.state_dict()
        new_state = {}
        for k, v in state.items():
            if k in self.shadow:
                new_state[k] = self.shadow[k].to(dtype=v.dtype, device=v.device)
            else:
                new_state[k] = v
        model.load_state_dict(new_state, strict=True)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.detach().clone().float() for k, v in state.items()}
