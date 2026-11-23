"""Generation / evaluation for the EDM branch.

    python evaluate.py method=edm eval.ckpt=/workspace/logs/.../best.pt \
                       eval.num=1024 eval.steps=1000 eval.eta=0.0

The architecture, the noise schedule and the data settings are restored from the
checkpoint; only the sampling knobs (number of molecules, number of reverse steps,
``eta``) and the output paths are taken from the command line.  ``eta=0`` is the
deterministic DDIM sampler, ``eta=1`` the ancestral one.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import torch
from omegaconf import DictConfig

from .config import Config, _build
from .diffusion import EDMDiffusion
from .analysis import analyse, append_log


def load_model(ckpt_path: str, device: torch.device, use_ema: bool = True
               ) -> Tuple[Config, EDMDiffusion]:
    """Rebuild the network from a checkpoint and load the (EMA) weights."""
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config()
    for section in ("data", "model", "diffusion", "train"):
        if section in payload.get("config", {}):
            setattr(cfg, section,
                    _build(type(getattr(cfg, section)), payload["config"][section]))
    model = EDMDiffusion(cfg.model, cfg.diffusion, n_types=len(cfg.data.atomic_nb),
                         atomic_nb=cfg.data.atomic_nb).to(device)
    state = payload.get("ema") if (use_ema and payload.get("ema")) else payload["model"]
    current = model.state_dict()
    merged = {k: (state[k].to(dtype=current[k].dtype, device=device) if k in state else current[k])
              for k in current}
    model.load_state_dict(merged, strict=True)
    model.eval()
    print(f"[edm] loaded {ckpt_path} ({'EMA' if state is payload.get('ema') else 'raw'} weights)")
    return cfg, model


def device_of_this_process() -> torch.device:
    """The CUDA device of this process (LOCAL_RANK when it is set)."""
    if torch.cuda.is_available():
        return torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    return torch.device("cpu")


def evaluate(hydra_cfg: DictConfig) -> Dict[str, float]:
    """Sample, score and optionally write molecules; returns the metric dict."""
    ev = hydra_cfg.eval
    if not ev.ckpt:
        raise ValueError("evaluation needs eval.ckpt=/path/to/checkpoint.pt")

    device = device_of_this_process()
    cfg, model = load_model(str(ev.ckpt), device, use_ema=bool(ev.use_ema))
    if ev.num is not None:
        cfg.data.eval_num = int(ev.num)
    if ev.steps is not None:
        cfg.diffusion.sample_steps = int(ev.steps)
    if ev.eta is not None:
        cfg.diffusion.sample_eta = float(ev.eta)
    if ev.seed is not None:
        torch.manual_seed(int(ev.seed))

    ckpt_dir = os.path.dirname(os.path.abspath(str(ev.ckpt)))
    metrics, samples = analyse(cfg, model, device, out_dir=ckpt_dir,
                               seed=int(ev.seed or cfg.train.seed))
    # the step of the checkpoint labels the log line (cheap: the file is already in
    # the page cache after load_model)
    ckpt_step = int(torch.load(str(ev.ckpt), map_location="cpu",
                               weights_only=False).get("step", -1))
    append_log(os.path.join(ckpt_dir, "eval_log.jsonl"),
               {"step": ckpt_step, "source": "evaluate.py", **metrics})

    if ev.out:
        from .analysis import write_samples

        write_samples(cfg, samples, str(ev.out))
        print(f"[edm] wrote {len(samples)} molecules to {ev.out}")
    if ev.json_out:
        os.makedirs(os.path.dirname(str(ev.json_out)) or ".", exist_ok=True)
        with open(str(ev.json_out), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    printable = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}
    print("[edm] " + json.dumps(printable))
    return metrics
