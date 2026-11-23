"""Generation / evaluation for the LLM branch.

    python evaluate.py method=llm eval.ckpt=/workspace/logs/.../last.pt \
                       eval.num=1024 eval.conformers=true

Samples SMILES from the checkpoint, reports validity / uniqueness / novelty on the
strings, and---when ``eval.conformers=true``---embeds the accepted molecules with
ETKDGv3 + MMFF so that the same 3D metrics as the two geometric branches can be
computed and the molecules can be written as XYZ.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import torch
from omegaconf import DictConfig

from .analysis import analyse, append_log
from .conformer import save_xyz, smiles_list_to_3d
from .config import Config, _build
from .model import MolGPT
from .tokenizer import SmilesTokenizer


def load_model(ckpt_path: str, device: torch.device):
    """Rebuild the tokeniser, the network and the weights from a checkpoint."""
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config()
    for section in ("data", "model", "train"):
        if section in payload.get("config", {}):
            setattr(cfg, section,
                    _build(type(getattr(cfg, section)), payload["config"][section]))
    tokenizer = SmilesTokenizer(payload["itos"])
    model = MolGPT(
        vocab_size=tokenizer.vocab_size,
        max_len=cfg.data.max_len,
        d_model=cfg.model.d_model,
        n_head=cfg.model.n_head,
        n_layer=cfg.model.n_layer,
        dropout=cfg.model.dropout,
        n_cond=cfg.data.n_cond_bins,
        tie_embeddings=cfg.model.tie_embeddings,
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    print(f"[llm] loaded {ckpt_path} (vocab={tokenizer.vocab_size}, "
          f"{model.num_parameters() / 1e6:.2f}M parameters)")
    return cfg, tokenizer, model


def device_of_this_process() -> torch.device:
    """The CUDA device of this process (LOCAL_RANK when it is set)."""
    if torch.cuda.is_available():
        return torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    return torch.device("cpu")


def evaluate(hydra_cfg: DictConfig) -> Dict[str, float]:
    """Sample strings, score them, optionally embed in 3D; returns the metrics."""
    ev = hydra_cfg.eval
    if not ev.ckpt:
        raise ValueError("evaluation needs eval.ckpt=/path/to/checkpoint.pt")

    device = device_of_this_process()
    cfg, tokenizer, model = load_model(str(ev.ckpt), device)
    if ev.seed is not None:
        torch.manual_seed(int(ev.seed))

    num = int(ev.num if ev.num is not None else cfg.train.sample_num)
    temperature = float(ev.temperature if ev.temperature is not None
                        else cfg.train.sample_temperature)
    top_k = int(ev.top_k if ev.top_k is not None else cfg.train.sample_top_k)
    cond = [int(ev.cond)] if ev.cond is not None else None

    ckpt_dir = os.path.dirname(os.path.abspath(str(ev.ckpt)))
    metrics, smiles = analyse(cfg, model, tokenizer, device, num=num, out_dir=ckpt_dir,
                              temperature=temperature, top_k=top_k, cond=cond)
    print(f"[llm] sampled {len(smiles)} strings, "
          f"validity={metrics['validity']:.3f} uniqueness={metrics['uniqueness']:.3f}")

    if ev.out:
        os.makedirs(os.path.dirname(str(ev.out)) or ".", exist_ok=True)
        with open(str(ev.out), "w", encoding="utf-8") as f:
            f.write("\n".join(smiles))
        print(f"[llm] wrote {ev.out}")

    if ev.conformers:
        from .chem import canonical_smiles, is_valid_smiles

        unique = sorted({c for c in (canonical_smiles(s) for s in smiles
                                     if is_valid_smiles(s)) if c})
        samples, failed = smiles_list_to_3d(unique, seed=int(ev.seed or 0))
        print(f"[llm] embedded {len(samples)}/{len(unique)} molecules in 3D "
              f"({len(failed)} failed)")
        if samples:
            from . import metrics as metrics_3d

            metrics.update({f"3d_{k}": v for k, v in
                            metrics_3d.evaluate(samples, reference_smiles=None).items()})
            if ev.xyz_out:
                save_xyz(str(ev.xyz_out), samples)
                print(f"[llm] wrote {len(samples)} conformers to {ev.xyz_out}")

    # the step of the checkpoint labels the log line
    ckpt_step = int(torch.load(str(ev.ckpt), map_location="cpu",
                               weights_only=False).get("step", -1))
    append_log(os.path.join(ckpt_dir, "eval_log.jsonl"),
               {"step": ckpt_step, "source": "evaluate.py", **metrics})

    if ev.json_out:
        os.makedirs(os.path.dirname(str(ev.json_out)) or ".", exist_ok=True)
        with open(str(ev.json_out), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    printable = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}
    print("[llm] " + json.dumps(printable))
    return metrics
