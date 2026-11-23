"""Optional adapter: drive a pretrained HuggingFace causal LM as a SMILES generator.

The course project trains its own small Transformer (``molgpt``), but the same
generation protocol applies to a pretrained model.  This adapter exposes the
identical ``generate_smiles`` interface so the evaluation harness does not care
which backbone produced the strings, which is exactly the "LLM as a black-box
molecule generator" baseline.

It is optional on purpose: importing this module without ``transformers``
installed raises a clear error only when the class is instantiated.
"""

from __future__ import annotations

from typing import List, Optional

import torch

DEFAULT_PROMPT = "SMILES: "


class HFBackbone:
    """Thin wrapper around ``AutoModelForCausalLM`` for SMILES continuation.

    Args:
        model_name: a HuggingFace hub id or local directory of a causal LM.
        device: target device string, e.g. "cuda" or "cpu".
        prompt: textual prompt that precedes every SMILES string.
        dtype: torch dtype for the weights ("float32", "bfloat16", ...).

    Example:
        backbone = HFBackbone("gpt2", "cpu")
        smiles = backbone.generate_smiles(64, max_new_tokens=48, temperature=0.9)
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        prompt: str = DEFAULT_PROMPT,
        dtype: str = "float32",
    ):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "HFBackbone requires the 'transformers' package; install it or use "
                "the from-scratch molgpt model instead."
            ) from exc

        torch_dtype = getattr(torch, dtype, torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.prompt = prompt
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @torch.no_grad()
    def generate_smiles(
        self,
        num_samples: int,
        max_new_tokens: int = 64,
        temperature: float = 0.9,
        top_p: float = 0.95,
        batch_size: int = 32,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Sample ``num_samples`` continuations and cut each at the first newline.

        A textual prompt is a much weaker conditioning signal than a dedicated
        tokenizer, so this baseline mainly demonstrates *why* the project trains
        a chemistry-aware model rather than freeing a general-purpose LLM.
        """
        if seed is not None:
            torch.manual_seed(seed)
        out: List[str] = []
        done = 0
        while done < num_samples:
            cur = min(batch_size, num_samples - done)
            enc = self.tokenizer([self.prompt] * cur, return_tensors="pt",
                                 padding=True).to(self.device)
            gen = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            text = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
            for t in text:
                body = t[len(self.prompt):] if t.startswith(self.prompt) else t
                out.append(body.split("\n")[0].strip())
            done += cur
        return out
