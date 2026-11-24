"""Shape test of the two neural backbones, with hand-made tensors.

This is the test that would have caught the EDM crash

    RuntimeError: The size of tensor a (256) must match the size of tensor b (128)

``EGNN.forward`` adds the time embedding to the node features, so the two widths must
agree; here a tiny model is built and one forward is run, asserting the output shapes.
It needs torch but no dataset, no GPU and no checkpoint, and it skips itself when torch
is not installed.
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    import torch
except ImportError:                                            # pragma: no cover
    print("SKIPPED: torch is not installed in this environment")
    raise SystemExit(0)

from molgen.edm.egnn import EGNN, EGNNParameters              # noqa: E402


def check_edm_egnn() -> None:
    """hidden_nf and time_embed_dim must be independent (projection, not equality)."""
    torch.manual_seed(0)
    n_atoms, n_edges, n_types = 6, 10, 5
    for hidden_nf, time_embed_dim in ((256, 128), (64, 64), (32, 128)):
        model = EGNN(EGNNParameters(in_node_nf=n_types, out_node_nf=n_types,
                                    hidden_nf=hidden_nf, n_layers=2,
                                    time_embed_dim=time_embed_dim))
        h = torch.zeros(n_atoms, n_types).scatter_(1, torch.randint(0, n_types, (n_atoms, 1)), 1.0)
        x = torch.randn(n_atoms, 3)
        edge_index = torch.randint(0, n_atoms, (2, n_edges))
        t = torch.rand(n_atoms, 1)
        logits, eps_hat = model(h, x, edge_index, t)
        print(f"  EGNN hidden={hidden_nf} time={time_embed_dim}: "
              f"logits {tuple(logits.shape)} eps {tuple(eps_hat.shape)}")
        assert logits.shape == (n_atoms, n_types), logits.shape
        assert eps_hat.shape == (n_atoms, 3), eps_hat.shape
        assert torch.isfinite(logits).all() and torch.isfinite(eps_hat).all()


def check_llm_model() -> None:
    """The language model returns logits [B, L, V] and a finite loss."""
    from molgen.llm.model import MolGPT

    torch.manual_seed(0)
    vocab, length, batch = 11, 7, 3
    model = MolGPT(vocab_size=vocab, max_len=length, d_model=16, n_head=2, n_layer=1,
                   dropout=0.0, n_cond=0, tie_embeddings=True)
    ids = torch.randint(0, vocab, (batch, length))
    logits, loss = model(input_ids=ids[:, :-1], targets=ids[:, 1:],
                         loss_mask=torch.ones(batch, length - 1))
    print(f"  MolGPT: logits {tuple(logits.shape)} loss {float(loss):.4f}")
    assert logits.shape == (batch, length - 1, vocab), logits.shape
    assert loss is not None and torch.isfinite(loss)


def main() -> None:
    print("EDM EGNN widths:")
    check_edm_egnn()
    print("LLM backbone:")
    check_llm_model()
    print("\nMODEL SHAPE CHECKS PASSED")


if __name__ == "__main__":
    main()

