"""E(3)-equivariant graph neural network used by the EDM branch.

The backbone is the EGNN of Satorras et al. (2021) as adapted for diffusion by
Hoogeboom et al. (2022):

* node features carry the (noisy) atom-type information,
* edge features carry the squared distance,
* the diffusion time t enters through a sinusoidal embedding that is *added* to
  both the node and the edge features, so a single network serves all t,
* the coordinate update uses the normalised relative vector (x_i - x_j)/||.|| and
  a scalar gate, which makes the vector field exactly equivariant to rotations and
  invariant to translations.

The network returns two things: atom-type logits (for the discrete reverse step)
and an equivariant displacement field that is interpreted as the noise estimate
eps_hat for the coordinates.  The displacement is accumulated over layers starting
from zero, so at initialisation eps_hat ~ 0 and the objective starts at the scale
of the data, which trains stably without any output rescaling.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Row-wise sum of ``src`` into ``dim_size`` buckets (no torch_scatter needed)."""
    out = src.new_zeros((dim_size, src.size(1)))
    return out.index_add_(0, index, src)


def coord_diff(x: torch.Tensor, edge_index: torch.Tensor, norm_constant: float = 1.0):
    """Squared distances (invariant) and normalised direction vectors (equivariant)."""
    row, col = edge_index
    diff = x[row] - x[col]
    d2 = (diff * diff).sum(dim=-1, keepdim=True)
    dist = torch.sqrt(d2 + 1e-8)
    unit = diff / (dist + norm_constant)
    return d2, unit


class SinusoidalTimeEmbedding(nn.Module):
    """Fixed sinusoidal embedding of the continuous time t, followed by an MLP."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.dim = dim
        half = dim // 2
        # frequencies 1 / 10000^(k/half), stored so they move with .to(device)
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / max(half - 1, 1))
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(nn.Linear(dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [N, 1] (or [N]) -> [N, out_dim]."""
        t = t.view(-1, 1) * 1000.0                      # scale [0,1] -> [0,1000]
        args = t * self.freqs.view(1, -1)               # [N, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.size(-1) < self.dim:                     # odd dim: pad one column
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)


class EGNNLayer(nn.Module):
    """One message-passing layer: update h, then update x with an equivariant gate."""

    def __init__(self, hidden_nf: int, time_emb_dim: int, act=nn.SiLU(),
                 attention: bool = False, tanh: bool = True,
                 coords_range: float = 10.0, norm_constant: float = 1.0):
        super().__init__()
        self.attention = attention
        self.tanh = tanh
        self.coords_range = coords_range
        self.norm_constant = norm_constant

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_nf + 1 + time_emb_dim, hidden_nf), act,
            nn.Linear(hidden_nf, hidden_nf), act,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_nf, hidden_nf), act,
            nn.Linear(hidden_nf, hidden_nf),
        )
        if attention:
            # gated attention (sigmoid, no softmax normalisation) as in EGNN
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())
        coord_head = nn.Linear(hidden_nf, 1, bias=False)
        nn.init.xavier_uniform_(coord_head.weight, gain=1e-3)
        self.coord_mlp = nn.Sequential(nn.Linear(hidden_nf, hidden_nf), act, coord_head)

    def forward(self, h: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor,
                temb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index
        d2, unit = coord_diff(x, edge_index, self.norm_constant)

        m = self.edge_mlp(torch.cat([h[row], h[col], d2, temb[row]], dim=-1))
        if self.attention:
            m = m * self.att_mlp(m)
        agg = scatter_add(m, row, h.size(0))
        h = h + self.node_mlp(torch.cat([h, agg], dim=-1))

        w = self.coord_mlp(m)                                  # [E,1] scalar gate
        if self.tanh:
            w = torch.tanh(w) * self.coords_range
        x = x + scatter_add(unit * w, row, x.size(0))           # [N,3]
        return h, x


class EGNN(nn.Module):
    """Stack of EGNN layers plus the two output heads.

    Args:
        in_node_nf: size of the atom-type embedding.
        out_node_nf: number of atom-type classes (the categorical diffusion state).
        hidden_nf / n_layers: width and depth.
        time_embed_dim: width of the time embedding added to nodes and edges.
    """

    def __init__(
        self,
        in_node_nf: int,
        out_node_nf: int,
        hidden_nf: int = 256,
        n_layers: int = 9,
        n_sublayers: int = 1,
        time_embed_dim: int = 128,
        act=nn.SiLU(),
        attention: bool = False,
        tanh: bool = True,
        coords_range: float = 10.0,
        norm_constant: float = 1.0,
    ):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim, time_embed_dim)
        # The blocks consume the raw time embedding (time_embed_dim), but it is added
        # to the node features, which live at width hidden_nf: project it, so that the
        # two widths are independent (they used to have to be equal).
        self.time_to_hidden = nn.Linear(time_embed_dim, hidden_nf)
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.blocks = nn.ModuleList(
            [
                EGNNLayer(
                    hidden_nf, time_embed_dim, act=act, attention=attention,
                    tanh=tanh, coords_range=coords_range / max(n_layers, 1),
                    norm_constant=norm_constant,
                )
                for _ in range(n_layers)
            ]
        )
        self.out = nn.Linear(hidden_nf, out_node_nf)

    def forward(self, h: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor,
                t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args: h [N,C] one-hot atom types, x [N,3], edge_index [2,E], t [N,1].

        Returns:
            logits [N,K] for the discrete state and eps_hat [N,3], an
            E(3)-equivariant estimate of the noise added to the coordinates.
        """
        temb = self.time_embed(t)                               # [N, T]
        h = self.embedding(h) + self.time_to_hidden(temb)
        x0 = x
        for blk in self.blocks:
            h, x = blk(h, x, edge_index, temb)
        return self.out(h), x - x0
