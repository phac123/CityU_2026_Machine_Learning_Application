# =============================================================================
# FILE ROLE
#   The EGNN (E(n) Equivariant Graph Neural Network) backbone -- i.e. the
#   network module Phi of the paper.
#   Input: node features h [N,C], coordinates x [N,3], edges edge_index [2,E]
#   Output: updated features h' [N,out_node_nf] (out=2 here, for the charge
#   Gaussian) and coordinates x' [N,3]
#
# IMPLEMENTATION
#   Three layers, stacked:
#     GCL                    : edge-message MLP + node residual update (features
#                              only, coordinates untouched)
#     EquivariantCoordUpdate : x_i' = x_i + sum_j (x_i-x_j)/(|x_i-x_j|+c) * phi_x(m_ij)
#     EGNNBlock              : each layer computes distances once, then calls
#                              the two modules above in order
#     EGNN                   : embedding -> L blocks -> output head (on h only)
#
#   * Why equivariance holds (the concrete form of paper Eq.9):
#     the coordinate update only involves relative coordinate differences
#     (x_i - x_j) and distances; the former becomes R(x_i-x_j) under rotation
#     and is unchanged by translation, the latter is fully invariant; phi_x
#     outputs a scalar. Hence overall
#         Phi(R theta_x + t, .) = R Phi(theta_x, .) + t
#     Stacking L layers keeps the map equivariant (a composition of equivariant
#     maps is equivariant).
#
#   * One engineering choice in this file: no dependency on torch_scatter /
#     torch_geometric; scatter-sum is hand-written with index_add_. This cuts
#     dependencies and makes "message aggregation" explicit.
#
# PAPER MAPPING
#   Eq.9   : the SE(3) equivariance constraint on Phi -- guaranteed by the
#            coordinate update in this file;
#   Sec.3.2: the network outputs two channels, "mu and ln sigma of the charge"
#            -- guaranteed by out_node_nf=2.
# =============================================================================
"""EGNN backbone (Phi)."""

from __future__ import annotations

import torch
import torch.nn as nn


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Add src rows onto dim_size rows per index (minimal torch_scatter.scatter_add).

    Args:
        src: [E, C] vector per edge (or per element).
        index: [E] target row indices (0..dim_size-1).
        dim_size: number of output rows (usually the number of nodes N).
    Returns:
        [dim_size, C]
    """
    out = src.new_zeros((dim_size, src.size(1)))
    return out.index_add_(0, index, src)


def coord_diff(x: torch.Tensor, edge_index: torch.Tensor, norm_constant: float = 1.0):
    """Compute per-edge "squared distance" and "normalized relative coordinate".

    Args:
        x: [N,3] coordinates
        edge_index: [2,E]
        norm_constant: normalization constant c, prevents 1/r blow-up when two
                       atoms are extremely close.
    Returns:
        d2:   [E,1]  ||x_i - x_j||^2 (rotation/translation invariant, edge feature)
        unit: [E,3]  (x_i - x_j) / (||x_i - x_j|| + c) (rotation-equivariant,
                     translation-invariant)
    """
    row, col = edge_index
    diff = x[row] - x[col]
    d2 = (diff * diff).sum(dim=-1, keepdim=True)
    dist = torch.sqrt(d2 + 1e-8)
    unit = diff / (dist + norm_constant)
    return d2, unit


class GCL(nn.Module):
    """Graph convolution layer: updates node features h only, never coordinates.

    Formulas:
        m_ij = phi_e([h_i || h_j || e_ij])
        h_i' = h_i + phi_h([h_i || sum_j m_ij])
    Both phi are two-layer MLP + SiLU; the residual connection keeps deep
    layers trainable.
    """

    def __init__(self, hidden_nf: int, edge_in_nf: int = 1, act=nn.SiLU(), attention: bool = False):
        super().__init__()
        self.attention = attention
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_nf + edge_in_nf, hidden_nf), act,
            nn.Linear(hidden_nf, hidden_nf), act,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_nf, hidden_nf), act,
            nn.Linear(hidden_nf, hidden_nf),
        )
        if attention:
            # gated attention (sigmoid, no softmax normalization), as in original EGNN
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_feat: torch.Tensor):
        row, col = edge_index
        mij = self.edge_mlp(torch.cat([h[row], h[col], edge_feat], dim=-1))  # [E,hidden]
        if self.attention:
            mij = mij * self.att_mlp(mij)
        agg = scatter_add(mij, row, h.size(0))                              # [N,hidden]
        h = h + self.node_mlp(torch.cat([h, agg], dim=-1))                  # residual
        return h, mij


class EquivariantCoordUpdate(nn.Module):
    """Equivariant coordinate update layer (the "E" in EGNN).

        x_i' = x_i + sum_j  unit_ij * phi_x(m_ij)

    The last layer of phi_x is **deliberately initialized very small**
    (xavier gain=0.001): coordinates barely move early in training, so the
    structure is not destroyed immediately (a trick from the EGNN paper).
    """

    def __init__(self, hidden_nf: int, edge_in_nf: int = 1, act=nn.SiLU(), tanh: bool = False,
                 coords_range: float = 15.0):
        super().__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        last = nn.Linear(hidden_nf, 1, bias=False)
        nn.init.xavier_uniform_(last.weight, gain=1e-3)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_nf + edge_in_nf, hidden_nf), act,
            nn.Linear(hidden_nf, hidden_nf), act,
            last,
        )

    def forward(self, h: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor,
                edge_feat: torch.Tensor, unit: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        w = self.mlp(torch.cat([h[row], h[col], edge_feat], dim=-1))  # [E,1] scalar weight
        if self.tanh:
            w = torch.tanh(w) * self.coords_range
        trans = unit * w                                              # [E,3]
        return x + scatter_add(trans, row, x.size(0))                 # aggregate to nodes by row


class EGNNBlock(nn.Module):
    """One equivariant block = several GCLs (update h) + one coordinate update (update x)."""

    def __init__(self, hidden_nf: int, n_sublayers: int = 1, act=nn.SiLU(),
                 attention: bool = False, tanh: bool = False, coords_range: float = 15.0,
                 norm_constant: float = 1.0):
        super().__init__()
        self.norm_constant = norm_constant
        self.gcls = nn.ModuleList(
            [GCL(hidden_nf, edge_in_nf=1, act=act, attention=attention) for _ in range(n_sublayers)]
        )
        self.coord_update = EquivariantCoordUpdate(
            hidden_nf, edge_in_nf=1, act=act, tanh=tanh, coords_range=coords_range
        )

    def forward(self, h: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor):
        # use current coordinates for distances/directions; the only geometry read here
        d2, unit = coord_diff(x, edge_index, self.norm_constant)
        for gcl in self.gcls:
            h, _ = gcl(h, edge_index, d2)
        x = self.coord_update(h, x, edge_index, d2, unit)
        return h, x

class EGNN(nn.Module):
    """Full backbone: embedding -> L x EGNNBlock -> output head.
    Note the output head acts on h only (coordinates are already [N,3] and need
    no further transform), which is why "the coordinate branch has no separate
    output layer while the h branch has a 2-D output".
    """

    def __init__(
        self,
        in_node_nf: int,
        hidden_nf: int = 256,
        out_node_nf: int = 2,
        n_layers: int = 9,
        n_sublayers: int = 1,
        act=nn.SiLU(),
        attention: bool = False,
        tanh: bool = False,
        coords_range: float = 15.0,
        norm_constant: float = 1.0,
    ):
        super().__init__()
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.blocks = nn.ModuleList(
            [
                EGNNBlock(
                    hidden_nf,
                    n_sublayers=n_sublayers,
                    act=act,
                    attention=attention,
                    tanh=tanh,
                    # per-layer coordinate update bound = coords_range / n_layers
                    coords_range=coords_range / max(n_layers, 1),
                    norm_constant=norm_constant,
                )
                for _ in range(n_layers)
            ]
        )
        self.out = nn.Linear(hidden_nf, out_node_nf)

    def forward(self, h: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor):
        h = self.embedding(h)
        for blk in self.blocks:
            h, x = blk(h, x, edge_index)
        return self.out(h), x
