# =============================================================================
# FILE ROLE
#   All of GeoBFN's mathematics lives in this one file (paper Sec.3.2-3.4):
#     1) Bayesian flow p_F for continuous variables:
#        gamma(t)=1-sigma1^(2t), theta ~ N(gamma x, gamma(1-gamma) I)
#     2) Output distribution p_O: Dirac for coordinates (point estimate),
#        Gaussian CDF over K bins for charges (Eq.15)
#     3) Training loss: the continuous-time loss of paper Eq.19 (can also switch
#        to the mathematically equivalent eps-space form)
#     4) Sampling: the recursive Bayesian update of paper Fig.2 plus the
#        nearest_center correction of Eq.20
#
# IMPLEMENTATION (one training step)
#   t ~ U(0,1) (per molecule, broadcast to atoms)
#     |-> (mu_x, gamma_x, eps_x) = bayes_flow(t, sigma1^x, x)    # coords theta
#         (mu_c, gamma_c, eps_c) = bayes_flow(t, sigma1^c, h_c)  # charges theta
#           |-> Phi: EGNN([gauss(mu_c) || t], mu_x)
#                 |-> (eps_hat_x, mu_eps, ln sigma_eps)
#                 |- x_hat = mu_x/gamma_x - sqrt((1-gamma_x)/gamma_x)*eps_hat_x
#                 |        (inverse Bayesian update)
#                 \- p_O(k) = CDF(k_r) - CDF(k_l)                (Eq.15)
#                     Phi_hc = sum_k p_O(k)*k_c                   (Phi_hc in Eq.19)
#                       |-> L = alpha_x(t)/2*||x-x_hat||^2
#                               + alpha_c(t)/2*||h_c-Phi_hc||^2
#
# WHY BOTH loss_space VALUES ARE SUPPORTED
#   The weight alpha(t)/2 = -ln sigma1 * sigma1^(-2t) in Eq.19 can reach 1e6 as
#   t->1; but substituting x - x_hat = sqrt((1-gamma)/gamma)(eps_hat - eps), the
#   loss is identically
#       (-ln sigma1)/(1-sigma1^(2t)) * ||eps - eps_hat||^2
#   whose weight only grows like 1/(2t) as t->0, far more benign numerically.
#   The two are **strictly equal per sample**, so `loss_space="eps"` (the
#   default) is not an approximation but a more stable form of the same
#   objective.
#
# PAPER MAPPING QUICK REFERENCE
#   Eq.11/12  -> bayes_flow() (closed-form equivalent) and the step-by-step
#                updates in sample()
#   Eq.13     -> bayes_flow(): gamma(t)=1-sigma1^(2t)
#   Eq.14     -> pos_pred returned by forward_theta() (the Dirac center)
#   Eq.15     -> bucket_probs() / gaussian_cdf()
#   Eq.19     -> loss()
#   Eq.20     -> nearest_center()
# =============================================================================
"""GeoBFN: Bayesian flow + EGNN backbone + Eq.19 loss + recursive sampling."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .config import ModelConfig
from .data import atomic_nb_to_bucket, zero_com
from .egnn import EGNN

def gaussian_cdf(mu: torch.Tensor, sigma: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Gaussian CDF: 0.5*(1+erf((x-mu)/(sigma*sqrt(2)))), truncated at bin edges.

    Args:
        mu:    [N,1]  reconstructed charge mean
        sigma: [N,1]  reconstructed charge standard deviation
        x:     [K]    bin edges (k_r or k_l)
    Returns:
        [N,K] CDF values; x >= 1 maps to 1, x <= -1 maps to 0.

    Why truncate: after normalization charges lie in [-1,1], while the outermost
    bin edges reach +/-(1+1/K). Without truncation the first/last bins would
    receive probability mass "outside the boundary", and sum p_O(k) would no
    longer equal 1 (in the paper this corresponds to the sentence "first and
    last bins are curated").
    """
    mu = mu.unsqueeze(1)        # [N,1,1]
    sigma = sigma.unsqueeze(1)  # [N,1,1]
    xb = x.view(1, -1, 1)       # [1,K,1]
    cdf = 0.5 * (1.0 + torch.erf((xb - mu) / (sigma * math.sqrt(2.0))))
    cdf = torch.where(xb >= 1.0, torch.ones_like(cdf), cdf)
    cdf = torch.where(xb <= -1.0, torch.zeros_like(cdf), cdf)
    return cdf.squeeze(-1)      # [N,K]

class GeoBFN(nn.Module):
    """GeoBFN model: an EGNN backbone plus BFN's analytic input/output distributions.

    Args:
        cfg: ModelConfig (sigma1, layer count, loss space, etc.)
        bins: number of charge discretization bins K
        atomic_nb: allowed atomic numbers (used to decode continuous charges
                   into elements at the end)
    """

    def __init__(self, cfg: ModelConfig, bins: int, atomic_nb=(1, 6, 7, 8, 9)):
        super().__init__()
        self.cfg = cfg
        self.bins = int(bins)
        self.atomic_nb = tuple(int(z) for z in atomic_nb)

        # ---- EGNN backbone: input = charge Gaussian basis (16) + time (1), ----
        # ---- output = charge (mu, ln sigma) ----
        self.backbone = EGNN(
            in_node_nf=cfg.in_node_nf + 1,
            hidden_nf=cfg.hidden_nf,
            out_node_nf=2,
            n_layers=cfg.n_layers,
            attention=cfg.attention,
        )

        # ---- BFN constants: bin centers/edges, Gaussian basis centers, sigma1 ----
        # register_buffer: moves with .to(device)/.half(), never updated by optimizer
        k = torch.arange(1, self.bins + 1, dtype=torch.float32)
        k_c = (2 * k - 1) / self.bins - 1.0           # k_c from Eq.15
        self.register_buffer("k_c", k_c)
        self.register_buffer("k_l", k_c - 1.0 / self.bins)  # left edge
        self.register_buffer("k_r", k_c + 1.0 / self.bins)  # right edge
        # anchors for "continuous charge value -> nearest atom type" (legal elements)
        self.register_buffer("z_anchors", atomic_nb_to_bucket(
            torch.tensor(self.atomic_nb, dtype=torch.float32), self.bins))

        start, end = -2.0, 2.0
        self.width = (end - start) / cfg.in_node_nf
        self.register_buffer("centers", torch.linspace(start, end, cfg.in_node_nf))

        self.register_buffer("sigma1_coord", torch.tensor(float(cfg.sigma1_coord)))
        self.register_buffer("sigma1_charges", torch.tensor(float(cfg.sigma1_charges)))


    def gaussian_basis(self, v: torch.Tensor) -> torch.Tensor:
        """Scalar v -> K-dimensional soft-binning feature (used as network input).

        Math: y_k(v) = 2 * softmax_k( -(v - c_k)^2 / (2w^2) ) - 1

        Steps (matching the per-line comments below):
        1. out  = (v - c_k) / w              # standardized distance z_k
        2. basis = exp(-0.5 * z_k^2)         # Gaussian kernel
        3. basis = basis / sum(basis)        # softmax normalization to a probability
        4. return 2 * basis - 1              # affine map of probability to [-1, 1]

        Property: sum_k y_k = 2 - K; the closer v is to c_k, the closer y_k is to
        +1 and the others to -1.
        """
        # step 1: z_k = (v - c_k) / w
        out = (v - self.centers) / self.width
        # step 2: b_k = exp(-z_k^2 / 2)
        basis = torch.exp(-0.5 * out ** 2)
        # step 3: normalize = softmax(-z_k^2 / 2), giving probabilities
        basis = basis / basis.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        # step 4: y_k = 2 * b_k - 1, mapped to [-1, 1]
        return basis * 2.0 - 1.0


    # Thanks to the additivity of Gaussians, gamma at time t is obtained
    # directly, and then the mean and variance at time t directly.
    # ------------------------------------------------------------------ #
    # Bayesian flow p_F
    # ------------------------------------------------------------------ #
    def bayes_flow(
        self, t: torch.Tensor, sigma1: torch.Tensor, x: torch.Tensor,
        seg: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample from the Bayesian flow p_F(theta|x;t) and return the noise eps.

        Math (closed form of paper Eq.13 in continuous time):
            gamma(t) = 1 - sigma1^(2t)
            p_F(theta|x;t) = N(gamma x, gamma(1-gamma) I)
        Reparameterization:
            theta = gamma x + sqrt(gamma(1-gamma))*eps,   eps ~ N(0,I)

        * Why zero-center eps first (rather than theta):
            Paper Theorem 3.1 requires theta^x, y^x and x to live in the
            zero-centroid subspace. If we sampled theta first and then
            subtracted the centroid, we would introduce an extra offset
            c = sqrt(gamma(1-gamma))*mean(eps), which breaks the identity
            x - x_hat = s(eps_hat - eps).
            Conversely, projecting eps onto the zero-centroid subspace first
            means (since x itself is zero-centroid) theta = gamma x +
            sqrt(gamma(1-gamma)) eps is automatically zero-centroid, and the
            identity holds exactly. This is also the precondition for the
            "x-space loss" and the "eps-space loss" being numerically fully
            equivalent (see section 5.3).

        Args:
            t: [N,1] time
            sigma1: scalar tensor, the modality's sigma1
            x: [N,D] ground truth (must already be zero-centroid)
            seg: [N] node->molecule; None means the whole batch is one molecule
        Returns:
            theta(mu) [N,D], gamma [N,1], eps [N,D]
        """
        gamma = 1.0 - sigma1 ** (2.0 * t)                      # [N,1]
        eps = torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device)
        if seg is not None:
            eps = zero_com(eps, seg)                          # noise is zero-centroid too
        mu = gamma * x + torch.sqrt(gamma * (1.0 - gamma)) * eps
        return mu, gamma, eps

    # ------------------------------------------------------------------ #
    # 2) Output distribution p_O
    # ------------------------------------------------------------------ #
    def bucket_probs(
        self, mu_c: torch.Tensor, gamma_c: torch.Tensor,
        mu_eps: torch.Tensor, log_sigma_eps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Charge branch: produce the probability p_O(k) of the K bins (Eq.15).

        First do the inverse Bayesian update, turning the network output
        (mu_eps, ln sigma_eps) back into data space (mu_hc, sigma_hc):
            mu_hc    = mu_c/gamma_c - s*mu_eps
            sigma_hc = s*exp(ln sigma_eps),   s = sqrt((1-gamma_c)/gamma_c)
        (where s comes from: x = mu/gamma - s*eps is an affine transform, so
        sigma scales by s as well.)

        Returns:
            p:      [N,K] bin probabilities
            mu_hc:  [N,1]
            sig_hc: [N,1]
        """
        s = torch.sqrt((1.0 - gamma_c) / gamma_c)
        mu_hc = mu_c / gamma_c - s * mu_eps
        sig_hc = s * torch.exp(log_sigma_eps)
        if self.cfg.charge_clamp:
            mu_hc = mu_hc.clamp(-2.0, 2.0)
            sig_hc = sig_hc.clamp(1e-6, 4.0)
        p = gaussian_cdf(mu_hc, sig_hc, self.k_r) - gaussian_cdf(mu_hc, sig_hc, self.k_l)
        # tiny negative values can appear due to floating-point error; clamp and
        # normalize to guarantee a valid distribution
        p = p.clamp_min(0.0)
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return p, mu_hc, sig_hc

    def nearest_center(self, v: torch.Tensor) -> torch.Tensor:
        """Map a continuous charge value to the nearest bin center (Eq.20).

        v: [N,1] -> [N,1]
        """
        diff = (v.unsqueeze(1) - self.k_c.view(1, -1, 1)).abs()  # [N,K,1]
        idx = diff.argmin(dim=1)                                  # [N,1]
        return self.k_c[idx]

    # ------------------------------------------------------------------ #
    # 3) One forward pass: theta -> parameters of p_O
    # ------------------------------------------------------------------ #
    def forward_theta(
        self, t: torch.Tensor, mu_x: torch.Tensor, mu_c: torch.Tensor,
        gamma_x: torch.Tensor, gamma_c: torch.Tensor,
        edge_index: torch.Tensor, batch: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """One forward pass of network Phi plus the inverse Bayesian update.

        Args:
            t:        [N,1] time (concatenated as the last node-feature column)
            mu_x:     [N,3] theta_x of the coordinate modality
            mu_c:     [N,1] theta_hc of the charge modality
            gamma_x / gamma_c: [N,1] signal ratio of the two modalities
        Returns:
            dict: pos_pred [N,3], p [N,K], phi_c [N,1], eps_x_hat [N,3],
                  mu_hc [N,1], sig_hc [N,1]
        """
        h_in = self.gaussian_basis(mu_c)              # [N,16], binning
        h = torch.cat([h_in, t], dim=-1)              # [N,17] concat mean and time
        h_out, coord_raw = self.backbone(h, mu_x, edge_index)   # h_out [N,2], coord_raw [N,3]

        # residual parameterization: the network fits "mu + eps"; subtract mu to
        # get the noise estimate eps_hat
        eps_x_hat = coord_raw - mu_x
        # keep the noise estimate in the zero-centroid space -> reconstructed
        # coordinates are also zero-centroid (translation invariant)
        eps_x_hat = zero_com(eps_x_hat, batch)
        eps_x_hat = eps_x_hat.clamp(-10.0, 10.0)

        mu_eps = h_out[:, :1].clamp(-10.0, 10.0)          # charge noise mean
        log_sigma_eps = h_out[:, 1:].clamp(-10.0, 10.0)   # charge noise log std

        # inverse Bayesian update for coordinates (the Dirac center of Eq.14):
        #   x_hat = mu_x/gamma_x - sqrt((1-gamma_x)/gamma_x)*eps_hat
        s_x = torch.sqrt((1.0 - gamma_x) / gamma_x)
        pos_pred = mu_x / gamma_x - s_x * eps_x_hat

        # charges: bin probabilities + expectation (Eq.15 / Phi_hc of Eq.19)
        p, mu_hc, sig_hc = self.bucket_probs(mu_c, gamma_c, mu_eps, log_sigma_eps)
        phi_c = (p * self.k_c.view(1, -1)).sum(dim=-1, keepdim=True)   # [N,1]

        return {
            "pos_pred": pos_pred,
            "p": p,
            "phi_c": phi_c,
            "eps_x_hat": eps_x_hat,
            "mu_hc": mu_hc,
            "sig_hc": sig_hc,
        }

    # ------------------------------------------------------------------ #
    # 4) Loss (paper Eq.19)
    # ------------------------------------------------------------------ #
    @staticmethod
    def x_space_weight(t: torch.Tensor, sigma1: torch.Tensor) -> torch.Tensor:
        """x-space time weight: alpha(t) / 2.

        Derivation:
            beta(t)  = sigma1^(-2t) - 1                  # accuracy schedule
            alpha(t) = d beta(t) / dt                    # Eq. (9): beta(t) = integral alpha dt
                     = -2 * ln(sigma1) * sigma1^(-2t)    # derivative of sigma1^(-2t)
            alpha(t) / 2 = -ln(sigma1) * sigma1^(-2t)    # exactly x_space_weight(t, sigma1)
        So:
            x_space_weight(t, sigma1) = alpha(t) / 2
        """

        return -torch.log(sigma1) * sigma1 ** (-2.0 * t)

    @staticmethod
    def eps_space_weight(t: torch.Tensor, sigma1: torch.Tensor) -> torch.Tensor:
        """eps-space time weight, strictly equivalent to x-space per sample.

        Derivation:
            # Step 1: BFN reparameterization for continuous data
            #   mu = gamma * x + sqrt(gamma * (1 - gamma)) * eps
            #   gamma = 1 - sigma1^(2t)

            # Step 2: solve x from mu (paper Eq. 83)
            #   x = mu / gamma - sqrt((1 - gamma) / gamma) * eps

            # Step 3: network predicts eps_hat, so x_hat has the same form (paper Eq. 84)
            #   x_hat = mu / gamma - sqrt((1 - gamma) / gamma) * eps_hat

            # Step 4: subtract the two equations
            #   x - x_hat = -sqrt((1 - gamma) / gamma) * (eps - eps_hat)

            # Step 5: square both sides
            #   ||x - x_hat||^2 = (1 - gamma) / gamma * ||eps - eps_hat||^2

            # Step 6: substitute into x-space loss
            #   x_space_weight * ||x - x_hat||^2
            #       = x_space_weight * (1 - gamma) / gamma * ||eps - eps_hat||^2

            # Step 7: simplify the coefficient
            #   x_space_weight = -ln(sigma1) * sigma1^(-2t)
            #   (1 - gamma) / gamma = sigma1^(2t) / (1 - sigma1^(2t))
            #   x_space_weight * (1 - gamma) / gamma
            #       = [-ln(sigma1) * sigma1^(-2t)] * [sigma1^(2t) / (1 - sigma1^(2t))]
            #       = -ln(sigma1) / (1 - sigma1^(2t))

        So:
            eps_space_weight(t, sigma1)
                = (-ln(sigma1)) / (1 - sigma1^(2t))
        """

        return -torch.log(sigma1) / (1.0 - sigma1 ** (2.0 * t)).clamp_min(1e-12)

    def loss(
        self, batch: Dict[str, torch.Tensor], generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the Eq.19 loss for one batch (returns the scalar and statistics).

        Args:
            batch: output of collate(), with pos/charges/edge_index/batch/n_mol
        Returns:
            loss:  scalar (mean over nodes, as in the original repo: large
                   molecules weigh more)
            stats: log-friendly decomposition (means of pos / charge / gamma)
        """
        pos, charges = batch["pos"], batch["charges"]
        edge_index, seg = batch["edge_index"], batch["batch"]
        n_mol, device = batch["n_mol"], pos.device

        # ---- sample t per molecule, broadcast to atoms; clamp to t_min ----
        t_mol = torch.rand(n_mol, 1, device=device, generator=generator, dtype=pos.dtype)
        t = t_mol.index_select(0, seg).clamp_min(self.cfg.t_min)   # [N,1]

        # ---- theta ~ p_F (the two modalities are independent) ----
        mu_x, gamma_x, eps_x = self.bayes_flow(t, self.sigma1_coord, pos, seg, generator)
        mu_c, gamma_c, eps_c = self.bayes_flow(t, self.sigma1_charges, charges, seg, generator)

        # numerical guard (mu_x is already zero-centroid: x and eps are
        # zero-centroid, so the convex combination stays zero-centroid)
        mu_x = mu_x.clamp(-self.cfg.mu_clamp, self.cfg.mu_clamp)
        mu_c = mu_c.clamp(-self.cfg.mu_clamp, self.cfg.mu_clamp)

        out = self.forward_theta(t, mu_x, mu_c, gamma_x, gamma_c, edge_index, seg)

        # ---- coordinate term ----
        if self.cfg.loss_space == "x":
            # paper's original form: alpha/2 * ||x - x_hat||^2
            w_x = self.x_space_weight(t, self.sigma1_coord)
            l_pos = w_x * ((out["pos_pred"] - pos) ** 2).sum(dim=-1, keepdim=True)
        else:
            # equivalent form: w_eps * ||eps_hat - eps||^2 (more stable)
            w_x = self.eps_space_weight(t, self.sigma1_coord)
            l_pos = w_x * ((out["eps_x_hat"] - eps_x) ** 2).sum(dim=-1, keepdim=True)

        # ---- charge term: alpha_c/2 * ||h_c - Phi_hc||^2 ----
        w_c = self.x_space_weight(t, self.sigma1_charges)
        l_chg = w_c * (out["phi_c"] - charges) ** 2

        per_node = l_pos + l_chg                      # [N,1]
        loss = per_node.mean()                        # mean over nodes (= atom-count weighted)

        with torch.no_grad():
            stats = {
                "loss_pos": l_pos.mean().detach(),
                "loss_charge": l_chg.mean().detach(),
                "gamma_x": gamma_x.mean().detach(),
                "gamma_c": gamma_c.mean().detach(),
            }
        return loss, stats

    # ------------------------------------------------------------------ #
    # 5) Sampling (the recursive process of paper Fig.2)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(
        self,
        n_nodes: int,
        edge_index: torch.Tensor,
        seg: torch.Tensor,
        steps: int = 1000,
        generator: Optional[torch.Generator] = None,
        keep_trajectory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Start from the prior theta_0 and iterate S Bayesian updates.

        Returns the generated coordinates and atomic numbers.
        Algorithm (for each modality):
            t_i = (i-1)/S,  gamma_i = 1 - sigma1^(2 t_i)
            g' = Phi(theta_{i-1}, t_i)                     # p_O
            y  ~ N(g', 1/alpha_i), alpha_i = sigma1^(-2i/S)*(1-sigma1^(2/S))  # p_S
            theta_i = (rho_{i-1} theta_{i-1} + alpha_i y)/(rho_{i-1}+alpha_i),
            rho_i = rho_{i-1}+alpha_i
        Note that alpha_i satisfies sum alpha_i = sigma1^(-2) - 1 = beta(1), so any
        number of steps S reaches the same accuracy (this is the basis of the
        paper's "sampling with an arbitrary number of steps").

        Args:
            n_nodes: total number of atoms
            edge_index: [2,E] fully connected graph
            seg: [N] node->molecule
            steps: number of sampling steps S
            keep_trajectory: whether to save intermediate results (for
                             debugging/visualization)
        Returns:
            dict(pos [N,3] zero-centroid coordinates, z [N] atomic numbers,
                 traj optional)
        """
        device = edge_index.device
        dtype = self.centers.dtype

        mu_x = torch.zeros(n_nodes, 3, device=device, dtype=dtype)
        mu_c = torch.zeros(n_nodes, 1, device=device, dtype=dtype)
        rho_x = 1.0   # prior precision rho_0 = 1
        rho_c = 1.0
        traj = []

        for i in range(1, steps + 1):
            t = torch.full((n_nodes, 1), (i - 1) / steps, device=device, dtype=dtype)
            t = t.clamp_min(self.cfg.t_min)
            gamma_x = 1.0 - self.sigma1_coord ** (2.0 * t)
            gamma_c = 1.0 - self.sigma1_charges ** (2.0 * t)

            mu_x = zero_com(mu_x.clamp(-self.cfg.mu_clamp, self.cfg.mu_clamp), seg)
            mu_c = mu_c.clamp(-self.cfg.mu_clamp, self.cfg.mu_clamp)

            out = self.forward_theta(t, mu_x, mu_c, gamma_x, gamma_c, edge_index, seg)

            # ---- coordinates: one explicit Eq.11 update ----
            alpha_x = self.sigma1_coord ** (-2.0 * i / steps) * (
                1.0 - self.sigma1_coord ** (2.0 / steps)
            )
            noise = torch.randn(out["pos_pred"].shape, device=device, dtype=dtype, generator=generator)
            y_x = out["pos_pred"] + noise * torch.sqrt(1.0 / alpha_x)
            y_x = zero_com(y_x.clamp(-10.0, 10.0), seg)
            mu_x = (rho_x * mu_x + alpha_x * y_x) / (rho_x + alpha_x)
            rho_x = rho_x + alpha_x

            # ---- charges: correct with nearest_center first, then add noise (Eq.20) ----
            phi = out["phi_c"]
            e_c = self.nearest_center(phi)
            alpha_c = self.sigma1_charges ** (-2.0 * i / steps) * (
                1.0 - self.sigma1_charges ** (2.0 / steps)
            )
            noise_c = torch.randn(e_c.shape, device=device, dtype=dtype, generator=generator)
            y_c = e_c + noise_c * torch.sqrt(1.0 / alpha_c)
            mu_c = (rho_c * mu_c + alpha_c * y_c) / (rho_c + alpha_c)
            rho_c = rho_c + alpha_c

            if keep_trajectory:
                traj.append({"pos": out["pos_pred"].clone(), "z": self.decode_charges(e_c)})

        # ---- one final forward pass at t=1 to obtain the final sample ----
        t1 = torch.ones(n_nodes, 1, device=device, dtype=dtype)
        out = self.forward_theta(
            t1, zero_com(mu_x, seg), mu_c,
            1.0 - self.sigma1_coord ** 2, 1.0 - self.sigma1_charges ** 2,
            edge_index, seg,
        )
        pos = zero_com(out["pos_pred"], seg)
        z_val = self.nearest_center(out["phi_c"])
        return {"pos": pos, "z": self.decode_charges(z_val), "traj": traj}

    # ------------------------------------------------------------------ #
    # 6) Decoding: continuous charge -> atomic number
    # ------------------------------------------------------------------ #
    def decode_charges(self, v: torch.Tensor) -> torch.Tensor:
        """Map a charge value to the nearest **legal element** anchor.

        Returns atomic numbers [N].
        Difference from nearest_center: this only searches among the 5 anchors
        corresponding to {Z=1,6,7,8,9}, because no other element can occur when
        generating molecules.
        """
        diff = (v.unsqueeze(1) - self.z_anchors.view(1, -1, 1)).abs()  # [N,5,1]
        idx = diff.argmin(dim=1).squeeze(-1)                            # [N]
        anchors = torch.tensor(self.atomic_nb, device=v.device, dtype=torch.long)
        return anchors[idx]
