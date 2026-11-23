"""The E(3)-equivariant diffusion model (EDM) itself.

Everything probabilistic lives in this file:

1. Noise schedule.  Cosine, in continuous time t in [0,1]:

       abar(t) = cos^2( (t + s) / (1 + s) * pi/2 ) / cos^2( s / (1 + s) * pi/2 )
       alpha(t) = sqrt(abar(t)),   sigma(t) = sqrt(1 - abar(t))

   so the forward (diffusion) kernel on coordinates is the standard
   variance-preserving Gaussian  q(x_t | x_0) = N(alpha(t) x_0, sigma(t)^2 I).

2. Discrete diffusion on atom types.  A uniform (D3PM-style) transition keeps
   the marginal away from zero:   q(h_t | h_0) = abar(t) * onehot(h_0) + (1-abar(t))/K
   i.e. with probability abar(t) the true type survives, otherwise a uniformly
   random type is drawn.

3. Training objective (Hoogeboom et al., 2022; Nichol and Dhariwal, 2021):

       L = E_{t, eps} [ w(t) ||eps - eps_hat||^2 ] + lambda_h * CE(h_0, logits)

   with the optional SNR weight w(t) = abar(t) / sigma(t)^2 (clipped) that
   upweights low-noise (structural) timesteps.

4. Reverse process.  A DDIM (eta = 0) or DDPM (eta > 0) ancestral sampler, plus
   the uniform-transition posterior for the atom types:

       x_s = alpha(s) x0_hat + sigma(s) eps_hat                       (DDIM)
       q(h_s | h_t, h0_hat) proportional to
           [r h_t + (1-r)/K] * [abar(s) h0_hat + (1-abar(s))/K],
           r = abar(t)/abar(s)

   with s < t, so abar(s) > abar(t) and r <= 1.  The centre of mass of the
   coordinates is projected to zero after every step: translation is a symmetry of
   the problem, not a degree of freedom to model.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DiffusionConfig, ModelConfig
from .data import zero_com
from .egnn import EGNN


class EDMDiffusion(nn.Module):
    """Equivariant diffusion model over (coordinates, atom types).

    Args:
        model_cfg: backbone hyper-parameters.
        diff_cfg: schedule and sampler hyper-parameters.
        n_types: number of atom-type classes K.
    """

    def __init__(self, model_cfg: ModelConfig, diff_cfg: DiffusionConfig, n_types: int,
                 atomic_nb: Optional[Sequence[int]] = None):
        super().__init__()
        self.mcfg = model_cfg
        self.dcfg = diff_cfg
        self.n_types = int(n_types)
        # class index -> atomic number, used to decode generated samples
        self.atomic_numbers = tuple(int(z) for z in (atomic_nb or (1, 6, 7, 8, 9)))
        self.backbone = EGNN(
            in_node_nf=self.n_types,
            out_node_nf=self.n_types,
            hidden_nf=model_cfg.hidden_nf,
            n_layers=model_cfg.n_layers,
            time_embed_dim=model_cfg.time_embed_dim,
            attention=model_cfg.attention,
            tanh=model_cfg.tanh,
            coords_range=model_cfg.coords_range,
            norm_constant=model_cfg.norm_constant,
        )

    # ------------------------------------------------------------------ #
    # schedule
    # ------------------------------------------------------------------ #
    def abar(self, t: torch.Tensor) -> torch.Tensor:
        """alpha_bar(t) for t in [0,1] (cosine schedule, normalised to abar(0)=1)."""
        s = self.dcfg.cosine_s
        f = torch.cos((t + s) / (1.0 + s) * (torch.pi / 2.0)) ** 2
        f0 = torch.cos(torch.tensor(s / (1.0 + s) * (torch.pi / 2.0),
                                   device=t.device, dtype=t.dtype)) ** 2
        return (f / f0).clamp(1e-8, 1.0 - 1e-8)

    def alpha_sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Signal and noise coefficients (alpha_t, sigma_t)."""
        a = self.abar(t)
        return torch.sqrt(a), torch.sqrt(1.0 - a)

    # ------------------------------------------------------------------ #
    # forward process
    # ------------------------------------------------------------------ #
    def q_sample_pos(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor,
                     seg: torch.Tensor) -> torch.Tensor:
        """x_t = alpha_t x_0 + sigma_t eps, with eps projected to zero COM."""
        alpha, sigma = self.alpha_sigma(t)
        eps = zero_com(eps, seg)
        x_t = alpha * x0 + sigma * eps
        return zero_com(x_t, seg)

    def q_sample_h(self, h0: torch.Tensor, t: torch.Tensor,
                   generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Uniform (D3PM) corruption of atom types; returns one-hot [N,K]."""
        onehot = F.one_hot(h0, self.n_types).float()
        a = self.abar(t)
        keep = torch.rand(onehot.shape, device=h0.device, dtype=onehot.dtype,
                          generator=generator) < a
        rand_type = torch.randint(0, self.n_types, h0.shape, device=h0.device,
                                  generator=generator)
        rand_onehot = F.one_hot(rand_type, self.n_types).float()
        return torch.where(keep, onehot, rand_onehot)

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def loss(self, batch: Dict[str, torch.Tensor],
             generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """One training step of the objective described in the module docstring."""
        pos, h = batch["pos"], batch["h"]
        edge_index, seg, n_mol = batch["edge_index"], batch["batch"], batch["n_mol"]
        device = pos.device

        # one continuous time per molecule, broadcast to its atoms
        t_mol = torch.rand(n_mol, 1, device=device, dtype=pos.dtype, generator=generator)
        t = t_mol.index_select(0, seg)

        eps = torch.randn(pos.shape, device=device, dtype=pos.dtype, generator=generator)
        x_t = self.q_sample_pos(pos, t, eps, seg)
        h_t = self.q_sample_h(h, t, generator)

        logits, eps_hat = self.backbone(h_t, x_t, edge_index, t)

        # coordinate term
        se = ((eps_hat - eps) ** 2).sum(dim=-1, keepdim=True)      # [N,1]
        if self.dcfg.loss_weight == "snr":
            a = self.abar(t)
            w = (a / (1.0 - a).clamp_min(1e-6)).clamp(max=self.dcfg.w_clip)
        else:
            w = torch.ones_like(t)
        loss_pos = (w * se).mean()

        # atom-type term: cross entropy against the clean type
        loss_h = F.cross_entropy(logits, h, reduction="mean")

        loss = loss_pos + self.dcfg.lambda_h * loss_h
        stats = {
            "loss_pos": loss_pos.detach(),
            "loss_h": loss_h.detach(),
            "alpha_mean": self.alpha_sigma(t)[0].mean().detach(),
        }
        return loss, stats

    # ------------------------------------------------------------------ #
    # reverse process
    # ------------------------------------------------------------------ #
    def _discrete_posterior(self, h_t_oh: torch.Tensor, p0: torch.Tensor,
                            a_t: torch.Tensor, a_s: torch.Tensor) -> torch.Tensor:
        """q(h_s | h_t, h0_hat) for the uniform transition kernel.

        This is Bayes' rule with the uniform kernel, written per class:
            q(h_s=j | h_t, h0_hat) proportional to
                [r h_t + (1-r)/K]_j * [abar(s) h0_hat + (1-abar(s))/K]_j
        where r = abar(t)/abar(s) <= 1 is the retention of the step (t is the
        noisier state and s < t the cleaner one, so abar(s) > abar(t)).

        Args:
            h_t_oh: [N,K] current one-hot state.
            p0: [N,K] predicted distribution of the clean atom type.
            a_t, a_s: [N,1] abar at the current and the next (lower) time.
        """
        k = float(self.n_types)
        ratio = (a_t / a_s).clamp(1e-6, 1.0)                       # retention of the step
        term_cur = ratio * h_t_oh + (1.0 - ratio) / k
        term_0 = a_s * p0 + (1.0 - a_s) / k
        post = term_cur * term_0
        return post / post.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    @torch.no_grad()
    def sample(
        self,
        n_nodes: int,
        edge_index: torch.Tensor,
        seg: torch.Tensor,
        steps: Optional[int] = None,
        eta: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        keep_trajectory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate coordinates and atom types for a batch of empty graphs.

        Args:
            n_nodes: total number of atoms in the batch.
            edge_index: [2,E] fully connected edges inside each molecule.
            seg: [N] molecule index of every atom.
            steps: number of reverse steps (defaults to ``diff_cfg.sample_steps``).
            eta: variance scale of the injected noise; 0 is deterministic DDIM,
                1 is ancestral DDPM sampling.
        Returns:
            dict with ``pos`` [N,3], ``z`` [N] atomic numbers and ``traj``.
        """
        steps = int(steps or self.dcfg.sample_steps)
        eta = float(self.dcfg.sample_eta if eta is None else eta)
        device, dtype = edge_index.device, self.backbone.out.weight.dtype

        x = zero_com(torch.randn(n_nodes, 3, device=device, dtype=dtype,
                                generator=generator), seg)
        h0_rand = torch.randint(0, self.n_types, (n_nodes,), device=device, generator=generator)
        h = F.one_hot(h0_rand, self.n_types).to(dtype)
        traj = []

        for i in range(steps, 0, -1):
            t_val = i / steps
            s_val = (i - 1) / steps
            t = torch.full((n_nodes, 1), t_val, device=device, dtype=dtype)
            s = torch.full((n_nodes, 1), s_val, device=device, dtype=dtype)

            logits, eps_hat = self.backbone(h, x, edge_index, t)
            alpha_t, sigma_t = self.alpha_sigma(t)
            alpha_s, sigma_s = self.alpha_sigma(s)

            # implied clean coordinates, then one reverse step
            x0_hat = ((x - sigma_t * eps_hat) / alpha_t).clamp(
                -self.dcfg.clip_denoise, self.dcfg.clip_denoise)
            if eta <= 0.0:
                x = alpha_s * x0_hat + sigma_s * eps_hat
            else:
                a_t = self.abar(t)
                a_s = self.abar(s)
                # one-step noise level between the two abar values (s is cleaner,
                # hence abar(s) > abar(t) and beta > 0)
                beta = (1.0 - a_t / a_s).clamp(0.0, 1.0)
                coef_x0 = torch.sqrt(a_s) * beta / (1.0 - a_t).clamp_min(1e-8)
                coef_xt = torch.sqrt(a_t / a_s) * (1.0 - a_s) / (1.0 - a_t).clamp_min(1e-8)
                var = beta * (1.0 - a_s) / (1.0 - a_t).clamp_min(1e-8)
                mean = coef_x0 * x0_hat + coef_xt * x
                noise = torch.randn(x.shape, device=device, dtype=dtype, generator=generator)
                x = mean + (eta * var).clamp_min(0.0).sqrt() * noise
            x = zero_com(x, seg)

            # atom types: posterior given the predicted clean distribution
            p0 = F.softmax(logits, dim=-1)
            if i == 1:
                h = F.one_hot(p0.argmax(dim=-1), self.n_types).to(dtype)
            else:
                post = self._discrete_posterior(h, p0, self.abar(t), self.abar(s))
                drawn = torch.multinomial(post, num_samples=1).squeeze(-1)
                h = F.one_hot(drawn, self.n_types).to(dtype)

            if keep_trajectory:
                traj.append({"pos": x0_hat.clone(), "h": h.argmax(dim=-1).clone()})

        z = self.decode_atoms(h.argmax(dim=-1))
        return {"pos": zero_com(x, seg), "z": z, "traj": traj}

    # ------------------------------------------------------------------ #
    # decoding
    # ------------------------------------------------------------------ #
    def decode_atoms(self, class_ids: torch.Tensor) -> torch.Tensor:
        """Class index -> atomic number, using the dataset's element order."""
        anchors = torch.tensor(self.atomic_numbers, device=class_ids.device,
                               dtype=torch.long)
        return anchors[class_ids.long()]
