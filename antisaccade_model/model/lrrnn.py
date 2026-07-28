"""Continuous-time low-rank RNN (Mastrogiuseppe & Ostojic 2018).

Dynamics (Euler integration, dt = task.dt)::

    tau dh/dt = -h + W_rec r + W_in u + noise
    r = phi(h)

The recurrent weight matrix is constrained to rank ``R`` via an outer-product
parameterization ``W_rec = (M N^T) / N_hidden`` with ``M, N`` of shape
``[N_hidden, R]`` (the M&O "input"/"output" modes). Maturation enters only as an
input channel (mechanism 1), so no maturation-specific parameters live here.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..task.task_params import TaskParams
from .model_params import ModelParams
from .readout import Readout


class LRRNN(nn.Module):
    """Rank-constrained continuous-time RNN with a linear two-option readout."""

    def __init__(self, model: ModelParams, task: TaskParams) -> None:
        super().__init__()
        self.model = model
        self.task = task
        n, r = model.n_hidden, model.n_rank

        scale = model.init_rec_scale / (n ** 0.5)
        # Low-rank connectivity modes M, N in R^{N x R}.
        self.M = nn.Parameter(torch.randn(n, r) * scale)
        self.N = nn.Parameter(torch.randn(n, r) * scale)

        # Input weights (Xavier initialization).
        self.W_in = nn.Parameter(torch.empty(n, model.n_input))
        nn.init.xavier_uniform_(self.W_in)

        self.readout = Readout(n, model.n_output)

        if model.phi == "tanh":
            self.phi = torch.tanh
        elif model.phi == "relu":
            self.phi = torch.relu
        else:  # pragma: no cover - guarded by config
            raise ValueError(f"Unknown nonlinearity: {model.phi}")

    def recurrent_matrix(self) -> torch.Tensor:
        """Return the rank-R recurrent weight matrix ``W_rec`` (``[N, N]``)."""
        return (self.M @ self.N.t()) / self.model.n_hidden

    def forward(
        self,
        u: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        add_noise: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Integrate the dynamics for an input sequence.

        Args:
            u: input tensor ``[T, B, n_input]``.
            h0: optional initial hidden state ``[B, N]`` (defaults to zeros;
                mechanism-1 conditioning does not modulate h0).
            add_noise: whether to inject recurrent noise.
            generator: optional RNG for reproducible noise.

        Returns:
            ``(h, r, z)`` with shapes ``[T, B, N]``, ``[T, B, N]``,
            ``[T, B, n_output]``.
        """
        n_steps, batch, _ = u.shape
        n = self.model.n_hidden
        dt, tau = self.task.dt, self.task.tau
        w_rec = self.recurrent_matrix()

        x = torch.zeros(batch, n) if h0 is None else h0
        noise_scale = self.task.sigma_noise * (dt / tau) ** 0.5

        hs, rs = [], []
        for t in range(n_steps):
            r = self.phi(x)
            rec = r @ w_rec.t()
            inp = u[t] @ self.W_in.t()
            dx = (dt / tau) * (-x + rec + inp)
            if add_noise and self.task.sigma_noise > 0:
                dx = dx + noise_scale * torch.randn(batch, n, generator=generator)
            x = x + dx
            hs.append(x)
            rs.append(self.phi(x))

        h = torch.stack(hs, dim=0)
        r = torch.stack(rs, dim=0)
        z = self.readout(r)
        return h, r, z
