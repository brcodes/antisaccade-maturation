"""Linear readout mapping hidden firing rates to a two-option motor plan."""

from __future__ import annotations

import torch
import torch.nn as nn


class Readout(nn.Module):
    """Linear map ``r(t) -> z(t)`` with ``z = [z_cue, z_goal]`` (unnormalized).

    The decision rule (race to threshold) is applied downstream in the training
    losses / analysis code, not here, so that the same readout can feed both the
    soft (differentiable) and hard (straight-through) commitment computations.
    """

    def __init__(self, n_hidden: int, n_output: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_hidden, n_output, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Map firing rates ``[T, B, N]`` to outputs ``[T, B, n_output]``."""
        return self.linear(r)
