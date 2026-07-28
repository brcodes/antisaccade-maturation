"""rPT (gap) curriculum scheduler.

Early in training the network only sees extreme gaps, where the task is easy
(very short gap -> long rPT -> clear goal-directed response; very long gap ->
short rPT -> pure guessing). The intermediate gaps that produce the vortex are
introduced gradually to avoid collapse to a trivial solution (gameplan 3.3).

This is implemented as a shrinking central "hole": early epochs exclude the
middle band of gaps, and the hole closes by ``warmup_epochs``.
"""

from __future__ import annotations

from typing import Optional

import torch

from ..task.task_params import TaskParams


def hole_halfwidth_for_epoch(
    epoch: int,
    task: TaskParams,
    warmup_epochs: int = 100,
    max_hole_frac: float = 0.45,
) -> float:
    """Half-width (ms) of the excluded central gap band for a given epoch."""
    frac = min(1.0, epoch / max(1, warmup_epochs))
    span = task.gap_max - task.gap_min
    return max_hole_frac * span * (1.0 - frac)


def sample_curriculum_gaps(
    batch_size: int,
    epoch: int,
    task: TaskParams,
    warmup_epochs: int = 100,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample gaps with the central band excluded according to the curriculum.

    Uses rejection sampling against the central hole; the hole shrinks to zero
    by ``warmup_epochs`` so the full gap range is eventually sampled.
    """
    center = 0.5 * (task.gap_min + task.gap_max)
    hole = hole_halfwidth_for_epoch(epoch, task, warmup_epochs)

    gaps = torch.empty(batch_size)
    filled = 0
    # Rejection sampling; the accepted fraction is bounded well away from 0.
    while filled < batch_size:
        n = batch_size - filled
        cand = torch.rand(n, generator=generator) * (task.gap_max - task.gap_min) + task.gap_min
        keep = cand[(cand - center).abs() >= hole]
        take = min(keep.shape[0], n)
        if take > 0:
            gaps[filled:filled + take] = keep[:take]
            filled += take
    return gaps
