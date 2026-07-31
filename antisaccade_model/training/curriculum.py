"""Gap samplers for training and legacy curriculum use.

Primary training uses equal-count stratified sampling across the full gap
range. The central-hole curriculum helpers remain available for experimental
callers that explicitly use them.

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

N_GAP_STRATA = 5


def sample_stratified_gaps(
    batch_size: int,
    task: TaskParams,
    n_strata: int = N_GAP_STRATA,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample equal numbers of gaps uniformly within contiguous strata."""
    if n_strata <= 0:
        raise ValueError("n_strata must be positive")
    if batch_size % n_strata != 0:
        raise ValueError(f"batch_size ({batch_size}) must be divisible by n_strata ({n_strata})")
    if task.gap_max <= task.gap_min:
        raise ValueError("task.gap_max must be greater than task.gap_min")

    samples_per_stratum = batch_size // n_strata
    stratum_width = (task.gap_max - task.gap_min) / n_strata
    stratum_indices = torch.arange(n_strata).repeat_interleave(samples_per_stratum)
    gaps = (
        task.gap_min
        + stratum_indices * stratum_width
        + torch.rand(batch_size, generator=generator) * stratum_width
    )
    return gaps[torch.randperm(batch_size, generator=generator)]


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
