"""Trial generation for the compelled antisaccade task.

A trial's input tensor has shape ``[T, B, N_INPUT]`` (time, batch, channels).
The go signal is a step function from ``go_step`` onward; the cue turns on at
``t_cue = t_pre + gap`` as an exogenous burst plus a sustained component; the
task rule and maturation scalar are constant across the whole trial (mechanism 1).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from .task_params import (
    CUE_LEFT,
    CUE_LEFT_IDX,
    CUE_RIGHT_IDX,
    GO_IDX,
    MATURATION_IDX,
    N_INPUT,
    RULE_IDX,
    DEFAULT_TASK,
    TaskParams,
)


def sample_initial_state(
    batch_size: int,
    n_hidden: int,
    task: Optional[TaskParams] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample the trial initial state with shared and private variability."""
    task = DEFAULT_TASK if task is None else task
    shared = torch.randn(batch_size, 1, generator=generator) * task.sigma_init_shared
    private = torch.randn(batch_size, n_hidden, generator=generator) * task.sigma_init_private
    return shared + private


def build_inputs(
    gaps: torch.Tensor,
    cue_sides: torch.Tensor,
    m_values: torch.Tensor,
    task: TaskParams,
    lapse_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the input tensor for a batch of trials.

    Args:
        gaps: ``[B]`` gap durations (ms) between go signal and cue onset.
        cue_sides: ``[B]`` cue side per trial (``CUE_LEFT`` or ``CUE_RIGHT``).
        m_values: ``[B]`` maturation scalar per trial in ``[0, 1]``.
        task: task parameters.

    Returns:
        ``(u, t_cue)`` where ``u`` is ``[T, B, N_INPUT]`` and ``t_cue`` is the
        ``[B]`` cue onset time (ms from simulation start).
    """
    gaps = gaps.float()
    m_values = m_values.float()
    lapse_mask = lapse_mask.bool() if lapse_mask is not None else None
    batch = gaps.shape[0]
    n_steps = task.n_steps
    time = torch.arange(n_steps, dtype=torch.float32) * task.dt  # ms from sim start

    u = torch.zeros(n_steps, batch, N_INPUT)

    # Go signal: step from t = 0 (go_step) onward.
    u[:, :, GO_IDX] = (time[:, None] >= task.t_pre).float()

    # Task rule (antisaccade) and maturation scalar: constant across the trial.
    u[:, :, RULE_IDX] = 1.0
    if lapse_mask is not None:
        u[:, lapse_mask, RULE_IDX] = 0.0
    u[:, :, MATURATION_IDX] = m_values[None, :]
    if lapse_mask is not None:
        u[:, lapse_mask, MATURATION_IDX] = 0.0

    # Cue onset per trial and time relative to it.
    t_cue = task.t_pre + gaps                       # [B]
    dt_from_cue = time[:, None] - t_cue[None, :]     # [T, B]
    after_cue = (dt_from_cue >= 0).float()

    burst = task.a_exo * torch.exp(-torch.clamp(dt_from_cue, min=0.0) / task.tau_exo)
    cue_drive = (burst + task.cue_sustained) * after_cue  # [T, B]

    is_left = (cue_sides == CUE_LEFT).float()[None, :]
    is_right = (cue_sides != CUE_LEFT).float()[None, :]
    u[:, :, CUE_LEFT_IDX] = cue_drive * is_left
    u[:, :, CUE_RIGHT_IDX] = cue_drive * is_right

    return u, t_cue


def sample_batch(
    batch_size: int,
    task: TaskParams,
    m_choices: Sequence[float] = (0.0, 1.0),
    gap_range: Optional[tuple[float, float]] = None,
    generator: Optional[torch.Generator] = None,
) -> dict:
    """Sample a random batch of trials.

    Gap is sampled uniformly (optionally restricted by ``gap_range`` for the
    curriculum), cue side is balanced, and the maturation scalar is drawn from
    the discrete ``m_choices`` (``{0, 1}`` for training).
    """
    gap_lo, gap_hi = gap_range if gap_range is not None else (task.gap_min, task.gap_max)
    gaps = torch.rand(batch_size, generator=generator) * (gap_hi - gap_lo) + gap_lo
    cue_sides = torch.randint(0, 2, (batch_size,), generator=generator)
    m_pool = torch.tensor(list(m_choices), dtype=torch.float32)
    m_idx = torch.randint(0, len(m_pool), (batch_size,), generator=generator)
    m_values = m_pool[m_idx]

    u, t_cue = build_inputs(gaps, cue_sides, m_values, task)
    return {
        "u": u,
        "gaps": gaps,
        "cue_sides": cue_sides,
        "m": m_values,
        "t_cue": t_cue,
    }


def sweep_batch(
    task: TaskParams,
    m_value: float,
    gaps: torch.Tensor,
    trials_per_gap: int,
    cue_sides: Optional[torch.Tensor] = None,
    n_hidden: Optional[int] = None,
    lapse_rate: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> dict:
    """Build a deterministic gap sweep for one maturation state (analysis use).

    Each gap value is repeated ``trials_per_gap`` times, with cue side balanced
    across repeats so the tachometric curve is symmetric by design.
    """
    gaps = gaps.float().repeat_interleave(trials_per_gap)
    batch = gaps.shape[0]
    if cue_sides is None:
        cue_sides = (torch.arange(batch) % 2)
    m_values = torch.full((batch,), float(m_value))
    lapse_mask = torch.rand(batch, generator=generator) < lapse_rate
    u, t_cue = build_inputs(gaps, cue_sides, m_values, task, lapse_mask=lapse_mask)
    h0 = sample_initial_state(batch, n_hidden, task, generator) if n_hidden is not None else None
    return {
        "u": u,
        "gaps": gaps,
        "cue_sides": cue_sides,
        "m": m_values,
        "t_cue": t_cue,
        "h0": h0,
        "lapse_mask": lapse_mask,
    }
