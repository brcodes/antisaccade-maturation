"""Extract tachometric curves from the trained model and fit summary statistics.

Unlike the training-time soft extractor, this module uses the *hard* first-
passage decision to get an emergent rPT per trial, bins trials into the rPT
grid, and fits the parametric curve with SciPy for reporting.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy.optimize import curve_fit
from scipy.stats import norm

from ..model.lrrnn import LRRNN
from ..task.task_params import TaskParams
from ..task.trial_generator import sweep_batch
from ..training.losses import hard_commitment


@torch.no_grad()
def run_gap_sweep(
    model: LRRNN,
    task: TaskParams,
    m_value: float,
    trials_per_gap: int = 200,
    gap_grid: Optional[np.ndarray] = None,
    add_noise: bool = True,
) -> dict:
    """Simulate a gap sweep and return per-trial behavior plus hidden rates.

    Returns a dict with ``rpt`` ``[B]``, ``correct`` ``[B]`` (0/1),
    ``crossed`` ``[B]``, ``r`` ``[T, B, N]`` firing rates, ``cue_sides`` ``[B]``,
    and ``t_cue`` ``[B]``.
    """
    if gap_grid is None:
        gap_grid = np.arange(task.gap_min, task.gap_max + 1e-9, 10.0)
    gaps = torch.tensor(gap_grid, dtype=torch.float32)
    batch = sweep_batch(
        task,
        m_value,
        gaps,
        trials_per_gap,
        n_hidden=model.model.n_hidden,
        lapse_rate=float(model.lapse_rate(m_value)),
    )

    _, r, z = model(batch["u"], h0=batch["h0"], add_noise=add_noise)
    commit = hard_commitment(z, task)
    rpt = commit["t_commit"] - batch["t_cue"]
    return {
        "rpt": rpt,
        "correct": commit["p_goal"],  # hard 0/1
        "crossed": commit["crossed"],
        "r": r,
        "z": z,
        "cue_sides": batch["cue_sides"],
        "t_cue": batch["t_cue"],
        "gaps": batch["gaps"],
    }


def empirical_tachometric_curve(
    rpt: torch.Tensor,
    correct: torch.Tensor,
    grid: np.ndarray,
    bin_width: float = 15.0,
    min_count: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin trials by emergent rPT and return (tc, count) on ``grid``.

    Bins with fewer than ``min_count`` trials are returned as NaN.
    """
    rpt_np = rpt.detach().cpu().numpy()
    correct_np = correct.detach().cpu().numpy()
    tc = np.full(len(grid), np.nan)
    counts = np.zeros(len(grid))
    for j, center in enumerate(grid):
        sel = np.abs(rpt_np - center) <= bin_width / 2.0
        counts[j] = sel.sum()
        if counts[j] >= min_count:
            tc[j] = correct_np[sel].mean()
    return tc, counts


def _parametric_tc(rpt, A, t_rise, sigma_rise, t_vortex, D, sigma_vortex):
    rise = (A - 0.5) * norm.cdf((rpt - t_rise) / sigma_rise)
    vortex = D * np.exp(-0.5 * ((rpt - t_vortex) / sigma_vortex) ** 2)
    return np.clip(0.5 + rise - vortex, 0.0, 1.0)


def fit_summary_stats(grid: np.ndarray, tc: np.ndarray) -> dict:
    """Fit the parametric tachometric curve to (grid, tc) and return statistics.

    NaN bins are ignored. Returns a dict with A, t_rise, sigma_rise, t_vortex,
    D, sigma_vortex, plus the derived 75%-crossing ``t_rise75``.
    """
    valid = ~np.isnan(tc)
    x, y = grid[valid], tc[valid]
    p0 = [0.85, 150.0, 20.0, 108.0, 0.45, 15.0]
    bounds = ([0.5, 80.0, 5.0, 60.0, 0.0, 3.0], [1.0, 300.0, 80.0, 200.0, 0.6, 60.0])
    try:
        popt, _ = curve_fit(_parametric_tc, x, y, p0=p0, bounds=bounds, maxfev=20000)
    except (RuntimeError, ValueError):
        popt = p0
    A, t_rise, sigma_rise, t_vortex, D, sigma_vortex = popt

    # Derived 75%-crossing on the fitted curve (recovery branch).
    fine = np.linspace(grid.min(), grid.max(), 2000)
    curve = _parametric_tc(fine, *popt)
    rising = fine >= t_vortex
    cross_idx = np.argmax((curve >= 0.75) & rising)
    t_rise75 = float(fine[cross_idx]) if np.any((curve >= 0.75) & rising) else float("nan")

    return {
        "A": float(A),
        "t_rise": float(t_rise),
        "sigma_rise": float(sigma_rise),
        "t_vortex": float(t_vortex),
        "D": float(D),
        "sigma_vortex": float(sigma_vortex),
        "t_rise75": t_rise75,
    }


def model_tachometric(
    model: LRRNN,
    task: TaskParams,
    m_value: float,
    trials_per_gap: int = 200,
) -> dict:
    """Convenience: sweep, bin, and fit for one maturation state."""
    grid = task.rpt_grid
    sweep = run_gap_sweep(model, task, m_value, trials_per_gap)
    tc, counts = empirical_tachometric_curve(sweep["rpt"], sweep["correct"], grid)
    stats = fit_summary_stats(grid, tc)
    return {"grid": grid, "tc": tc, "counts": counts, "stats": stats, "sweep": sweep}
