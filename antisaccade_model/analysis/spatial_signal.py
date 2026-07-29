"""FEF/dlPFC-analog spatial selectivity index SI(t, rPT) from hidden units.

Post-hoc neural prediction (never trained on). Steps follow gameplan Section 4:

1. Select spatially tuned "FEF-analog" units from their differential response to
   cue-left vs cue-right at long rPT.
2. Split them into left-preferring and right-preferring populations.
3. For a reference cue side, compute the population spatial signal
   ``SI = (R_pref - R_anti) / (R_pref + R_anti)`` as a function of time and rPT.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..model.lrrnn import LRRNN
from ..task.task_params import CUE_LEFT, CUE_RIGHT, TaskParams
from ..task.trial_generator import build_inputs, sample_initial_state
from ..training.losses import hard_commitment


@torch.no_grad()
def _run(model, task, gaps, cue_sides, m_value, add_noise):
    m_values = torch.full((gaps.shape[0],), float(m_value))
    lapse_mask = torch.rand(gaps.shape[0]) < float(model.lapse_rate(m_value))
    u, t_cue = build_inputs(gaps, cue_sides, m_values, task, lapse_mask=lapse_mask)
    h0 = sample_initial_state(gaps.shape[0], model.model.n_hidden, task)
    _, r, z = model(u, h0=h0, add_noise=add_noise)
    commit = hard_commitment(z, task)
    return r, commit["t_commit"], t_cue, commit["crossed"]


@torch.no_grad()
def select_fef_units(
    model: LRRNN,
    task: TaskParams,
    m_value: float,
    n_trials: int = 500,
    ref_gap: float = 0.0,
    window_ms: float = 50.0,
    top_fraction: float = 0.5,
) -> dict:
    """Select spatially tuned units from cue-left vs cue-right responses.

    ``ref_gap=0`` yields long rPT (goal-directed) trials, matching the gameplan.
    Returns indices of left-/right-preferring units and the tuning vector.
    """
    gaps = torch.full((n_trials,), float(ref_gap))
    r_left, _, t_cue_l, _ = _run(model, task, gaps, torch.full((n_trials,), CUE_LEFT), m_value, True)
    r_right, _, t_cue_r, _ = _run(model, task, gaps, torch.full((n_trials,), CUE_RIGHT), m_value, True)

    win = int(round(window_ms / task.dt))
    c0 = int(round(float(t_cue_l[0]) / task.dt))
    # Mean activity in [t_cue, t_cue+window], averaged over trials.
    mean_left = r_left[c0:c0 + win].mean(dim=(0, 1))    # [N]
    mean_right = r_right[c0:c0 + win].mean(dim=(0, 1))   # [N]
    tuning = (mean_left - mean_right)                    # [N]

    abs_t = tuning.abs()
    thresh = torch.quantile(abs_t, 1.0 - top_fraction)
    tuned = abs_t >= thresh
    left_pref = torch.where(tuned & (tuning > 0))[0]
    right_pref = torch.where(tuned & (tuning < 0))[0]
    return {
        "tuning": tuning,
        "tuned_mask": tuned,
        "left_pref_idx": left_pref,
        "right_pref_idx": right_pref,
    }


@torch.no_grad()
def compute_si(
    model: LRRNN,
    task: TaskParams,
    m_value: float,
    units: dict,
    rpt_bins: Optional[np.ndarray] = None,
    trials_per_gap: int = 200,
    align: str = "cue",
    pre_ms: float = 50.0,
    post_ms: float = 300.0,
    eps: float = 1e-6,
) -> dict:
    """Compute SI(t, rPT) for reference cue = right.

    Preferred = right-preferring units (cue-side); anti-preferred = left-
    preferring units (goal-side). Trials are binned by emergent rPT.

    Args:
        align: ``"cue"``, ``"go"``, or ``"saccade"`` alignment.

    Returns dict with ``si`` ``[n_bins, n_time]``, ``time`` (ms rel. to event),
    and ``rpt_bins``.
    """
    if rpt_bins is None:
        rpt_bins = task.rpt_grid

    # Sweep gaps to sample many rPTs, reference cue on the right.
    gap_grid = torch.arange(task.gap_min, task.gap_max + 1e-9, 10.0)
    gaps = gap_grid.repeat_interleave(trials_per_gap)
    cue_sides = torch.full((gaps.shape[0],), CUE_RIGHT)
    r, t_commit, t_cue, crossed = _run(model, task, gaps, cue_sides, m_value, True)

    rpt = (t_commit - t_cue).numpy()
    r_pref = r[:, :, units["right_pref_idx"]].mean(dim=-1)   # [T, B]
    r_anti = r[:, :, units["left_pref_idx"]].mean(dim=-1)    # [T, B]

    pre = int(round(pre_ms / task.dt))
    post = int(round(post_ms / task.dt))
    time = np.arange(-pre, post) * task.dt

    def event_step(trial):
        if align == "go":
            return task.go_step
        if align == "cue":
            return int(round(float(t_cue[trial]) / task.dt))
        if align == "saccade":
            return int(round(float(t_commit[trial]) / task.dt))
        raise ValueError(f"Unknown alignment: {align}")

    n_bins = len(rpt_bins)
    n_time = pre + post
    si = np.full((n_bins, n_time), np.nan)
    bin_width = float(task.rpt_step)

    for b, center in enumerate(rpt_bins):
        sel = np.where((np.abs(rpt - center) <= bin_width / 2.0) & crossed.numpy())[0]
        if len(sel) < 5:
            continue
        pref_stack, anti_stack = [], []
        for trial in sel:
            e = event_step(trial)
            lo, hi = e - pre, e + post
            if lo < 0 or hi > r.shape[0]:
                continue
            pref_stack.append(r_pref[lo:hi, trial].numpy())
            anti_stack.append(r_anti[lo:hi, trial].numpy())
        if not pref_stack:
            continue
        rp = np.mean(pref_stack, axis=0)
        ra = np.mean(anti_stack, axis=0)
        si[b] = (rp - ra) / (rp + ra + eps)

    return {"si": si, "time": time, "rpt_bins": np.asarray(rpt_bins), "align": align}


def si_correlation(si_young: np.ndarray, si_adult: np.ndarray) -> float:
    """Pearson correlation between two SI maps over their shared valid entries."""
    a, b = si_young.ravel(), si_adult.ravel()
    valid = ~(np.isnan(a) | np.isnan(b))
    if valid.sum() < 2:
        return float("nan")
    return float(np.corrcoef(a[valid], b[valid])[0, 1])
