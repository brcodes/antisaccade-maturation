"""Losses for behavior-only training with a straight-through decision rule.

Pipeline (per batch)::

    z(t)  --commitment-->  (p_goal, t_commit)  --> rPT = t_commit - t_cue
          --soft binning--> TC_model(rPT grid)  --> summary stats  --> MSE loss

The threshold-crossing decision is non-differentiable, so a straight-through
estimator is used: the forward pass returns the hard first-passage decision and
commitment time, while the backward pass uses a smooth first-passage proxy
(``p_goal_soft`` and ``t_commit_soft``). rPT is emergent (Section 1.1): gap is
imposed, the network's crossing sets ``t_commit``, and trials are binned by the
resulting rPT.

The behavioral objective is the summary-statistic loss on (t_rise, A, t_vortex,
D), extracted with the differentiable extractor in ``tachometric_targets``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..task.task_params import RULE_IDX, TOWARD_GOAL_IDX, TaskParams
from ..task.tachometric_targets import extract_summary_stats, params_for_m, tachometric_curve


def soft_commitment(z: torch.Tensor, task: TaskParams) -> dict:
    """Differentiable soft first-passage decision.

    Args:
        z: outputs ``[T, B, 2]`` = [toward_cue, toward_goal].
        task: task parameters (threshold, temperatures, dt).

    Returns:
        Dict with ``p_goal`` ``[B]`` (prob. of committing to goal), ``t_commit``
        ``[B]`` (ms), plus internals ``w`` (commit density ``[T, B]``) and
        ``surv`` (probability of never crossing, ``[B]``).
    """
    n_steps = z.shape[0]
    time = torch.arange(n_steps, dtype=torch.float32) * task.dt  # [T]

    # Soft "value" of the leading option = smooth max over the two outputs.
    a = task.option_temp * torch.logsumexp(z / task.option_temp, dim=-1)  # [T, B]

    # Per-step probability the threshold is exceeded.
    c = torch.sigmoid((a - task.threshold) / task.commit_temp)            # [T, B]
    one_minus = torch.clamp(1.0 - c, min=1e-6)

    # Survival to *before* step t: prod_{k<t} (1 - c_k).
    surv_incl = torch.cumprod(one_minus, dim=0)                            # through t
    surv_prev = torch.cat([torch.ones(1, z.shape[1]), surv_incl[:-1]], 0)  # before t
    w = surv_prev * c                                                      # commit density
    surv_all = surv_incl[-1]                                               # never cross

    # Soft commitment time (deadline mass placed at the last step).
    t_commit = (w * time[:, None]).sum(0) + surv_all * time[-1]

    # Soft decision probability (goal) integrated over the commit density.
    p_goal_step = torch.softmax(z / task.option_temp, dim=-1)[..., TOWARD_GOAL_IDX]  # [T, B]
    p_goal = (w * p_goal_step).sum(0) + surv_all * p_goal_step[-1]

    return {"p_goal": p_goal, "t_commit": t_commit, "w": w, "surv": surv_all}


def hard_commitment(z: torch.Tensor, task: TaskParams) -> dict:
    """Non-differentiable hard first-passage decision (forward-pass truth).

    Returns ``p_goal`` (0/1), ``t_commit`` (ms), and ``crossed`` (bool mask).
    """
    n_steps, batch, _ = z.shape
    a = z.max(dim=-1).values                       # [T, B]
    exceeded = a > task.threshold                  # [T, B]
    crossed = exceeded.any(dim=0)                  # [B]
    # argmax returns the first index of the max; on all-False columns it is 0,
    # which we override with the deadline below.
    first = exceeded.float().argmax(dim=0)         # [B]
    commit_step = torch.where(crossed, first, torch.full_like(first, n_steps - 1))

    batch_idx = torch.arange(batch)
    winner = z[commit_step, batch_idx].argmax(dim=-1)   # 0=cue, 1=goal
    p_goal = (winner == TOWARD_GOAL_IDX).float()
    t_commit = commit_step.float() * task.dt
    return {"p_goal": p_goal, "t_commit": t_commit, "crossed": crossed}


def straight_through_commitment(z: torch.Tensor, task: TaskParams) -> dict:
    """Straight-through decision: hard forward value, soft backward gradient."""
    soft = soft_commitment(z, task)
    hard = hard_commitment(z, task)
    p_goal = hard["p_goal"] + (soft["p_goal"] - soft["p_goal"].detach())
    t_commit = hard["t_commit"] + (soft["t_commit"] - soft["t_commit"].detach())
    return {
        "p_goal": p_goal,
        "t_commit": t_commit,
        "crossed": hard["crossed"],
        "surv": soft["surv"],
    }


def soft_tachometric_curve(
    p_goal: torch.Tensor,
    rpt: torch.Tensor,
    grid: torch.Tensor,
    bin_width: float,
) -> torch.Tensor:
    """Differentiable soft-binned tachometric curve.

    Each trial contributes to every grid bin with a Gaussian weight in rPT,
    yielding ``TC(grid_j) = sum_i w_ij p_goal_i / sum_i w_ij``.

    Args:
        p_goal: ``[B]`` probability correct per trial.
        rpt: ``[B]`` emergent rPT per trial (ms).
        grid: ``[G]`` rPT bin centers (ms).
        bin_width: Gaussian kernel width (ms).

    Returns:
        ``[G]`` proportion-correct curve.
    """
    eps = 1e-8
    d = rpt[None, :] - grid[:, None]                 # [G, B]
    weights = torch.exp(-0.5 * (d / bin_width) ** 2)  # [G, B]
    num = (weights * p_goal[None, :]).sum(dim=1)
    den = weights.sum(dim=1) + eps
    return num / den


def rpt_weight(rpt: torch.Tensor) -> torch.Tensor:
    """Piecewise rPT weighting that emphasizes the developmental transition."""
    weights = torch.full_like(rpt, 0.5)
    weights = torch.where((rpt >= 70.0) & (rpt <= 200.0), torch.full_like(weights, 3.0), weights)
    weights = torch.where((rpt > 200.0) & (rpt <= 300.0), torch.full_like(weights, 1.0), weights)
    return weights


def summary_stat_loss(
    stats_model: dict,
    stats_target: dict,
    weights: dict | None = None,
) -> torch.Tensor:
    """MSE between model and target summary statistics.

    Statistics are on different scales (times in ms, accuracies in [0,1]); the
    default weights normalize the time-valued stats so they are comparable to
    the accuracy-valued ones.
    """
    if weights is None:
        # 1/ms^2 scaling for times keeps terms O(1); accuracies use unit weight.
        weights = {"t_rise": 1e-4, "A": 1.0, "t_vortex": 1e-4, "D": 1.0}
    loss = torch.zeros(())
    for key in ("t_rise", "A", "t_vortex", "D"):
        loss = loss + weights[key] * (stats_model[key] - stats_target[key]) ** 2
    return loss


def regularization(model, r: torch.Tensor, lambda_reg: float) -> torch.Tensor:
    """Frobenius-norm penalty on W_rec plus a mean-squared-activity penalty."""
    w_rec = model.recurrent_matrix()
    return lambda_reg * ((w_rec ** 2).sum() + (r ** 2).mean())


def behavioral_loss(
    model,
    batch: dict,
    task: TaskParams,
    targets: dict[float, dict],
    grid: torch.Tensor,
    extractor_kwargs: dict | None = None,
) -> tuple[torch.Tensor, dict]:
    """Full behavior-only loss for one batch.

    ``targets`` maps each training maturation value (0.0, 1.0) to its target
    summary statistics (from :func:`target_summary_stats`).
    """
    extractor_kwargs = extractor_kwargs or {}
    _, r, z = model(batch["u"], h0=batch.get("h0"), add_noise=True)
    soft_commit = soft_commitment(z, task)
    commit = straight_through_commitment(z, task)
    rpt = commit["t_commit"] - batch["t_cue"]

    u_lapse = batch["u"].clone()
    u_lapse[:, :, RULE_IDX] = 0.0
    _, _, z_lapse = model(u_lapse, h0=batch.get("h0"), add_noise=True)
    soft_lapse = soft_commitment(z_lapse, task)

    beh_loss = torch.zeros(())
    per_m = {}
    curve_loss = torch.zeros(())
    for m_value, tgt in targets.items():
        mask = (batch["m"] == m_value)
        if mask.sum() < 2:
            continue
        lambda_m = model.lapse_rate(float(m_value))
        p_goal_mix = (1.0 - lambda_m) * soft_commit["p_goal"][mask] + lambda_m * soft_lapse["p_goal"][mask]
        rpt_mix = (1.0 - lambda_m) * rpt[mask] + lambda_m * (soft_lapse["t_commit"][mask] - batch["t_cue"][mask])
        tc = soft_tachometric_curve(p_goal_mix, rpt_mix, grid, task.rpt_bin_width)
        stats = extract_summary_stats(tc, grid, **extractor_kwargs)
        beh_loss = beh_loss + summary_stat_loss(stats, tgt)
        target_curve = tachometric_curve(rpt_mix, params_for_m(float(m_value)))
        trial_bce = F.binary_cross_entropy(p_goal_mix, target_curve, reduction="none")
        curve_loss = curve_loss + (rpt_weight(rpt_mix) * trial_bce).mean()
        per_m[m_value] = {"tc": tc.detach(), "stats": {k: v.detach() for k, v in stats.items()}}

    reg = regularization(model, r, model.model.lambda_reg)
    total = beh_loss + curve_loss + reg
    info = {
        "behavior": beh_loss.detach(),
        "curve": curve_loss.detach(),
        "reg": reg.detach(),
        "total": total.detach(),
        "per_m": per_m,
        "frac_crossed": commit["crossed"].float().mean().detach(),
    }
    return total, info
