"""Parametric behavioral targets (tachometric curves) for young and adult states.

The tachometric curve is proportion-correct as a function of raw processing
time (rPT). It has four phases: a guessing plateau at chance (0.5), an
exogenous-capture vortex dipping below chance, a sigmoidal recovery, and an
asymptote below 1.0.

We parameterize it as a chance baseline plus a sigmoidal rise minus a Gaussian
vortex::

    TC(rPT) = 0.5
            + (A - 0.5) * Phi((rPT - t_rise) / sigma_rise)      # sigmoidal recovery
            - D * exp(-0.5 * ((rPT - t_vortex) / sigma_vortex)^2) # exogenous vortex

with ``Phi`` the standard normal CDF. Parameter values are from Zhu et al. (2024)
Figs. 3-4 (see the gameplan, Section 3.1).

The four fit-target summary statistics (t_rise, A, t_vortex, D) are extracted
from *both* the target curve and the model curve with the *same* differentiable
extractor (``extract_summary_stats``), so their definitions are guaranteed
consistent and the training loss is well posed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .task_params import TaskParams


@dataclass
class TCParams:
    """Parameters of a single parametric tachometric curve."""

    A: float          # asymptotic accuracy (plateau)
    t_rise: float     # ms, sigmoid midpoint of the recovery
    sigma_rise: float # ms, width of the recovery
    t_vortex: float   # ms, location of the vortex minimum
    D: float          # vortex depth below chance
    sigma_vortex: float  # ms, width of the vortex


# Values from the gameplan Section 3.1 (Zhu et al. 2024).
YOUNG_PARAMS = TCParams(A=0.80, t_rise=170.0, sigma_rise=25.0,
                        t_vortex=110.0, D=0.50, sigma_vortex=15.0)
ADULT_PARAMS = TCParams(A=0.92, t_rise=145.0, sigma_rise=20.0,
                        t_vortex=105.0, D=0.42, sigma_vortex=15.0)


def params_for_m(m: float) -> TCParams:
    """Return the target curve parameters for a maturation state.

    Training uses only m in {0, 1}; intermediate values (analysis only) are a
    linear interpolation of the endpoint parameters.
    """
    if m <= 0.0:
        return YOUNG_PARAMS
    if m >= 1.0:
        return ADULT_PARAMS
    y, a = YOUNG_PARAMS, ADULT_PARAMS
    lerp = lambda p, q: p + m * (q - p)  # noqa: E731
    return TCParams(
        A=lerp(y.A, a.A),
        t_rise=lerp(y.t_rise, a.t_rise),
        sigma_rise=lerp(y.sigma_rise, a.sigma_rise),
        t_vortex=lerp(y.t_vortex, a.t_vortex),
        D=lerp(y.D, a.D),
        sigma_vortex=lerp(y.sigma_vortex, a.sigma_vortex),
    )


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def tachometric_curve(rpt: torch.Tensor, params: TCParams) -> torch.Tensor:
    """Evaluate the parametric target tachometric curve at rPT values (ms)."""
    rise = (params.A - 0.5) * _normal_cdf((rpt - params.t_rise) / params.sigma_rise)
    vortex = params.D * torch.exp(-0.5 * ((rpt - params.t_vortex) / params.sigma_vortex) ** 2)
    return torch.clamp(0.5 + rise - vortex, 0.0, 1.0)


def extract_summary_stats(
    tc: torch.Tensor,
    grid: torch.Tensor,
    asymptote_min_rpt: float = 200.0,
    vortex_max_rpt: float = 160.0,
    rise_level: float = 0.75,
    temp: float = 0.05,
) -> dict:
    """Differentiable extraction of the four summary statistics from a curve.

    Args:
        tc: ``[G]`` proportion-correct values on ``grid``.
        grid: ``[G]`` rPT bin centers (ms), assumed increasing.
        asymptote_min_rpt: rPTs at/above this define the asymptote region.
        vortex_max_rpt: rPTs at/below this define the vortex search region.
        rise_level: accuracy level defining the rise point (0.75 by default).
        temp: softness of the soft-min / soft-argmin operators.

    Returns:
        Dict with tensors ``A``, ``t_rise``, ``t_vortex``, ``D`` (all scalars).
    """
    eps = 1e-8

    # Asymptote A: mean over the long-rPT region.
    asym_mask = (grid >= asymptote_min_rpt).float()
    A = (tc * asym_mask).sum() / (asym_mask.sum() + eps)

    # Vortex: soft-min of the curve over the short-rPT region.
    vortex_mask = (grid <= vortex_max_rpt).float()
    masked_tc = torch.where(vortex_mask > 0, tc, torch.full_like(tc, 1e9))
    soft_min_w = torch.softmax(-masked_tc / temp, dim=0)
    tc_vortex = (soft_min_w * tc).sum()
    t_vortex = (soft_min_w * grid).sum()
    D = 0.5 - tc_vortex  # depth below chance

    # Rise point: first grid rPT where the curve rises above ``rise_level``.
    above = torch.sigmoid((tc - rise_level) / temp)          # ~1 where above level
    below_prev = torch.cumprod(
        torch.cat([torch.ones(1), 1.0 - above[:-1]]), dim=0
    )                                                        # all previous below
    first_cross = above * below_prev                         # weight on first crossing
    denom = first_cross.sum() + eps
    t_rise = (first_cross * grid).sum() / denom
    # Fallback: if the curve never crosses, use the last grid point.
    never = torch.clamp(1.0 - first_cross.sum(), 0.0, 1.0)
    t_rise = t_rise + never * grid[-1]

    return {"A": A, "t_rise": t_rise, "t_vortex": t_vortex, "D": D}


def target_summary_stats(m: float, task: TaskParams, **extractor_kwargs) -> dict:
    """Target summary statistics for a maturation state.

    The parametric curve is sampled on the analysis rPT grid and passed through
    the same extractor used on the model, guaranteeing consistent definitions.
    """
    grid = torch.tensor(task.rpt_grid, dtype=torch.float32)
    tc = tachometric_curve(grid, params_for_m(m))
    stats = extract_summary_stats(tc, grid, **extractor_kwargs)
    return {k: v.detach() for k, v in stats.items()}
