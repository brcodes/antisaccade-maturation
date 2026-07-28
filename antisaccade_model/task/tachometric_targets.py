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
    t_rise: float     # ms, rPT at which accuracy crosses 75% (the rise point).
                      # NOTE: this is the 75% crossing, NOT the sigmoid midpoint.
                      # The curve is constructed to cross 0.75 at exactly this rPT.
    sigma_rise: float # ms, width of the recovery
    t_vortex: float   # ms, location of the vortex minimum
    D: float          # vortex depth below chance
    sigma_vortex: float  # ms, width of the vortex


# Values from Zhu et al. (2024) Fig. 3: t_rise is the 75%-correct crossing
# (young = 155 ms, adult = 140 ms). Claude looking at 3B, and or info from that Fig neighborhood.
YOUNG_PARAMS = TCParams(A=0.92, t_rise=155.0, sigma_rise=25.0,
                        t_vortex=105.0, D=0.28, sigma_vortex=25.0)
ADULT_PARAMS = TCParams(A=0.97, t_rise=140.0, sigma_rise=15.0,
                        t_vortex=106.0, D=0.27, sigma_vortex=20.0)


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


def _probit(p: float) -> float:
    """Inverse standard-normal CDF (quantile function)."""
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return float(torch.erfinv(torch.tensor(2.0 * p - 1.0)) * math.sqrt(2.0))


def _rise_midpoint(params: TCParams, rise_level: float = 0.75) -> float:
    """Sigmoid midpoint that makes the curve cross ``rise_level`` at ``t_rise``.

    ``params.t_rise`` is defined as the 75% crossing, so we solve for the latent
    sigmoid midpoint ``mu`` such that ``TC(t_rise) = rise_level`` (accounting for
    the small vortex contribution at that rPT)::

        rise_level = 0.5 + (A - 0.5) * Phi((t_rise - mu) / sigma_rise) - vortex(t_rise)
    """
    vortex_at_cross = params.D * math.exp(
        -0.5 * ((params.t_rise - params.t_vortex) / params.sigma_vortex) ** 2
    )
    target_phi = (rise_level - 0.5 + vortex_at_cross) / (params.A - 0.5)
    return params.t_rise - params.sigma_rise * _probit(target_phi)


def tachometric_curve(rpt: torch.Tensor, params: TCParams) -> torch.Tensor:
    """Evaluate the parametric target tachometric curve at rPT values (ms).

    The curve is constructed so that it crosses 0.75 at exactly ``params.t_rise``
    (the empirical 75% rise point), not at the latent sigmoid midpoint.
    """
    midpoint = _rise_midpoint(params)
    rise = (params.A - 0.5) * _normal_cdf((rpt - midpoint) / params.sigma_rise)
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

    # Rise point = rPT where accuracy crosses ``rise_level`` (0.75), i.e. the
    # first upward crossing of the curve. On the monotonic recovery branch this
    # is identical to ``argmin(|tc - 0.75|)``. This is the 75% crossing, NOT the
    # sigmoid midpoint.
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

    A, t_vortex, and D are read from the parametric curve with the same
    extractor used on the model (consistent definitions). t_rise is set to the
    exact empirical 75% crossing (``params.t_rise``) so the behavioral target is
    anchored to the Zhu et al. (2024) rise points (young = 155 ms, adult = 140 ms)
    and free of any sigmoid-midpoint / grid-quantization bias.
    """
    grid = torch.tensor(task.rpt_grid, dtype=torch.float32)
    params = params_for_m(m)
    tc = tachometric_curve(grid, params)
    stats = extract_summary_stats(tc, grid, **extractor_kwargs)
    stats["t_rise"] = torch.tensor(float(params.t_rise))
    return {k: v.detach() for k, v in stats.items()}
