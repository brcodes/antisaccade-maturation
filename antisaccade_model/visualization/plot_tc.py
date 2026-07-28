"""Tachometric curve plots (model vs. target, young vs. adult)."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..task.tachometric_targets import params_for_m, tachometric_curve


def plot_tachometric(
    grid: np.ndarray,
    tc_young: np.ndarray,
    tc_adult: np.ndarray,
    show_targets: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot model tachometric curves for young and adult, with target overlays."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    ax.plot(grid, tc_young, "o-", color="tab:blue", label="model young (m=0)")
    ax.plot(grid, tc_adult, "o-", color="tab:red", label="model adult (m=1)")

    if show_targets:
        g = torch.tensor(grid, dtype=torch.float32)
        ty = tachometric_curve(g, params_for_m(0.0)).numpy()
        ta = tachometric_curve(g, params_for_m(1.0)).numpy()
        ax.plot(grid, ty, "--", color="tab:blue", alpha=0.6, label="target young")
        ax.plot(grid, ta, "--", color="tab:red", alpha=0.6, label="target adult")

    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("raw processing time rPT (ms)")
    ax.set_ylabel("proportion correct")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("Tachometric curves")
    return ax
