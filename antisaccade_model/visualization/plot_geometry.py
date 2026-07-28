"""Population geometry plots: PC trajectories and low-rank mode activations."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_pc_trajectories(
    proj: np.ndarray,
    color_by: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    max_trials: int = 60,
) -> plt.Axes:
    """Plot PC1 vs PC2 trajectories for a subset of trials.

    Args:
        proj: ``[T, B, >=2]`` PCA projections.
        color_by: optional ``[B]`` values (e.g., rPT) to color trajectories.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    n_trials = min(max_trials, proj.shape[1])
    cmap = plt.get_cmap("viridis")
    for b in range(n_trials):
        c = cmap(color_by[b] / np.nanmax(color_by)) if color_by is not None else "gray"
        ax.plot(proj[:, b, 0], proj[:, b, 1], color=c, alpha=0.5, lw=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("State-space trajectories")
    return ax


def plot_mode_activations(
    kappa: np.ndarray,
    task_dt: float = 1.0,
    ax: Optional[plt.Axes] = None,
    label_prefix: str = "kappa",
) -> plt.Axes:
    """Plot trial-averaged low-rank mode activation time courses.

    Args:
        kappa: ``[T, B, R]`` mode activations.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    mean_kappa = np.nanmean(kappa, axis=1)   # [T, R]
    time = np.arange(mean_kappa.shape[0]) * task_dt
    for r in range(mean_kappa.shape[1]):
        ax.plot(time, mean_kappa[:, r], label=f"{label_prefix}_{r + 1}")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("mode activation")
    ax.set_title("Low-rank mode activations")
    ax.legend(fontsize=8)
    return ax
