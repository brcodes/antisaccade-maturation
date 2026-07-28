"""Spatial signal SI(t, rPT) plots, matched to Zhu et al. (2024) Fig. 7 style."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_si_heatmap(
    si: np.ndarray,
    time: np.ndarray,
    rpt_bins: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "SI(t, rPT)",
) -> plt.Axes:
    """Heatmap of the spatial signal over time (x) and rPT bin (y)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    vmax = np.nanmax(np.abs(si)) if np.isfinite(si).any() else 1.0
    im = ax.imshow(
        si, aspect="auto", origin="lower", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        extent=[time[0], time[-1], rpt_bins[0], rpt_bins[-1]],
    )
    ax.axvline(0.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("time from event (ms)")
    ax.set_ylabel("rPT (ms)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="SI")
    return ax


def plot_si_traces(
    si_young: np.ndarray,
    si_adult: np.ndarray,
    time: np.ndarray,
    rpt_bins: np.ndarray,
    probe_rpts=(100.0, 150.0, 200.0),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Overlay SI time courses at selected rPT bins for young vs. adult."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    for rpt in probe_rpts:
        b = int(np.argmin(np.abs(rpt_bins - rpt)))
        ax.plot(time, si_young[b], color="tab:blue", alpha=0.4 + 0.2 * (rpt / 200))
        ax.plot(time, si_adult[b], color="tab:red", alpha=0.4 + 0.2 * (rpt / 200),
                label=f"rPT={rpt:.0f}")
    ax.axhline(0.0, color="gray", lw=0.8, ls=":")
    ax.axvline(0.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("time from event (ms)")
    ax.set_ylabel("SI")
    ax.set_title("SI traces (blue=young, red=adult)")
    ax.legend(fontsize=8)
    return ax
