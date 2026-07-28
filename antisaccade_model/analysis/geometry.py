"""Population geometry analyses: PCA trajectories, participation ratio,
low-rank mode activations, and pre-go goal bias (gameplan Section 4.3)."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA

from ..model.lrrnn import LRRNN
from ..task.task_params import TaskParams


def pca_trajectories(r: torch.Tensor, n_components: int = 3) -> dict:
    """Fit PCA on trial-averaged activity and return components and projections.

    Args:
        r: firing rates ``[T, B, N]``.

    Returns dict with ``pca`` (fitted object), ``proj`` ``[T, B, n_components]``,
    and ``explained_variance_ratio``.
    """
    n_steps, batch, n = r.shape
    flat = r.reshape(n_steps * batch, n).detach().cpu().numpy()
    pca = PCA(n_components=n_components)
    proj_flat = pca.fit_transform(flat)
    proj = proj_flat.reshape(n_steps, batch, n_components)
    return {
        "pca": pca,
        "proj": proj,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


def participation_ratio(r: torch.Tensor) -> float:
    """Participation ratio PR = (sum lambda)^2 / sum lambda^2 of the covariance."""
    n_steps, batch, n = r.shape
    flat = r.reshape(n_steps * batch, n).detach().cpu().numpy()
    cov = np.cov(flat, rowvar=False)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0.0, None)
    denom = (eig ** 2).sum()
    if denom <= 0:
        return float("nan")
    return float((eig.sum() ** 2) / denom)


@torch.no_grad()
def mode_activations(model: LRRNN, r: torch.Tensor) -> dict:
    """Project activity onto the low-rank connectivity modes M and N.

    Returns time courses ``kappa_n`` = (N^T r)/sqrt(N) and ``kappa_m`` =
    (M^T r)/sqrt(N), each ``[T, B, R]``. These are the interpretable collective
    variables of the low-rank RNN.
    """
    n = model.model.n_hidden
    m_modes = model.M.detach()   # [N, R]
    n_modes = model.N.detach()   # [N, R]
    kappa_n = torch.einsum("tbn,nr->tbr", r, n_modes) / (n ** 0.5)
    kappa_m = torch.einsum("tbn,nr->tbr", r, m_modes) / (n ** 0.5)
    return {"kappa_m": kappa_m, "kappa_n": kappa_n}


@torch.no_grad()
def goal_pre_bias(
    model: LRRNN,
    r: torch.Tensor,
    task: TaskParams,
    goal_units: torch.Tensor,
    cue_units: torch.Tensor,
    window_ms: float = 100.0,
) -> dict:
    """Mean pre-go activity in goal- vs cue-preferring units.

    A higher goal-minus-cue pre-go bias in the adult (m=1) network corresponds
    to the mature PFC's preparatory presetting toward the goal.
    """
    win = int(round(window_ms / task.dt))
    lo = max(0, task.go_step - win)
    pre = r[lo:task.go_step]                  # [win, B, N]
    goal_act = pre[:, :, goal_units].mean().item()
    cue_act = pre[:, :, cue_units].mean().item()
    return {"goal": goal_act, "cue": cue_act, "bias": goal_act - cue_act}
