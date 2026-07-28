"""Secondary / optional experiment (direction 2): fit SI, predict behavior.

The primary study uses direction 1 (fit behavior, predict SI; see
``run_behavior_fit``). This module is a documented scaffold for the reverse
cross-prediction described in gameplan Section 5.3: train a separate model
instance against an SI(t, rPT) target, then generate tachometric curves forward
and check whether the behavioral summary statistics are recovered.

It is intentionally minimal: the empirical SI target from Zhu et al. (2024) is
not bundled here, so ``load_si_target`` must be supplied by the user before this
can be run end to end.

Run (once an SI target is provided)::

    python -m antisaccade_model.experiments.run_neural_fit
"""

from __future__ import annotations

import numpy as np
import torch

from ..analysis.spatial_signal import compute_si, select_fef_units
from ..model.lrrnn import LRRNN
from ..model.model_params import DEFAULT_MODEL
from ..task.task_params import DEFAULT_TASK, TaskParams
from ..task.trial_generator import build_inputs
from ..training.curriculum import sample_curriculum_gaps
from ..training.losses import straight_through_commitment


def load_si_target(task: TaskParams, m_value: float) -> np.ndarray:
    """Return an empirical SI(t, rPT) target for maturation ``m_value``.

    Placeholder: wire this to the Zhu et al. (2024) data deposit (per-monkey SI
    maps aligned to cue onset on the analysis rPT grid).
    """
    raise NotImplementedError(
        "Provide an empirical SI(t, rPT) target array to run direction 2."
    )


def si_loss(model: LRRNN, task: TaskParams, m_value: float, si_target: torch.Tensor) -> torch.Tensor:
    """MSE between model SI and a target SI map (differentiable path optional).

    NOTE: ``compute_si`` uses hard binning and is not differentiable; a
    differentiable SI surrogate (soft rPT binning of pref/anti populations)
    would be required for gradient-based neural fitting. This function is a
    starting point for that extension.
    """
    units = select_fef_units(model, task, m_value)
    si = compute_si(model, task, m_value, units, align="cue")
    si_model = torch.tensor(np.nan_to_num(si["si"]), dtype=torch.float32)
    return torch.mean((si_model - si_target) ** 2)


def main() -> None:
    task = DEFAULT_TASK
    model = LRRNN(DEFAULT_MODEL, task)
    print(
        "Direction 2 is optional and not the primary study path.\n"
        "Supply an empirical SI target via load_si_target() and add a\n"
        "differentiable SI surrogate before running gradient-based fitting.\n"
        "Model and task are constructed and ready:", type(model).__name__,
    )
    # Sanity: a single forward pass to confirm the pipeline wires together.
    gaps = sample_curriculum_gaps(8, epoch=999, task=task)
    cue_sides = torch.randint(0, 2, (8,))
    m_values = torch.zeros(8)
    u, _ = build_inputs(gaps, cue_sides, m_values, task)
    _, _, z = model(u, add_noise=False)
    commit = straight_through_commitment(z, task)
    print("forward OK; fraction crossed:", float(commit["crossed"].float().mean()))


if __name__ == "__main__":
    main()
