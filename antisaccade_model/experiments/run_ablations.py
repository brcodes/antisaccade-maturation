"""Ablation analyses (gameplan Section 5.2), adapted to maturation mechanism 1.

The gameplan's original ablations 1-2 target conditioning mechanisms 2 (initial
state) and 3 (exogenous gain), which are not part of this mechanism-1-only
model. The mechanism-appropriate ablations here isolate the contribution of the
maturation input channel and the two low-rank modes.

Run::

    python -m antisaccade_model.experiments.run_ablations
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from ..analysis.spatial_signal import compute_si, select_fef_units
from ..analysis.tachometric_analysis import model_tachometric
from ..task.task_params import MATURATION_IDX
from ..training.train import load_checkpoint


def ablate_maturation_input(model):
    """Return a copy of the model with the maturation input weights zeroed.

    Young and adult curves should then collapse together, quantifying how much
    behavior is driven by the maturation channel.
    """
    m2 = copy.deepcopy(model)
    with torch.no_grad():
        m2.W_in[:, MATURATION_IDX] = 0.0
    return m2


def ablate_mode(model, mode_idx: int):
    """Return a copy of the model with one low-rank mode removed (rank -> 1)."""
    m2 = copy.deepcopy(model)
    with torch.no_grad():
        m2.M[:, mode_idx] = 0.0
        m2.N[:, mode_idx] = 0.0
    return m2


def _report_behavior(model, task, label: str) -> None:
    print(f"\n[{label}]")
    for m in (0.0, 1.0):
        stats = model_tachometric(model, task, m)["stats"]
        print(f"  m={m:.0f}: A={stats['A']:.3f} t_rise75={stats['t_rise75']:.1f} "
              f"t_vortex={stats['t_vortex']:.1f} D={stats['D']:.3f}")


def main(ckpt_path: str = "checkpoints/behavior_fit.pt") -> None:
    model, ckpt = load_checkpoint(ckpt_path)
    task = ckpt["task"]

    _report_behavior(model, task, "intact")
    _report_behavior(ablate_maturation_input(model), task, "maturation-input lesion")
    _report_behavior(ablate_mode(model, 0), task, "mode-0 lesion (rank 1)")
    _report_behavior(ablate_mode(model, 1), task, "mode-1 lesion (rank 1)")

    # Spatial-signal sensitivity to mode lesions (adult).
    for label, mdl in (("intact", model),
                       ("mode-0 lesion", ablate_mode(model, 0)),
                       ("mode-1 lesion", ablate_mode(model, 1))):
        units = select_fef_units(mdl, task, 1.0)
        si = compute_si(mdl, task, 1.0, units, align="cue")
        mean_si = float(np.nanmean(si["si"])) if si["si"].size else float("nan")
        print(f"[SI adult | {label}] mean SI = {mean_si:.3f}")


if __name__ == "__main__":
    main()
