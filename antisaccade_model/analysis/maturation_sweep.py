"""Evaluate the trained model across maturation states (gameplan Section 5.1).

Although training uses only m in {0, 1}, mechanism-1 conditioning lets us query
intermediate m to test whether behavioral/neural statistics vary monotonically.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..model.lrrnn import LRRNN
from ..task.task_params import TaskParams
from .spatial_signal import compute_si, select_fef_units
from .tachometric_analysis import model_tachometric


def maturation_sweep(
    model: LRRNN,
    task: TaskParams,
    m_values: Sequence[float] = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    trials_per_gap: int = 200,
    si_rpt_probe: Sequence[float] = (100.0, 150.0, 200.0),
) -> dict:
    """Sweep maturation and collect behavioral stats and SI probes.

    Returns a dict keyed by statistic name with arrays aligned to ``m_values``.
    """
    m_values = list(m_values)
    out = {
        "m_values": np.array(m_values),
        "A": [], "t_rise": [], "t_vortex": [], "D": [],
        "si_probe": {rpt: [] for rpt in si_rpt_probe},
    }

    for m in m_values:
        beh = model_tachometric(model, task, m, trials_per_gap)
        stats = beh["stats"]
        out["A"].append(stats["A"])
        out["t_rise"].append(stats.get("t_rise75", stats["t_rise"]))
        out["t_vortex"].append(stats["t_vortex"])
        out["D"].append(stats["D"])

        units = select_fef_units(model, task, m)
        si = compute_si(model, task, m, units, trials_per_gap=trials_per_gap, align="cue")
        # SI value shortly after cue onset for each probe rPT.
        t0 = np.argmin(np.abs(si["time"] - 0.0))
        for rpt in si_rpt_probe:
            b = int(np.argmin(np.abs(si["rpt_bins"] - rpt)))
            window = si["si"][b, t0:t0 + 50]
            out["si_probe"][rpt].append(float(np.nanmean(window)))

    for key in ("A", "t_rise", "t_vortex", "D"):
        out[key] = np.array(out[key])
    for rpt in si_rpt_probe:
        out["si_probe"][rpt] = np.array(out["si_probe"][rpt])
    return out
