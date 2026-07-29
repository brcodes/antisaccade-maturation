"""Task constants and timing parameters for the compelled antisaccade task.

All times are in milliseconds. The simulation timeline is::

    [ baseline: t_pre ][ go signal @ t=0 ................. t_post ]
    index 0            index go_step                       index n_steps-1

The go signal turns on at ``go_step``. The cue turns on ``gap`` ms after the go
signal (``t_cue = t_pre + gap``). ``gap`` is the experimentally controlled
independent variable; the raw processing time ``rPT = t_commit - t_cue`` is an
emergent quantity read out from the network's threshold crossing.
"""

from dataclasses import dataclass

import numpy as np

# --- Input channel indices (N_INPUT = 5), mechanism-1 conditioning ---------
GO_IDX = 0          # go signal (step from go_step onward)
CUE_LEFT_IDX = 1    # cue on the left (exogenous burst + sustained)
CUE_RIGHT_IDX = 2   # cue on the right
RULE_IDX = 3        # task rule: antisaccade (constant 1.0)
MATURATION_IDX = 4  # maturation scalar m in [0, 1] (constant per trial)
N_INPUT = 5

# --- Output channel indices (decision is in a cue/goal reference frame) -----
TOWARD_CUE_IDX = 0   # reflexive / prosaccade error direction
TOWARD_GOAL_IDX = 1  # correct antisaccade direction
N_OUTPUT = 2

# --- Cue side encoding ------------------------------------------------------
CUE_LEFT = 0
CUE_RIGHT = 1


@dataclass
class TaskParams:
    """Container for task timing, cue, decision, and noise parameters."""

    # integration
    dt: float = 1.0        # ms per step
    tau: float = 10.0      # ms membrane time constant

    # trial timeline
    t_pre: float = 100.0   # ms baseline before the go signal (for pre-go analyses)
    t_post: float = 500.0  # ms after the go signal
    gap_min: float = 0.0   # ms
    gap_max: float = 350.0 # ms

    # exogenous cue burst (Salinas et al. 2019 element that produces the vortex)
    a_exo: float = 3.0     # burst amplitude
    tau_exo: float = 30.0  # ms burst decay
    cue_sustained: float = 1.0  # sustained cue drive after onset

    # race-to-threshold decision
    threshold: float = 1.0      # commitment threshold theta on max output
    commit_temp: float = 0.2    # steepness of the soft (backward) threshold crossing
    option_temp: float = 0.2    # softmax temperature for the soft decision proxy

    # recurrent noise standard deviation (free parameter)
    sigma_noise: float = 0.1

    # Trial-to-trial initial-state variability shared across or private to units.
    sigma_init_shared: float = 0.7
    sigma_init_private: float = 0.05

    # analysis rPT grid
    rpt_min: float = 0.0
    rpt_max: float = 300.0
    rpt_step: float = 10.0
    rpt_bin_width: float = 12.0  # ms; kernel width for soft rPT binning in training

    @property
    def n_steps(self) -> int:
        """Total number of integration steps."""
        return int(round((self.t_pre + self.t_post) / self.dt))

    @property
    def go_step(self) -> int:
        """Integration step at which the go signal turns on (t = 0)."""
        return int(round(self.t_pre / self.dt))

    @property
    def rpt_grid(self) -> np.ndarray:
        """Evenly spaced rPT bin centers used for analysis and soft binning."""
        return np.arange(self.rpt_min, self.rpt_max + 1e-9, self.rpt_step)


DEFAULT_TASK = TaskParams()
