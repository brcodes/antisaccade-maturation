"""Architecture hyperparameters for the low-rank RNN."""

from dataclasses import dataclass


@dataclass
class ModelParams:
    """Hyperparameters for :class:`~antisaccade_model.model.lrrnn.LRRNN`."""

    n_input: int = 5     # go, cue_left, cue_right, rule, maturation
    n_hidden: int = 200  # recurrent units
    n_rank: int = 2      # connectivity rank R (fixed at 2)
    n_output: int = 2    # toward_cue, toward_goal
    phi: str = "tanh"    # nonlinearity: "tanh" or "relu"

    # W_rec modes are initialized at scale init_rec_scale / sqrt(N) to start in
    # the stable near-linear regime (gameplan Section 3.3).
    init_rec_scale: float = 0.1

    # Regularization weight for ||W_rec||_F^2 + mean(r^2) (gameplan Section 3.2).
    lambda_reg: float = 1e-4

    # Learned lapse endpoints, initialized from the gameplan priors.
    lapse_young_init: float = 0.08
    lapse_adult_init: float = 0.02


DEFAULT_MODEL = ModelParams()
