"""Antisaccade maturation modeling: low-rank RNN conditioned on a maturation scalar.

Package layout follows Section 6.1 of the implementation gameplan:
    task/         trial generation and behavioral (tachometric) targets
    model/        low-rank RNN and readout
    training/     training loop, losses, curriculum
    analysis/     tachometric fit, spatial signal (SI), population geometry
    visualization/ plotting utilities
    experiments/  runnable entry points

Design decisions (fixed for this implementation):
    * Connectivity rank R = 2.
    * Maturation conditioning mechanism 1 only (m enters as a constant input channel).
    * Non-differentiable threshold crossing handled with a straight-through estimator
      (hard decision in the forward pass, soft first-passage proxy in the backward pass).
    * Behavioral objective = summary-statistic loss on (t_rise, A, t_vortex, D),
      extracted with a differentiable soft extractor during training.
    * Training maturation states are discrete: m in {0, 1}.
    * rPT is emergent: gap is imposed, the network's threshold crossing sets t_commit,
      and rPT = t_commit - t_cue is computed post-hoc and binned.
    * CPU only.
"""

__all__ = ["task", "model", "training", "analysis", "visualization"]
