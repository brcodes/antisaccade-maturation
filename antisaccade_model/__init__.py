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

# --- OpenMP / BLAS runtime guard -------------------------------------------
# Some conda environments ship a numpy/SciPy BLAS whose OpenMP runtime conflicts
# with the one bundled by PyTorch, causing a hard segmentation fault (e.g. inside
# numpy.linalg.svd via scikit-learn's PCA) once both are loaded. These settings
# must be applied *before* torch / numpy are first imported, so they live here at
# package import time. ``setdefault`` keeps any values the user has already set.
# The model is small (N=200) and the per-timestep loop dominates, so pinning BLAS
# to a single thread has no meaningful performance cost.
import os as _os

_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")

__all__ = ["task", "model", "training", "analysis", "visualization"]
