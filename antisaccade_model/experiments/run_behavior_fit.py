"""Main experiment (direction 1): fit behavior, then predict SI.

Run as a module from the workspace root::

    python -m antisaccade_model.experiments.run_behavior_fit

Steps:
    1. Train the LR-RNN on young/adult tachometric summary statistics.
    2. Extract and plot model tachometric curves; report summary statistics.
    3. Predict SI(t, rPT) for young and adult (never trained on) and compare.
    4. Population geometry (PCA, participation ratio, mode activations).
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt

from ..analysis.geometry import (
    mode_activations,
    participation_ratio,
    pca_trajectories,
)
from ..analysis.spatial_signal import compute_si, select_fef_units, si_correlation
from ..analysis.tachometric_analysis import model_tachometric
from ..model.model_params import DEFAULT_MODEL
from ..task.task_params import DEFAULT_TASK
from ..task.tachometric_targets import target_summary_stats
from ..training.train import TrainConfig, load_checkpoint, train
from ..visualization.plot_geometry import plot_mode_activations
from ..visualization.plot_si import plot_si_heatmap
from ..visualization.plot_tc import plot_tachometric
from ._logging import tee_run_output
from .opt_behavior_fit import EvalConfig

OUT_DIR = "results/behavior_fit"


def _format_config_value(value):
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, tuple):
        return "[" + ",".join(_format_config_value(item) for item in value) + "]"
    if isinstance(value, list):
        return "[" + ",".join(_format_config_value(item) for item in value) + "]"
    if isinstance(value, dict):
        parts = [f"{key!r}: {_format_config_value(val)}" for key, val in value.items()]
        return "{" + ", ".join(parts) + "}"
    return repr(value)


def _log_resolved_configs(eval_cfg, model_params, task, train_cfg) -> None:
    print("Resolved full params:")
    for prefix, obj in (("eval", eval_cfg), ("model", model_params), ("task", task), ("train", train_cfg)):
        for field_name in obj.__dataclass_fields__:
            print(f"{prefix}.{field_name}={_format_config_value(getattr(obj, field_name))}")


def _print_stats_table(model, task) -> None:
    print("\nSummary statistics (model vs target):")
    print(f"{'stat':>10} {'m':>4} {'model':>10} {'target':>10}")
    for m in (0.0, 1.0):
        beh = model_tachometric(model, task, m)
        tgt = target_summary_stats(m, task)
        for key in ("A", "t_vortex", "D"):
            print(f"{key:>10} {m:>4.0f} {beh['stats'][key]:>10.3f} {float(tgt[key]):>10.3f}")
        print(f"{'t_rise':>10} {m:>4.0f} {beh['stats']['t_rise75']:>10.3f} "
              f"{float(tgt['t_rise']):>10.3f}")


def main(epochs: int = 1000, retrain: bool = True) -> None:
    with tee_run_output(os.path.join(OUT_DIR, "log.txt")):
        os.makedirs(OUT_DIR, exist_ok=True)
        task = DEFAULT_TASK
        model_params = DEFAULT_MODEL

        ckpt_path = "checkpoints/behavior_fit.pt"
        if retrain or not os.path.exists(ckpt_path):
            cfg = TrainConfig(epochs=epochs, checkpoint_path=ckpt_path)
            _log_resolved_configs(EvalConfig(), model_params, task, cfg)
            model, _ = train(cfg, model_params, task)
        else:
            model, _ = load_checkpoint(ckpt_path)

        # --- Behavior ---------------------------------------------------------
        beh_y = model_tachometric(model, task, 0.0)
        beh_a = model_tachometric(model, task, 1.0)
        ax = plot_tachometric(beh_y["grid"], beh_y["tc"], beh_a["tc"])
        ax.figure.savefig(os.path.join(OUT_DIR, "tachometric.png"), dpi=150, bbox_inches="tight")
        plt.close(ax.figure)
        _print_stats_table(model, task)

        # --- Neural prediction (SI) ------------------------------------------
        units_y = select_fef_units(model, task, 0.0)
        units_a = select_fef_units(model, task, 1.0)
        si_y = compute_si(model, task, 0.0, units_y, align="cue")
        si_a = compute_si(model, task, 1.0, units_a, align="cue")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plot_si_heatmap(si_y["si"], si_y["time"], si_y["rpt_bins"], ax=axes[0], title="SI young (m=0)")
        plot_si_heatmap(si_a["si"], si_a["time"], si_a["rpt_bins"], ax=axes[1], title="SI adult (m=1)")
        fig.savefig(os.path.join(OUT_DIR, "si_heatmaps.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSI young-vs-adult correlation: {si_correlation(si_y['si'], si_a['si']):.3f}")

        # --- Geometry ---------------------------------------------------------
        r = beh_a["sweep"]["r"]
        pca = pca_trajectories(r)
        pr = participation_ratio(r)
        print(f"Participation ratio (adult): {pr:.2f}")
        print(f"PC explained variance: {pca['explained_variance_ratio'][:3]}")

        kappa = mode_activations(model, r)
        ax = plot_mode_activations(kappa["kappa_n"].numpy(), task_dt=task.dt, label_prefix="kappa_n")
        ax.figure.savefig(os.path.join(OUT_DIR, "mode_activations.png"), dpi=150, bbox_inches="tight")
        plt.close(ax.figure)

        print(f"\nDone. Figures written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
