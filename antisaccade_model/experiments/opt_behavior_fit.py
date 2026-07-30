"""Parameterized experiment driver for behavior fitting: smoke tests + sweeps.

This is the iteration-speed entrypoint. It wraps the training loop
(``training.train.train``) and the tachometric analysis so you can quickly try
configurations, run a coarse fast "smoke" preset, and sweep optimization /
architecture hyperparameters, with all diagnostics saved to disk.

Two run modes
-------------
* single run (default): train once with the given config and write a
  tachometric-curve figure plus a metrics JSON.
* sweep (``--sweep key=v1,v2 ...``): expand the Cartesian product of the swept
  values, run each, and write a sorted ``results.csv`` and ``best_config.json``.

Configuration is layered, applied in order:
    defaults  ->  --preset  ->  explicit flags  ->  --set k=v  ->  --sweep k=v1,v2

Any field of the underlying configs can be addressed with a dotted key:
    ``task.<field>``   e.g. task.threshold, task.a_exo, task.t_post
    ``model.<field>``  e.g. model.n_hidden, model.n_rank, model.lambda_reg
    ``train.<field>``  e.g. train.lr, train.epochs, train.batch_size
    ``eval.<field>``   e.g. eval.trials_per_gap

Examples
--------
Fast smoke test (small net, short horizon, few epochs)::

    python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke

Single run with a lower threshold and stronger exogenous burst::

    python -m antisaccade_model.experiments.opt_behavior_fit \
        --preset smoke --threshold 0.4 --a-exo 5

Grid search over learning rate, threshold, and burst amplitude::

    python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
        --sweep train.lr=1e-3,3e-4 --sweep task.threshold=0.3,0.5 \
        --sweep task.a_exo=3,5 --top 5
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any

import numpy as np
import torch

from ..analysis.tachometric_analysis import model_tachometric
from ..model.model_params import ModelParams
from ..task.task_params import TaskParams
from ..task.tachometric_targets import target_summary_stats
from ..training.train import TrainConfig, train
from ..visualization.plot_tc import plot_tachometric
from ._logging import tee_run_output

logger = logging.getLogger("opt_behavior_fit")

# Weights matching training.losses.summary_stat_loss (times in ms use 1e-4).
STAT_WEIGHTS = {"t_rise": 1e-4, "A": 1.0, "t_vortex": 1e-4, "D": 1.0}
VORTEX_MAX_RPT = 160.0  # rPTs at/below this define the vortex search region


@dataclass
class EvalConfig:
    """Post-training evaluation settings for tachometric-curve extraction."""

    trials_per_gap: int = 100
    m_values: tuple[float, ...] = (0.0, 1.0)


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #
# A preset is a dict of dotted-key -> value overrides applied on top of defaults.
PRESETS: dict[str, dict[str, Any]] = {
    # Fast debugging loop: is it learning, does the vortex appear, are trials
    # crossing threshold? Trim dead time and shrink everything.
    "smoke": {
        "model.n_hidden": 64,
        "task.t_pre": 100.0,
        "task.t_post": 500.0,
        "task.gap_max": 350.0,
        "task.rpt_max": 350.0,
        "task.rpt_step": 10.0,
        "task.rpt_bin_width": 20.0,
        "train.epochs": 50,
        "train.batch_size": 64,
        "train.warmup_epochs": 10,
        "train.log_every": 10,
        "eval.trials_per_gap": 100,
    },
    # Full configuration = library defaults (kept explicit for discoverability).
    "full": {
        "model.n_hidden": 200,
        "task.t_pre": 100.0,
        "task.t_post": 500.0,
        "task.gap_max": 350.0,
        "task.rpt_max": 350.0,
        "task.rpt_step": 10.0,
        "task.rpt_bin_width": 20.0,
        "train.epochs": 1000,
        "train.batch_size": 256,
        "train.warmup_epochs": 10,
        "train.log_every": 10,
        "eval.trials_per_gap": 200,
    },
}


# Explicit CLI flags -> dotted config keys.
FLAG_TO_KEY = {
    "n_hidden": "model.n_hidden",
    "rank": "model.n_rank",
    "phi": "model.phi",
    "init_rec_scale": "model.init_rec_scale",
    "lambda_reg": "model.lambda_reg",
    "epochs": "train.epochs",
    "batch": "train.batch_size",
    "lr": "train.lr",
    "grad_clip": "train.grad_clip",
    "warmup": "train.warmup_epochs",
    "seed": "train.seed",
    "log_every": "train.log_every",
    "plateau_patience": "train.plateau_patience",
    "plateau_factor": "train.plateau_factor",
    "threshold": "task.threshold",
    "sigma_noise": "task.sigma_noise",
    "a_exo": "task.a_exo",
    "tau_exo": "task.tau_exo",
    "commit_temp": "task.commit_temp",
    "option_temp": "task.option_temp",
    "tau": "task.tau",
    "t_pre": "task.t_pre",
    "t_post": "task.t_post",
    "gap_max": "task.gap_max",
    "rpt_min": "task.rpt_min",
    "rpt_max": "task.rpt_max",
    "rpt_step": "task.rpt_step",
    "rpt_bin_width": "task.rpt_bin_width",
    "trials_per_gap": "eval.trials_per_gap",
}


# --------------------------------------------------------------------------- #
# Config assembly with dotted-key overrides
# --------------------------------------------------------------------------- #
def _coerce(raw: str, reference: Any) -> Any:
    """Coerce a string ``raw`` to the type of ``reference``."""
    if isinstance(reference, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(reference, int):
        return int(raw)
    if isinstance(reference, float):
        return float(raw)
    if isinstance(reference, tuple):
        return tuple(float(x) for x in raw.split(","))
    return raw


def apply_override(configs: dict, dotted: str, raw: Any) -> None:
    """Apply ``prefix.field = raw`` to the corresponding config dataclass."""
    prefix, _, field = dotted.partition(".")
    if prefix not in configs or not field:
        raise KeyError(f"Unknown config key: {dotted!r}")
    obj = configs[prefix]
    if not hasattr(obj, field):
        raise KeyError(f"{prefix!r} has no field {field!r}")
    current = getattr(obj, field)
    value = _coerce(str(raw), current) if not isinstance(raw, type(current)) else raw
    setattr(obj, field, value)


def base_configs() -> dict:
    """Return a fresh dict of default configs keyed by prefix."""
    return {
        "task": TaskParams(),
        "model": ModelParams(),
        "train": TrainConfig(),
        "eval": EvalConfig(),
    }


def build_configs(args: argparse.Namespace) -> dict:
    """Assemble configs from defaults, preset, explicit flags, and --set."""
    configs = base_configs()

    # 1. preset
    if args.preset:
        for key, val in PRESETS[args.preset].items():
            apply_override(configs, key, val)

    # 2. explicit flags (only those provided)
    for flag, key in FLAG_TO_KEY.items():
        val = getattr(args, flag)
        if val is not None:
            apply_override(configs, key, val)

    # m_values applies to both training and evaluation.
    if args.m_values is not None:
        mv = tuple(float(x) for x in args.m_values.split(","))
        configs["train"].m_choices = mv
        configs["eval"].m_values = mv

    # 3. generic --set overrides
    for item in args.set or []:
        key, _, val = item.partition("=")
        apply_override(configs, key.strip(), val.strip())

    return configs


# --------------------------------------------------------------------------- #
# Evaluation metrics
# --------------------------------------------------------------------------- #
def _observed_vortex_depth(grid: np.ndarray, tc: np.ndarray) -> float:
    region = tc[grid <= VORTEX_MAX_RPT]
    if region.size == 0 or np.all(np.isnan(region)):
        return float("nan")
    return float(0.5 - np.nanmin(region))


def compute_metrics(model, task: TaskParams, eval_cfg: EvalConfig) -> dict:
    """Evaluate a trained model against the behavioral targets.

    Returns a flat dict of metrics plus a per-m ``curves`` entry (used for
    plotting) and a scalar ``score`` (lower is better).
    """
    metrics: dict[str, Any] = {}
    curves: dict[float, dict] = {}
    score = 0.0

    for m in eval_cfg.m_values:
        res = model_tachometric(model, task, m, eval_cfg.trials_per_gap)
        stats, tc, grid = res["stats"], res["tc"], res["grid"]
        sweep = res["sweep"]
        frac_crossed = float(sweep["crossed"].float().mean())
        target = {k: float(v) for k, v in target_summary_stats(m, task).items()}

        model_vals = {
            "A": stats["A"],
            "t_rise": stats["t_rise75"],  # 75% crossing (see tachometric_targets)
            "t_vortex": stats["t_vortex"],
            "D": stats["D"],
        }

        tag = f"m{m:g}"
        beh_err = 0.0
        for key, w in STAT_WEIGHTS.items():
            mv, tv = model_vals[key], target[key]
            if mv is None or (isinstance(mv, float) and math.isnan(mv)):
                # Untrained / degenerate curve: treat as a large fixed miss.
                err = w * (150.0 ** 2 if key.startswith("t_") else 0.25)
            else:
                err = w * (mv - tv) ** 2
            beh_err += err
            metrics[f"{key}_{tag}"] = mv
            metrics[f"{key}_target_{tag}"] = tv

        vortex_depth = _observed_vortex_depth(grid, tc)
        metrics[f"frac_crossed_{tag}"] = frac_crossed
        metrics[f"vortex_depth_{tag}"] = vortex_depth

        # Penalize configurations where the race rarely reaches threshold.
        crossing_penalty = 5.0 * max(0.0, 0.4 - frac_crossed)
        score += beh_err + crossing_penalty
        curves[m] = {"grid": grid, "tc": tc}

    metrics["score"] = score
    metrics["curves"] = curves
    return metrics


# --------------------------------------------------------------------------- #
# Single run
# --------------------------------------------------------------------------- #
def run_single(configs: dict, out_dir: str, make_plots: bool = True) -> dict:
    """Train once and evaluate; write artifacts to ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    task, model_p = configs["task"], configs["model"]
    train_cfg, eval_cfg = configs["train"], configs["eval"]
    train_cfg.checkpoint_path = os.path.join(out_dir, "model.pt")

    model, history = train(train_cfg, model_p, task)
    metrics = compute_metrics(model, task, eval_cfg)
    metrics["final_train_loss"] = float(history[-1]["loss"]) if history else float("nan")

    curves = metrics.pop("curves")
    if make_plots and len(curves) >= 1:
        _plot_curves(curves, eval_cfg, out_dir)

    # Persist config + metrics (drop non-serializable curve arrays already popped).
    serializable = {k: v for k, v in metrics.items() if _json_safe(v)}
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(serializable, fh, indent=2)
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(_config_dump(configs), fh, indent=2)

    return metrics


def _plot_curves(curves: dict, eval_cfg: EvalConfig, out_dir: str) -> None:
    import matplotlib.pyplot as plt

    ms = list(curves.keys())
    young_m = 0.0 if 0.0 in curves else ms[0]
    adult_m = 1.0 if 1.0 in curves else ms[-1]
    grid = curves[young_m]["grid"]
    ax = plot_tachometric(
        np.asarray(grid),
        np.asarray(curves[young_m]["tc"]),
        np.asarray(curves[adult_m]["tc"]),
    )
    ax.figure.savefig(os.path.join(out_dir, "tachometric.png"), dpi=150, bbox_inches="tight")
    plt.close(ax.figure)


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
def expand_sweep(sweep_items: list[str]) -> list[dict]:
    """Expand ``key=v1,v2`` items into a list of override dicts (grid product)."""
    axes: dict[str, list[str]] = {}
    for item in sweep_items:
        key, _, vals = item.partition("=")
        axes[key.strip()] = [v.strip() for v in vals.split(",") if v.strip()]
    if not axes:
        return [{}]
    keys = list(axes.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*axes.values())]


def run_sweep(args: argparse.Namespace, out_root: str) -> None:
    """Run the grid of sweep combinations and rank them by score."""
    combos = expand_sweep(args.sweep)
    logger.info("Sweep over %d configuration(s).", len(combos))

    rows: list[dict] = []
    for i, combo in enumerate(combos):
        configs = build_configs(args)
        for key, raw in combo.items():
            apply_override(configs, key, raw)
        run_dir = os.path.join(out_root, f"run_{i:03d}")
        with tee_run_output(os.path.join(run_dir, "log.txt")):
            logger.info("[%d/%d] %s", i + 1, len(combos), combo or "(base)")

            metrics = run_single(configs, run_dir, make_plots=args.plots)
            row = {"run_id": i, **combo}
            row.update({k: v for k, v in metrics.items() if k != "curves" and _json_safe(v)})
            row["out_dir"] = run_dir
            rows.append(row)

    _write_sweep_results(rows, combos, out_root, top=args.top)


def _write_sweep_results(rows: list[dict], combos: list[dict], out_root: str, top: int) -> None:
    if not rows:
        return
    rows_sorted = sorted(rows, key=lambda r: r.get("score", float("inf")))

    columns: list[str] = []
    for row in rows_sorted:
        for key in row:
            if key not in columns:
                columns.append(key)

    csv_path = os.path.join(out_root, "results.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows_sorted)
    logger.info("Wrote %s", csv_path)

    best = rows_sorted[0]
    with open(os.path.join(out_root, "best_config.json"), "w") as fh:
        json.dump(best, fh, indent=2)

    swept_keys = list(combos[0].keys()) if combos and combos[0] else []
    logger.info("Top %d configurations by score:", min(top, len(rows_sorted)))
    for row in rows_sorted[:top]:
        swept = {k: row[k] for k in swept_keys}
        logger.info(
            "  score=%.4f  final_loss=%s  %s",
            row.get("score", float("nan")),
            _fmt(row.get("final_train_loss")),
            swept,
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _json_safe(value: Any) -> bool:
    return isinstance(value, (int, float, str, bool, type(None)))


def _fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _config_dump(configs: dict) -> dict:
    dump = {}
    for prefix, obj in configs.items():
        dump[prefix] = {f.name: getattr(obj, f.name) for f in fields(obj)}
    return dump


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="opt_behavior_fit",
        description="Parameterized smoke tests and hyperparameter sweeps for behavior fitting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--preset", choices=sorted(PRESETS), default="smoke",
                   help="Base configuration preset.")
    p.add_argument("--out", default=None, help="Output root directory.")
    p.add_argument("--top", type=int, default=5, help="How many best runs to print in a sweep.")
    p.add_argument("--no-plots", dest="plots", action="store_false",
                   help="Skip tachometric-curve figures (faster sweeps).")
    p.set_defaults(plots=True)

    # Explicit numeric/string flags (None => not overridden).
    p.add_argument("--n-hidden", dest="n_hidden", type=int)
    p.add_argument("--rank", type=int)
    p.add_argument("--phi", choices=["tanh", "relu"])
    p.add_argument("--init-rec-scale", dest="init_rec_scale", type=float)
    p.add_argument("--lambda-reg", dest="lambda_reg", type=float)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--grad-clip", dest="grad_clip", type=float)
    p.add_argument("--warmup", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--log-every", dest="log_every", type=int)
    p.add_argument("--plateau-patience", dest="plateau_patience", type=int)
    p.add_argument("--plateau-factor", dest="plateau_factor", type=float)
    p.add_argument("--threshold", type=float)
    p.add_argument("--sigma-noise", dest="sigma_noise", type=float)
    p.add_argument("--a-exo", dest="a_exo", type=float)
    p.add_argument("--tau-exo", dest="tau_exo", type=float)
    p.add_argument("--commit-temp", dest="commit_temp", type=float)
    p.add_argument("--option-temp", dest="option_temp", type=float)
    p.add_argument("--tau", type=float)
    p.add_argument("--t-pre", dest="t_pre", type=float)
    p.add_argument("--t-post", dest="t_post", type=float)
    p.add_argument("--gap-max", dest="gap_max", type=float)
    p.add_argument("--rpt-min", dest="rpt_min", type=float)
    p.add_argument("--rpt-max", dest="rpt_max", type=float)
    p.add_argument("--rpt-step", dest="rpt_step", type=float)
    p.add_argument("--rpt-bin-width", dest="rpt_bin_width", type=float)
    p.add_argument("--trials-per-gap", dest="trials_per_gap", type=int)
    p.add_argument("--m-values", dest="m_values", default=None,
                   help="Comma-separated maturation states, e.g. '0,1'.")

    p.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                   help="Generic dotted-key override (repeatable), e.g. task.threshold=0.4.")
    p.add_argument("--sweep", action="append", default=[], metavar="KEY=V1,V2",
                   help="Sweep a dotted key over values (repeatable); triggers grid search.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.sweep:
        out_root = args.out or os.path.join("results", "opt", f"sweep_{stamp}")
        os.makedirs(out_root, exist_ok=True)
        with tee_run_output(os.path.join(out_root, "log.txt")):
            logger.info("Preset=%s  output=%s", args.preset, out_root)
            run_sweep(args, out_root)
    else:
        out_dir = args.out or os.path.join("results", "opt", f"single_{stamp}")
        with tee_run_output(os.path.join(out_dir, "log.txt")):
            configs = build_configs(args)
            logger.info("Preset=%s  output=%s", args.preset, out_dir)
            logger.info("argv=%s", sys.argv)
            logger.info("raw --set items=%s", args.set)
            logger.info(
                "Resolved train plateau settings: patience=%s factor=%s threshold=%s threshold_mode=%s mode=%s cooldown=%s min_lr=%s eps=%s",
                configs["train"].plateau_patience,
                configs["train"].plateau_factor,
                1e-4,
                "rel",
                "min",
                0,
                0.0,
                1e-8,
            )
            metrics = run_single(configs, out_dir, make_plots=args.plots)
            logger.info("score=%.4f  final_loss=%s", metrics["score"], _fmt(metrics.get("final_train_loss")))
            for m in configs["eval"].m_values:
                tag = f"m{m:g}"
                logger.info(
                    "  [%s] A=%s (t=%s) t_rise=%s (t=%s) t_vortex=%s D=%s frac_crossed=%.2f vortex_depth=%s",
                    tag, _fmt(metrics.get(f"A_{tag}")), _fmt(metrics.get(f"A_target_{tag}")),
                    _fmt(metrics.get(f"t_rise_{tag}")), _fmt(metrics.get(f"t_rise_target_{tag}")),
                    _fmt(metrics.get(f"t_vortex_{tag}")), _fmt(metrics.get(f"D_{tag}")),
                    metrics.get(f"frac_crossed_{tag}", float("nan")),
                    _fmt(metrics.get(f"vortex_depth_{tag}")),
                )
            logger.info("Artifacts in %s", out_dir)


if __name__ == "__main__":
    main()
