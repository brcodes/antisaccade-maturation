"""Find optimization runs whose saved configs match a command-derived spec.

This utility parses a raw optimization command, reconstructs the expected
initialization configs using the same argument semantics as
``antisaccade_model.experiments.opt_behavior_fit``, and then searches
``results/opt`` for matching experiment roots.

Exact matches are listed first. Partial matches come second and include the
fraction of expected configs that matched, assuming sweep run order follows the
ordered ``--sweep`` axes used at initialization.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from ..model.model_params import ModelParams
from ..task.task_params import TaskParams


@dataclass
class TrainConfig:
    epochs: int = 1000
    batch_size: int = 256
    lr: float = 1e-3
    grad_clip: float = 1.0
    warmup_epochs: int = 100
    resume_checkpoint: str | None = None
    m_choices: tuple[float, ...] = (0.0, 1.0)
    log_every: int = 50
    hard_eval_trials_per_gap: int = 10
    plateau_patience: int = 50
    plateau_factor: float = 0.5
    seed: int = 0
    checkpoint_path: str = "checkpoints/behavior_fit.pt"
    device: str = "cpu"


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "opt"
IGNORED_FIELDS = {
    "train": {"checkpoint_path", "device"},
}


PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "model.n_hidden": 64,
        "task.t_pre": 50.0,
        "task.t_post": 250.0,
        "task.gap_max": 180.0,
        "task.rpt_max": 240.0,
        "task.rpt_step": 30.0,
        "train.epochs": 50,
        "train.batch_size": 64,
        "train.warmup_epochs": 10,
        "train.log_every": 10,
    },
    "full": {
        "model.n_hidden": 200,
        "task.t_pre": 100.0,
        "task.t_post": 500.0,
        "task.gap_max": 350.0,
        "task.rpt_max": 300.0,
        "task.rpt_step": 10.0,
        "train.epochs": 1000,
        "train.batch_size": 256,
        "train.warmup_epochs": 100,
        "train.log_every": 50,
    },
}

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
}


def _strip_python_wrapper(tokens: list[str]) -> list[str]:
    """Remove an outer ``python ...`` wrapper if the command includes one."""
    if not tokens:
        return tokens

    if tokens[0].endswith("python") or tokens[0].endswith("python3"):
        tokens = tokens[1:]

    if tokens[:2] == ["-m", "antisaccade_model.experiments.opt_behavior_fit"]:
        return tokens[2:]

    if tokens and tokens[0].endswith("opt_behavior_fit.py"):
        return tokens[1:]

    return tokens


def _coerce(raw: str, reference: Any) -> Any:
    if isinstance(reference, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(reference, int):
        return int(raw)
    if isinstance(reference, float):
        return float(raw)
    if isinstance(reference, tuple):
        return tuple(float(x) for x in raw.split(","))
    return raw


def _apply_override(configs: dict[str, Any], dotted: str, raw: Any) -> None:
    prefix, _, field = dotted.partition(".")
    if prefix not in configs or not field:
        raise KeyError(f"Unknown config key: {dotted!r}")
    obj = configs[prefix]
    if not hasattr(obj, field):
        raise KeyError(f"{prefix!r} has no field {field!r}")
    current = getattr(obj, field)
    value = _coerce(str(raw), current) if not isinstance(raw, type(current)) else raw
    setattr(obj, field, value)


def _base_configs() -> dict[str, Any]:
    return {
        "task": TaskParams(),
        "model": ModelParams(),
        "train": TrainConfig(),
    }


def _build_configs(args: argparse.Namespace) -> dict[str, Any]:
    configs = _base_configs()

    if args.preset:
        for key, val in PRESETS[args.preset].items():
            _apply_override(configs, key, val)

    for flag, key in FLAG_TO_KEY.items():
        val = getattr(args, flag)
        if val is not None:
            _apply_override(configs, key, val)

    if args.m_values is not None:
        mv = tuple(float(x) for x in args.m_values.split(","))
        configs["train"].m_choices = mv

    for item in args.set or []:
        key, _, val = item.partition("=")
        _apply_override(configs, key.strip(), val.strip())

    return configs


def _expand_sweep(sweep_items: list[str]) -> list[dict[str, str]]:
    axes: dict[str, list[str]] = {}
    for item in sweep_items:
        key, _, vals = item.partition("=")
        axes[key.strip()] = [v.strip() for v in vals.split(",") if v.strip()]
    if not axes:
        return [{}]
    keys = list(axes.keys())
    combos: list[dict[str, str]] = []
    import itertools

    for combo in itertools.product(*axes.values()):
        combos.append(dict(zip(keys, combo)))
    return combos


def _parse_command(command: str) -> argparse.Namespace:
    tokens = _strip_python_wrapper(shlex.split(command))

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--out", default=None)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    parser.set_defaults(plots=True)

    parser.add_argument("--n-hidden", dest="n_hidden", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--phi", choices=["tanh", "relu"])
    parser.add_argument("--init-rec-scale", dest="init_rec_scale", type=float)
    parser.add_argument("--lambda-reg", dest="lambda_reg", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--grad-clip", dest="grad_clip", type=float)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log-every", dest="log_every", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--sigma-noise", dest="sigma_noise", type=float)
    parser.add_argument("--a-exo", dest="a_exo", type=float)
    parser.add_argument("--tau-exo", dest="tau_exo", type=float)
    parser.add_argument("--commit-temp", dest="commit_temp", type=float)
    parser.add_argument("--option-temp", dest="option_temp", type=float)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--t-pre", dest="t_pre", type=float)
    parser.add_argument("--t-post", dest="t_post", type=float)
    parser.add_argument("--gap-max", dest="gap_max", type=float)
    parser.add_argument("--rpt-min", dest="rpt_min", type=float)
    parser.add_argument("--rpt-max", dest="rpt_max", type=float)
    parser.add_argument("--rpt-step", dest="rpt_step", type=float)
    parser.add_argument("--rpt-bin-width", dest="rpt_bin_width", type=float)
    parser.add_argument("--trials-per-gap", dest="trials_per_gap", type=int)
    parser.add_argument("--m-values", dest="m_values", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--sweep", action="append", default=[])

    return parser.parse_args(tokens)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _config_projection(configs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Project a config bundle into a JSON-friendly nested mapping."""
    projected: dict[str, dict[str, Any]] = {}
    for prefix, obj in configs.items():
        projected[prefix] = {field.name: getattr(obj, field.name) for field in fields(obj)}
    return projected


def _normalize(value: Any) -> Any:
    """Normalize sequences so list/tuple representations compare consistently."""
    if isinstance(value, dict):
        return {key: _normalize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def _matches_subset(actual: Any, expected: Any, path: tuple[str, ...] = ()) -> bool:
    """Return True when ``actual`` contains at least the expected config values."""
    actual = _normalize(actual)
    expected = _normalize(expected)

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        for key, actual_value in actual.items():
            if path and path[0] in IGNORED_FIELDS and key in IGNORED_FIELDS[path[0]]:
                continue
            if key in expected and not _matches_subset(actual_value, expected[key], path=path + (key,)):
                return False
        return True

    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False
        return all(
            _matches_subset(actual_item, expected_item, path=path)
            for actual_item, expected_item in zip(actual, expected)
        )

    return actual == expected


def _experiment_root(config_path: Path) -> Path:
    run_dir = config_path.parent
    return run_dir.parent if run_dir.name.startswith("run_") else run_dir


def _run_index(path: Path) -> int:
    name = path.name
    if not name.startswith("run_"):
        return sys.maxsize
    try:
        return int(name.split("_", 1)[1])
    except ValueError:
        return sys.maxsize


def _group_config_files(results_root: Path) -> dict[Path, list[Path]]:
    grouped: dict[Path, list[Path]] = {}
    for config_path in results_root.rglob("config.json"):
        root = _experiment_root(config_path)
        grouped.setdefault(root, []).append(config_path)
    return grouped


def _expected_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs = _build_configs(args)
    if not args.sweep:
        return [_config_projection(configs)]

    expected_runs: list[dict[str, Any]] = []
    for combo in _expand_sweep(args.sweep):
        combo_configs = _build_configs(args)
        for key, raw in combo.items():
            _apply_override(combo_configs, key, raw)
        expected_runs.append(_config_projection(combo_configs))
    return expected_runs


def _compare_sweep_root(root: Path, expected_runs: list[dict[str, Any]]) -> tuple[int, int]:
    run_dirs = sorted((child for child in root.iterdir() if child.is_dir() and child.name.startswith("run_")), key=_run_index)
    actual_runs: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        config_path = run_dir / "config.json"
        if config_path.exists():
            actual_runs.append(_load_json(config_path))

    matched = 0
    for index, expected in enumerate(expected_runs):
        if index < len(actual_runs) and _matches_subset(actual_runs[index], expected):
            matched += 1
    return matched, len(expected_runs)


def _compare_single_root(root: Path, expected_run: dict[str, Any]) -> tuple[int, int]:
    config_path = root / "config.json"
    if not config_path.exists():
        return 0, 1
    actual = _load_json(config_path)
    return (1, 1) if _matches_subset(actual, expected_run) else (0, 1)


def _format_fraction(matched: int, expected: int) -> str:
    return f"{matched}/{expected}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="find_opt_runs_with_orig_command",
        description="Find opt runs whose saved configs match a raw initialization command.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--command",
        required=True,
        help="Raw optimization command to match, including python -m ... and its args.",
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Root directory containing results/opt runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    results_root = Path(args.results_root)
    expected_runs = _expected_runs(_parse_command(args.command))

    grouped = _group_config_files(results_root)
    exact: list[tuple[Path, int, int]] = []
    partial: list[tuple[Path, int, int]] = []

    if len(expected_runs) == 1:
        expected = expected_runs[0]
        for root in sorted(grouped):
            if root.name.startswith("single_"):
                matched, total = _compare_single_root(root, expected)
                if matched == total:
                    exact.append((root, matched, total))
                elif matched > 0:
                    partial.append((root, matched, total))
    else:
        for root in sorted(grouped):
            if not root.name.startswith("sweep_"):
                continue
            matched, total = _compare_sweep_root(root, expected_runs)
            if matched == total:
                run_count = sum(1 for child in root.iterdir() if child.is_dir() and child.name.startswith("run_"))
                if run_count == total:
                    exact.append((root, matched, total))
                else:
                    partial.append((root, matched, total))
            elif matched > 0:
                partial.append((root, matched, total))

    print("Exact matches:")
    if exact:
        for root, matched, total in exact:
            print(f"  {root} ({_format_fraction(matched, total)})")
    else:
        print("  None")

    print("Partial matches:")
    if partial:
        partial_sorted = sorted(partial, key=lambda item: (-item[1] / item[2] if item[2] else 0.0, str(item[0])))
        for root, matched, total in partial_sorted:
            print(f"  {root} ({_format_fraction(matched, total)} configs present)")
    else:
        print("  None")


if __name__ == "__main__":
    main()