"""Under-the-hood utilities for antisaccade checkpoints and opt-run matching.

This module combines checkpoint inspection with a path-finding diagnosis that
locates archived optimization outputs from a raw opt command, a folder name, or
a timestamp fragment.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "opt"

try:
    import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - depends on local environment
    torch = None  # type: ignore[assignment]


def _require_torch() -> Any:
    if torch is None:  # pragma: no cover - depends on local environment
        raise SystemExit(
            "This diagnostic requires PyTorch. Install the project dependencies first."
        )
    return torch

try:
    from ..task.task_params import (
        CUE_LEFT_IDX,
        CUE_RIGHT_IDX,
        GO_IDX,
        MATURATION_IDX,
        RULE_IDX,
        TaskParams,
    )
    from ..model.model_params import ModelParams
except ImportError:  # pragma: no cover - direct script execution path
    from antisaccade_model.task.task_params import (
        CUE_LEFT_IDX,
        CUE_RIGHT_IDX,
        GO_IDX,
        MATURATION_IDX,
        RULE_IDX,
        TaskParams,
    )
    from antisaccade_model.model.model_params import ModelParams


@dataclass
class TrainConfig:
    epochs: int = 1000
    batch_size: int = 200
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


INPUT_CHANNEL_LABELS = {
    GO_IDX: "go_signal",
    CUE_LEFT_IDX: "cue_left",
    CUE_RIGHT_IDX: "cue_right",
    RULE_IDX: "antisaccade_rule",
    MATURATION_IDX: "maturation_state",
}

DEFAULT_CHECKPOINT = REPO_ROOT / "results/opt/sweep_20260729_133029/run_001/model.pt"
DEFAULT_INIT_REC_SCALE_PATHS = (
    REPO_ROOT / "results/opt/sweep_20260729_165140/run_000/model.pt",
    REPO_ROOT / "results/opt/sweep_20260729_165140/run_001/model.pt",
    REPO_ROOT / "results/opt/sweep_20260729_133029/run_001/model.pt",
)

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
        "train.batch_size": 60,
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
        "train.batch_size": 250,
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
}


def _load_checkpoint(checkpoint_path: Path) -> Any:
    torch_mod = _require_torch()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        return torch_mod.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch_mod.load(checkpoint_path, map_location="cpu")


def _extract_w_in(checkpoint: Any) -> Any:
    torch_mod = _require_torch()
    if isinstance(checkpoint, torch_mod.Tensor):
        tensor = checkpoint
    elif isinstance(checkpoint, dict):
        if "W_in" in checkpoint:
            tensor = checkpoint["W_in"]
        elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
            matches = [key for key in state_dict if key == "W_in" or key.endswith(".W_in")]
            if len(matches) == 1:
                tensor = state_dict[matches[0]]
            elif "W_in" in state_dict:
                tensor = state_dict["W_in"]
            else:
                raise KeyError(
                    "Could not find W_in in checkpoint['state_dict']; expected a key named "
                    "'W_in' or one ending in '.W_in'."
                )
        else:
            matches = [key for key in checkpoint if key == "W_in" or key.endswith(".W_in")]
            if len(matches) == 1:
                tensor = checkpoint[matches[0]]
            else:
                raise KeyError("Could not find W_in in checkpoint.")
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    if not torch_mod.is_tensor(tensor):
        tensor = torch_mod.as_tensor(tensor)
    if tensor.ndim != 2:
        raise ValueError(f"Expected W_in to be a matrix, got shape {tuple(tensor.shape)}")
    return tensor


def check_W_in_norms(checkpoint_path: str | Path = DEFAULT_CHECKPOINT) -> Any:
    _require_torch()
    checkpoint = _load_checkpoint(Path(checkpoint_path))
    w_in = _extract_w_in(checkpoint)
    norms = w_in.norm(dim=0)
    print("W_in column norms by input channel:")
    for index, value in enumerate(norms.tolist()):
        label = INPUT_CHANNEL_LABELS.get(index, f"input_{index}")
        print(f"  [{index}] {label}: {value:.6f}")
    return norms


def _print_tensor_summary(name: str, value: Any, *, indent: str = "  ") -> None:
    torch_mod = _require_torch()
    if not torch_mod.is_tensor(value):
        return
    shape = tuple(value.shape)
    try:
        norm = float(value.norm().item())
        norm_text = f"{norm:.6f}"
    except Exception:
        norm_text = "n/a"
    print(f"{indent}{name}: shape={shape}, norm={norm_text}")


def _print_checkpoint_tensor_summaries(checkpoint: Any, *, indent: str = "  ") -> None:
    torch_mod = _require_torch()
    if not isinstance(checkpoint, dict):
        _print_tensor_summary("checkpoint", checkpoint, indent=indent)
        return

    for key, value in checkpoint.items():
        if torch_mod.is_tensor(value):
            _print_tensor_summary(key, value, indent=indent)
        elif isinstance(value, dict) and key == "state_dict":
            print(f"{indent}{key}:")
            for sub_key, sub_value in value.items():
                _print_tensor_summary(sub_key, sub_value, indent=indent + "  ")


def _extract_init_rec_scale(checkpoint: Any) -> float | None:
    if not isinstance(checkpoint, dict):
        return None

    model_params = checkpoint.get("model_params")
    if model_params is None:
        return None

    value = getattr(model_params, "init_rec_scale", None)
    return float(value) if value is not None else None


def check_init_rec_scale(paths: Sequence[str | Path] = DEFAULT_INIT_REC_SCALE_PATHS) -> None:
    _require_torch()
    for path in paths:
        checkpoint_path = Path(path)
        checkpoint = _load_checkpoint(checkpoint_path)
        print(f"\n{checkpoint_path}")
        init_rec_scale = _extract_init_rec_scale(checkpoint)
        if init_rec_scale is not None:
            print(f"model.init_rec_scale: {init_rec_scale:.6f}")
        else:
            print("model.init_rec_scale: unavailable")
        if isinstance(checkpoint, dict):
            print("Keys:", list(checkpoint.keys()))
        else:
            print("Keys:", [type(checkpoint).__name__])
        _print_checkpoint_tensor_summaries(checkpoint)


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


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def _matches_subset(actual: Any, expected: Any, path: tuple[str, ...] = ()) -> bool:
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


def _config_projection(configs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    projected: dict[str, dict[str, Any]] = {}
    for prefix, obj in configs.items():
        projected[prefix] = {field.name: getattr(obj, field.name) for field in fields(obj)}
    return projected


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


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


def _strip_python_wrapper(tokens: list[str]) -> list[str]:
    if not tokens:
        return tokens

    if tokens[0].endswith("python") or tokens[0].endswith("python3"):
        tokens = tokens[1:]

    if tokens[:2] == ["-m", "antisaccade_model.experiments.opt_behavior_fit"]:
        return tokens[2:]

    if tokens and tokens[0].endswith("opt_behavior_fit.py"):
        return tokens[1:]

    return tokens


def _parse_command(command: str) -> argparse.Namespace:
    command = re.sub(r"\\\r?\n", " ", command)
    command = command.replace("\n", " ")
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
    parser.add_argument("--plateau-patience", dest="plateau_patience", type=int)
    parser.add_argument("--plateau-factor", dest="plateau_factor", type=float)
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


def _is_command_like(spec: str) -> bool:
    normalized = re.sub(r"\\\r?\n", " ", spec).replace("\n", " ").strip()
    if not normalized:
        return False
    tokens = shlex.split(normalized)
    if len(tokens) > 1:
        return True
    return any(marker in normalized for marker in ("python ", "python-m", " --", "\t--", " -m "))


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
    actual_index = 0
    for expected in expected_runs:
        while actual_index < len(actual_runs) and not _matches_subset(actual_runs[actual_index], expected):
            actual_index += 1
        if actual_index >= len(actual_runs):
            break
        matched += 1
        actual_index += 1

    return matched, len(actual_runs)


def _compare_single_root(root: Path, expected_run: dict[str, Any]) -> tuple[int, int]:
    config_path = root / "config.json"
    if not config_path.exists():
        return 0, 1
    actual = _load_json(config_path)
    return (1, 1) if _matches_subset(actual, expected_run) else (0, 1)


def _format_fraction(matched: int, expected: int) -> str:
    return f"{matched}/{expected}"


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _files_under(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _print_files_under(root: Path, *, indent: str = "  ") -> None:
    print(f"{indent}{_relative_path(root)}")
    for path in _files_under(root):
        print(f"{indent}  {_relative_path(path)}")


def _search_repo_paths(query: str) -> list[Path]:
    matches: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if query in str(path.relative_to(REPO_ROOT)):
            matches.append(path)
    return sorted(matches)


def _print_repo_path_matches(paths: list[Path]) -> None:
    if not paths:
        print("No matches")
        return

    grouped: dict[Path, list[Path]] = {}
    for path in paths:
        key = path.parent
        grouped.setdefault(key, []).append(path)

    for parent in sorted(grouped):
        print(_relative_path(parent))
        for path in sorted(grouped[parent]):
            print(f"  {_relative_path(path)}")


def _flatten_config_sections(config: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for section, values in config.items():
        if isinstance(values, dict):
            for key, value in values.items():
                if section in IGNORED_FIELDS and key in IGNORED_FIELDS[section]:
                    continue
                flat[f"{section}.{key}"] = value
    return flat


def _format_config_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _common_config_fields(configs: list[dict[str, Any]]) -> dict[str, Any]:
    if not configs:
        return {}
    common = _flatten_config_sections(configs[0])
    for config in configs[1:]:
        flat = _flatten_config_sections(config)
        for key in list(common):
            if key not in flat or flat[key] != common[key]:
                common.pop(key, None)
    return common


def _config_diff_text(config: dict[str, Any], common_fields: dict[str, Any]) -> str:
    flat = _flatten_config_sections(config)
    diffs = [key for key, value in flat.items() if common_fields.get(key) != value]
    if not diffs:
        return ""
    return ", ".join(f"{key}={_format_config_value(flat[key])}" for key in sorted(diffs))


def _config_share_lines(config: dict[str, Any]) -> list[str]:
    flat = _flatten_config_sections(config)
    return [f"{key}={_format_config_value(flat[key])}" for key in sorted(flat)]


def _run_dir_configs(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    run_dirs = sorted((child for child in root.iterdir() if child.is_dir() and child.name.startswith("run_")), key=_run_index)
    run_configs: list[tuple[Path, dict[str, Any]]] = []
    for run_dir in run_dirs:
        config_path = run_dir / "config.json"
        if config_path.exists():
            run_configs.append((run_dir, _load_json(config_path)))
    return run_configs


def _print_experiment_tree(root: Path, *, annotation: str = "") -> None:
    root_label = _relative_path(root)
    suffix = f" {annotation}" if annotation else ""
    print(f"  {root_label}{suffix}")

    root_files = sorted(path for path in root.iterdir() if path.is_file())
    for path in root_files:
        print(f"    {_relative_path(path)}")

    run_configs = _run_dir_configs(root)
    common_fields = _common_config_fields([config for _, config in run_configs]) if run_configs else {}
    for run_dir, config in run_configs:
        diff_text = _config_diff_text(config, common_fields)
        run_suffix = f" (config diff: {diff_text})" if diff_text else ""
        print(f"    {_relative_path(run_dir)}{run_suffix}")
        for child in sorted(run_dir.iterdir()):
            if child.is_file():
                print(f"      {_relative_path(child)}")

        print(f"      config share:")
        for line in _config_share_lines(config):
            print(f"        {line}")


def _print_root_forest(roots: list[Path], *, annotations: dict[Path, str] | None = None) -> None:
    if not roots:
        print("No matches")
        return

    print(_relative_path(DEFAULT_RESULTS_ROOT))
    for root in roots:
        _print_experiment_tree(root, annotation=(annotations or {}).get(root, ""))


def _search_experiment_roots(query: str) -> list[Path]:
    roots: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if path.is_dir() and path.name.startswith(("sweep_", "single_")) and query in path.name:
            roots.append(path)
    return sorted(set(roots))


def _is_command_like(spec: str) -> bool:
    normalized = re.sub(r"\\\r?\n", " ", spec).replace("\n", " ").strip()
    if not normalized:
        return False
    tokens = shlex.split(normalized)
    if len(tokens) > 1:
        return True
    return any(marker in normalized for marker in ("python ", " --", "\t--", " -m "))


def _run_get_checkpoint_path(spec: str) -> None:
    spec = spec.strip()
    if not spec:
        raise SystemExit("get_checkpoint_path requires a non-empty value")

    if _is_command_like(spec):
        expected_runs = _expected_runs(_parse_command(spec))
        grouped = _group_config_files(DEFAULT_RESULTS_ROOT)
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
                matched, actual_count = _compare_sweep_root(root, expected_runs)
                if matched == len(expected_runs) and actual_count == len(expected_runs):
                    exact.append((root, matched, actual_count))
                elif matched > 0:
                    partial.append((root, matched, actual_count))

        print("Exact matches:")
        if exact:
            print(_relative_path(DEFAULT_RESULTS_ROOT))
            for root, matched, total in exact:
                _print_experiment_tree(root, annotation=f"({_format_fraction(matched, total)})")
        else:
            print("  None")

        print("Partial matches:")
        if partial:
            partial_sorted = sorted(partial, key=lambda item: (-item[1] / item[2] if item[2] else 0.0, str(item[0])))
            print(_relative_path(DEFAULT_RESULTS_ROOT))
            for root, matched, total in partial_sorted:
                _print_experiment_tree(root, annotation=f"({_format_fraction(matched, total)} configs present)")
        else:
            print("  None")
        return

    roots = _search_experiment_roots(spec)
    annotations = {root: f"({len(_run_dir_configs(root))} configs present)" for root in roots}
    _print_root_forest(roots, annotations=annotations)


def _parse_diagnosis_value(value: str) -> tuple[str, str | None]:
    if value == "get_checkpoint_path":
        raise SystemExit("get_checkpoint_path requires '=' and a value")
    if value.startswith("get_checkpoint_path="):
        return "get_checkpoint_path", value.split("=", 1)[1]
    return value, None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="under_the_hood",
        description="Checkpoint inspection utilities and opt-run matching for antisaccade-model artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--diagnosis",
        required=True,
        help=(
            "Inspection routine to run. Use get_checkpoint_path=<spec> for run/path lookup."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Path to the checkpoint to inspect.",
    )
    return parser


DIAGNOSES_RUNNERS: dict[str, Callable[..., Any]] = {
    "check_W_in_norms": check_W_in_norms,
    "check_init_rec_scale": check_init_rec_scale,
}


def main(argv: list[str] | None = None) -> None:
    tokens = list(sys.argv[1:] if argv is None else argv)
    args = build_arg_parser().parse_args(tokens)
    diagnosis_name, diagnosis_value = _parse_diagnosis_value(args.diagnosis)

    if diagnosis_name == "get_checkpoint_path":
        _run_get_checkpoint_path(diagnosis_value or "")
        return

    diagnosis = DIAGNOSES_RUNNERS.get(diagnosis_name)
    if diagnosis is None:
        raise SystemExit(f"Unknown diagnosis: {diagnosis_name}")
    if diagnosis_name == "check_init_rec_scale":
        diagnosis()
    else:
        diagnosis(args.checkpoint)


if __name__ == "__main__":
    main()