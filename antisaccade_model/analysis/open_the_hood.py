"""Open-the-hood diagnostics for trained antisaccade checkpoints.

This module is intentionally small and CLI-driven: each diagnosis is gated by
``--diagnosis <name>`` so future inspection routines can be added without
changing the entrypoint shape.

Run from the repository root, for example::

    python antisaccade_model/analysis/open_the_hood.py --diagnosis check_W_in_norms
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    # Allow direct execution via ``python antisaccade_model/analysis/open_the_hood.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "This diagnostic requires PyTorch. Install the project dependencies first."
    ) from exc

try:
    from ..task.task_params import (
        CUE_LEFT_IDX,
        CUE_RIGHT_IDX,
        GO_IDX,
        MATURATION_IDX,
        RULE_IDX,
    )
except ImportError:  # pragma: no cover - direct script execution path
    from antisaccade_model.task.task_params import (
        CUE_LEFT_IDX,
        CUE_RIGHT_IDX,
        GO_IDX,
        MATURATION_IDX,
        RULE_IDX,
    )

DEFAULT_CHECKPOINT = REPO_ROOT / "results/opt/sweep_20260729_133029/run_001/model.pt"

INPUT_CHANNEL_LABELS = {
    GO_IDX: "go_signal",
    CUE_LEFT_IDX: "cue_left",
    CUE_RIGHT_IDX: "cue_right",
    RULE_IDX: "antisaccade_rule",
    MATURATION_IDX: "maturation_state",
}


def _load_checkpoint(checkpoint_path: Path) -> Any:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def _extract_w_in(checkpoint: Any) -> torch.Tensor:
    """Return the input-weight matrix from a checkpoint-like object."""
    if isinstance(checkpoint, torch.Tensor):
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

    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    if tensor.ndim != 2:
        raise ValueError(f"Expected W_in to be a matrix, got shape {tuple(tensor.shape)}")
    return tensor


def check_W_in_norms(checkpoint_path: str | Path = DEFAULT_CHECKPOINT) -> torch.Tensor:
    """Load a checkpoint and print the norm of each input column in ``W_in``."""
    checkpoint = _load_checkpoint(Path(checkpoint_path))
    w_in = _extract_w_in(checkpoint)
    norms = w_in.norm(dim=0)
    print("W_in column norms by input channel:")
    for index, value in enumerate(norms.tolist()):
        label = INPUT_CHANNEL_LABELS.get(index, f"input_{index}")
        print(f"  [{index}] {label}: {value:.6f}")
    return norms


DIAGNOSES: dict[str, Callable[[str | Path], torch.Tensor]] = {
    "check_W_in_norms": check_W_in_norms,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="open_the_hood",
        description="Checkpoint inspection utilities for antisaccade-model artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--diagnosis",
        required=True,
        choices=sorted(DIAGNOSES),
        help="Inspection routine to run.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Path to the checkpoint to inspect.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    diagnosis = DIAGNOSES[args.diagnosis]
    diagnosis(args.checkpoint)


if __name__ == "__main__":
    main()