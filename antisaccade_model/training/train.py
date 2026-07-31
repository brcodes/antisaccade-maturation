"""Main training loop: fit behavior only (direction 1).

Trains the low-rank RNN to reproduce the young (m=0) and adult (m=1) tachometric
summary statistics. No neural (SI) term enters the loss -- SI is a post-hoc
prediction (gameplan Sections 3.2, 5.3).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch

from ..analysis.tachometric_analysis import hard_summary_stat_mse
from ..model.lrrnn import LRRNN
from ..model.model_params import DEFAULT_MODEL, ModelParams
from ..task.task_params import DEFAULT_TASK, TaskParams
from ..task.tachometric_targets import target_summary_stats
from ..task.trial_generator import build_inputs, sample_initial_state
from .curriculum import sample_stratified_gaps
from .losses import behavioral_loss


@dataclass
class TrainConfig:
    """Optimization and logging configuration."""

    epochs: int = 1000
    batch_size: int = 200
    lr: float = 1e-3
    grad_clip: float = 1.0
    warmup_epochs: int = 10
    resume_checkpoint: str | None = None
    m_choices: tuple[float, ...] = (0.0, 1.0)
    log_every: int = 10
    graceful_exit: int | None = 1
    hard_eval_trials_per_gap: int = 100
    plateau_patience: int = 99999
    plateau_factor: float = 0.99999
    seed: int = 0
    checkpoint_path: str = "checkpoints/behavior_fit.pt"
    device: str = "cpu"  # CPU only


def _log_plateau_reduction(
    epoch: int,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    old_lrs: list[float],
    new_lrs: list[float],
    metric: float,
) -> None:
    """Print a detailed scheduler reduction event for terminal and log capture."""
    print(
        f"lr reduced at epoch {epoch:4d} | metric {metric:.6f} | "
        f"mode={scheduler.mode} | threshold={scheduler.threshold:g} | "
        f"threshold_mode={scheduler.threshold_mode} | patience={scheduler.patience} | "
        f"factor={scheduler.factor:g} | cooldown={scheduler.cooldown} | "
        f"best={scheduler.best:.6f} | num_bad_epochs={scheduler.num_bad_epochs} | "
        f"cooldown_counter={scheduler.cooldown_counter} | last_epoch={scheduler.last_epoch} | "
        f"min_lr={scheduler.min_lrs} | eps={scheduler.eps:g} | "
        f"old_lr={old_lrs} | new_lr={new_lrs}"
    )


def make_batch(
    batch_size: int,
    epoch: int = 0,
    task: TaskParams = DEFAULT_TASK,
    cfg: Optional[TrainConfig] = None,
    generator: Optional[torch.Generator] = None,
    n_hidden: int = DEFAULT_MODEL.n_hidden,
    m_values: Optional[list[float]] = None,
) -> dict:
    """Assemble a training batch with equal representation across gap strata."""
    cfg = cfg or TrainConfig()
    gaps = sample_stratified_gaps(batch_size, task, generator=generator)
    cue_sides = torch.randint(0, 2, (batch_size,), generator=generator)
    m_pool = torch.tensor(cfg.m_choices if m_values is None else m_values)
    sampled_m = m_pool[torch.randint(0, len(m_pool), (batch_size,), generator=generator)]
    u, t_cue = build_inputs(gaps, cue_sides, sampled_m, task)
    h0 = sample_initial_state(batch_size, n_hidden, task, generator=generator)
    return {"u": u, "gaps": gaps, "cue_sides": cue_sides, "m": sampled_m, "t_cue": t_cue, "h0": h0}


def train(
    cfg: Optional[TrainConfig] = None,
    model_params: ModelParams = DEFAULT_MODEL,
    task: TaskParams = DEFAULT_TASK,
) -> tuple[LRRNN, list[dict]]:
    """Run behavior-only training and return the trained model and history."""
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)

    model = LRRNN(model_params, task).to(cfg.device)
    if cfg.resume_checkpoint is not None:
        ckpt = torch.load(cfg.resume_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=cfg.plateau_factor, patience=cfg.plateau_patience
    )

    grid = torch.tensor(task.rpt_grid, dtype=torch.float32)
    targets = {m: target_summary_stats(m, task) for m in cfg.m_choices}

    history: list[dict] = []
    interrupted = False
    try:
        for epoch in range(cfg.epochs):
            model.train()
            batch = make_batch(cfg.batch_size, epoch, task, cfg, generator, model_params.n_hidden)

            optimizer.zero_grad()
            loss, info = behavioral_loss(model, batch, task, targets, grid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            old_lrs = [group["lr"] for group in optimizer.param_groups]
            scheduler.step(info["total"])
            new_lrs = [group["lr"] for group in optimizer.param_groups]
            if new_lrs != old_lrs:
                _log_plateau_reduction(epoch, scheduler, old_lrs, new_lrs, float(info["total"]))

            record = {"epoch": epoch, "loss": float(info["total"]),
                      "curve": float(info["curve"]), "reg": float(info["reg"]),
                      "frac_crossed": float(info["frac_crossed"])}

            if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
                model.eval()
                hard_mse, hard_stats = hard_summary_stat_mse(
                    model, task, targets, cfg.m_choices, cfg.hard_eval_trials_per_gap
                )
                model.train()
                record["hard_summary_mse"] = hard_mse
                record["hard_stats"] = hard_stats
                print(
                    f"epoch {epoch:4d} | loss {record['loss']:.5f} "
                    f"| curve {record['curve']:.5f} | hard {hard_mse:.5f} | reg {record['reg']:.5f} "
                    f"| crossed {record['frac_crossed']:.2f}"
                )
            history.append(record)
    except KeyboardInterrupt:
        if not cfg.graceful_exit:
            raise
        interrupted = True
        print("KeyboardInterrupt received; saving partial training state.")

    _save_checkpoint(model, cfg, model_params, task, history)
    if interrupted:
        print("Stopped early after graceful exit.")
    return model, history


def _save_checkpoint(model, cfg, model_params, task, history) -> None:
    path = cfg.checkpoint_path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_params": model_params,
            "task": task,
            "config": cfg,
            "history": history,
        },
        path,
    )
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path: str) -> tuple[LRRNN, dict]:
    """Load a trained model and its metadata from ``path``."""
    ckpt = torch.load(path, weights_only=False)
    model = LRRNN(ckpt["model_params"], ckpt["task"])
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model, ckpt
