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

from ..model.lrrnn import LRRNN
from ..model.model_params import DEFAULT_MODEL, ModelParams
from ..task.task_params import DEFAULT_TASK, TaskParams
from ..task.tachometric_targets import target_summary_stats
from ..task.trial_generator import build_inputs, sample_initial_state
from .curriculum import sample_curriculum_gaps
from .losses import behavioral_loss


@dataclass
class TrainConfig:
    """Optimization and logging configuration."""

    epochs: int = 1000
    batch_size: int = 256
    lr: float = 1e-3
    grad_clip: float = 1.0
    warmup_epochs: int = 100
    resume_checkpoint: str | None = None
    m_choices: tuple[float, ...] = (0.0, 1.0)
    log_every: int = 50
    plateau_patience: int = 50
    plateau_factor: float = 0.5
    seed: int = 0
    checkpoint_path: str = "checkpoints/behavior_fit.pt"
    device: str = "cpu"  # CPU only


def make_batch(
    batch_size: int,
    epoch: int,
    task: TaskParams,
    cfg: TrainConfig,
    generator: torch.Generator,
    n_hidden: int,
) -> dict:
    """Assemble a training batch using the curriculum gap schedule."""
    gaps = sample_curriculum_gaps(batch_size, epoch, task, cfg.warmup_epochs, generator)
    cue_sides = torch.randint(0, 2, (batch_size,), generator=generator)
    m_pool = torch.tensor(cfg.m_choices)
    m_values = m_pool[torch.randint(0, len(m_pool), (batch_size,), generator=generator)]
    u, t_cue = build_inputs(gaps, cue_sides, m_values, task)
    h0 = sample_initial_state(batch_size, n_hidden, task, generator=generator)
    return {"u": u, "gaps": gaps, "cue_sides": cue_sides, "m": m_values, "t_cue": t_cue, "h0": h0}


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
    for epoch in range(cfg.epochs):
        model.train()
        batch = make_batch(cfg.batch_size, epoch, task, cfg, generator, model_params.n_hidden)

        optimizer.zero_grad()
        loss, info = behavioral_loss(model, batch, task, targets, grid)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step(info["total"])

        record = {"epoch": epoch, "loss": float(info["total"]),
                  "behavior": float(info["behavior"]), "reg": float(info["reg"]),
                  "frac_crossed": float(info["frac_crossed"])}
        history.append(record)

        if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
            print(
                f"epoch {epoch:4d} | loss {record['loss']:.5f} "
                f"| beh {record['behavior']:.5f} | reg {record['reg']:.5f} "
                f"| crossed {record['frac_crossed']:.2f}"
            )

    _save_checkpoint(model, cfg, model_params, task, history)
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
