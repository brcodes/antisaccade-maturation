# OPT_README.md — Behavior-Fit Optimization Harness

Architecture and operating manual for [`opt_behavior_fit.py`](antisaccade_model/experiments/opt_behavior_fit.py):
the fast-iteration driver for training and tuning the low-rank RNN against the
Zhu et al. (2024) tachometric targets.

This file is the **reference / "what we have and where to go"** document.
Actual run results and the live decision log live in
[opt_progress.md](opt_progress.md).

---

## 1. Purpose

`run_behavior_fit.py` is the full, slow, canonical experiment. `opt_behavior_fit.py`
is its lightweight sibling built for **iteration speed**:

- run a coarse **smoke** config in seconds-to-minutes to answer "is the
  architecture doing anything?" before committing to long runs;
- **sweep** hyperparameters over a grid and rank configurations by a single
  behavioral score;
- do it all from the CLI with layered config overrides, no code edits.

---

## 2. Quick start — the smoke startpoint

```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate tcia-lung1-seg-class-cpu
cd /Users/brycerogers/Documents/antisaccade_maturation
MPLBACKEND=Agg python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke
```

The `smoke` preset intentionally trims everything so a full train+evaluate cycle
is cheap. It is a **diagnostic**, not a fit: the vortex either starts to appear
within ~50 epochs or it doesn't, which tells you immediately whether the
architecture + exogenous burst are wired correctly.

| Knob | `smoke` | `full` | Why smoke shrinks it |
|---|---|---|---|
| `model.n_hidden` | 64 | 200 | fewer units → fast forward/backward |
| `task.t_pre` / `task.t_post` | 50 / 250 | 100 / 500 | shorter timeline → fewer Euler steps |
| `task.gap_max` | 180 | 350 | narrower rPT range to cover |
| `task.rpt_step` | 30 | 10 | coarse bins → cheaper curve, less noise |
| `train.epochs` | 50 | 1000 | just enough to see a trend |
| `train.batch_size` | 64 | 256 | smaller batches |
| `train.warmup_epochs` | 10 | 100 | curriculum ramps quickly |
| `eval.trials_per_gap` | 100 | 200 | faster Monte-Carlo curve |

---

## 3. Modes

### Single run (default)
Trains once, evaluates, writes artifacts, logs the fit vs. targets.

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke --threshold 0.5 --a-exo 5
```

### Sweep (`--sweep`)
Any `--sweep key=v1,v2` flag switches to grid mode. Multiple `--sweep` flags form
the **Cartesian product**. Each combination is a separate run; results are ranked
by score.

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --sweep train.lr=1e-3,3e-4 \
    --sweep task.threshold=0.3,0.5 \
    --sweep task.a_exo=3,5 \
    --top 5 --no-plots
```

---

## 4. Configuration model

Config is assembled in **layers**, later layers win:

```
defaults  →  --preset  →  explicit flags  →  --set k=v  →  --sweep k=v1,v2
```

- **Dotted keys** address any field of the four underlying dataclasses:
  - `task.*`  → `TaskParams`  (biology/task: `threshold`, `a_exo`, `tau_exo`, `sigma_noise`, `commit_temp`, `tau`, `t_post`, `rpt_*`, …)
  - `model.*` → `ModelParams` (`n_hidden`, `n_rank`, `phi`, `init_rec_scale`, `lambda_reg`)
  - `train.*` → `TrainConfig` (`lr`, `epochs`, `batch_size`, `grad_clip`, `warmup_epochs`, `seed`, …)
  - `eval.*`  → `EvalConfig`  (`trials_per_gap`, `m_values`)
- **Explicit flags** exist for the common knobs (e.g. `--lr`, `--n-hidden`, `--threshold`, `--a-exo`); see `--help`.
- **`--set task.foo=bar`** is the generic escape hatch for any field without a flag.
- **`--m-values 0,1`** sets both `train.m_choices` and `eval.m_values` together.

Types are coerced automatically from the target field's type (int/float/bool/tuple).

---

## 5. Expected artifacts

Everything is written under `results/opt/`.

**Single run** → `results/opt/single_<timestamp>/`
| File | Contents |
|---|---|
| `model.pt` | trained checkpoint (loadable via `training.train.load_checkpoint`) |
| `tachometric.png` | young vs. adult tachometric curves with targets (skipped with `--no-plots`) |
| `metrics.json` | flat scalar metrics + `score` + `final_train_loss` |
| `config.json` | the fully-resolved config used (reproducibility) |

**Sweep** → `results/opt/sweep_<timestamp>/`
| File | Contents |
|---|---|
| `run_000/`, `run_001/`, … | one single-run folder per grid point |
| `results.csv` | one row per run, **sorted by score ascending**, all metrics + swept keys |
| `best_config.json` | the winning row |

Console output logs the top-N configs by score at the end of a sweep.

---

## 6. Scoring — what "better" means

`score` (lower = better) is a weighted sum over each maturation state `m`:

- **behavioral MSE** of the four summary stats vs. the Zhu et al. targets, using
  the same weights as the training loss (`t_rise` and `t_vortex` in ms are
  down-weighted `1e-4`; `A` and `D` weighted `1.0`);
- a **crossing penalty** `5·max(0, 0.4 − frac_crossed)` that punishes configs
  where the decision race rarely reaches threshold θ (a degenerate, untrained
  regime where the curve is meaningless).

`score` is deliberately **not** the raw training loss — a low training loss with
`frac_crossed ≈ 0` is a false victory. Always read `frac_crossed_*` alongside score.

Key per-`m` metrics in `metrics.json` / `results.csv`:
`A_m*`, `t_rise_m*` (75% crossing), `t_vortex_m*`, `D_m*`,
`frac_crossed_m*`, `vortex_depth_m*`, each with a `*_target_*` companion.

---

## 7. Principled sweep order (generalizable framework)

The governing rule: **make the computation exist before you make it pretty, and
make it train stably before you make it biological.** Fix the thing that most
invalidates every downstream measurement first. Concretely, in phases:

### Phase 0 — Plumbing / "does anything cross?"
Nothing else matters until the decision race reaches threshold on a healthy
fraction of trials (`frac_crossed` well above the 0.4 penalty knee, ideally
0.5–0.9). This is a **dynamic-range** problem, not a learning problem.
- Primary knobs: `task.threshold`, `model.init_rec_scale`, readout gain, `task.a_exo`.
- Success gate: `frac_crossed_m* ≳ 0.5` at init / very early training.

### Phase 1 — Optimization stability ("can it learn at all?")
Now that gradients flow through a live race, get training to descend cleanly.
- Order: **learning rate first** (single most impactful, most interacting knob),
  then `grad_clip`, then `batch_size`, then `warmup_epochs`.
- Do these on the **smoke** preset. A good LR generalizes across sizes better
  than most people expect, so lock it before scaling.
- Success gate: monotone-ish loss decrease over 50 epochs, no NaNs/divergence.

### Phase 2 — Capacity ("does it have room to fit?")
Only after LR is sane, vary representational capacity.
- Order: `model.n_hidden` (unit size) → `model.lambda_reg` (regularization) →
  `model.phi` (nonlinearity) → `train.epochs`.
- Rank `model.n_rank` is **fixed at R=2** by the modeling decision; treat it as a
  structural constant, not a sweep axis, unless doing a deliberate ablation.
- Success gate: the asymptote `A` and the qualitative vortex shape appear.

### Phase 3 — Behavioral fit ("does it match the data?")
With a trainable, adequately-sized model, tune toward the target statistics.
- Knobs: `task.commit_temp` / `task.option_temp` (sharpness of the decision),
  `task.sigma_noise` (trial-to-trial spread → vortex depth `D`),
  `eval.trials_per_gap` and `task.rpt_step` (measurement resolution, raise for
  final scoring only — they cost time, not fit).
- Success gate: `score` dominated by genuine stat error, not the crossing penalty.

### Phase 4 — Biological / mechanistic tuning (later; see §9)
Once a config reliably fits, interpret and constrain it biologically.

**Sweep hygiene throughout:** sweep **one phase at a time**; keep grids small
(2–3 values per axis, ≤ ~8 combos); freeze the winner via `best_config.json`
before opening the next phase; log every accepted move in
[opt_progress.md](opt_progress.md).

---

## 8. Principled scale-up plan

Scale in a fixed sequence, re-checking the Phase-1 gate after each step so you
never debug a training failure and a scale change at the same time:

1. **Epochs first (cheapest signal):** `smoke` but `--epochs 150–300`. Confirms
   the trend seen at 50 epochs is real and not a transient.
2. **Measurement resolution:** raise `eval.trials_per_gap` (100 → 200) and drop
   `task.rpt_step` (30 → 10). Sharpens the curve without touching the model.
3. **Timeline to full length:** restore `task.t_pre/ t_post` and `task.gap_max`
   to `full` values so the real rPT range is represented.
4. **Capacity to target:** raise `model.n_hidden` toward 200. Re-confirm the LR
   from Phase 1 still trains stably (usually yes; nudge down if not).
5. **Batch + epochs to full:** `batch_size` 64 → 256, `epochs` → 1000.
6. **Promote:** once a `full`-scale config fits, hand the resolved `config.json`
   to `run_behavior_fit.py` for the canonical, figure-producing run.

Rule of thumb: **change one scale axis per step, re-run smoke-length validation,
then proceed.** If a step breaks training, the culprit is that step.

---

## 9. Biological / mechanistic tuning (the "later" layer)

These are meaningful only after a config trains and fits; they turn a curve-fitter
into a model that says something. Approach them as **constrained** sweeps —
narrow, theory-motivated ranges — not blind grids:

- **Exogenous capture (`task.a_exo`, `task.tau_exo`):** the reflexive
  cue-driven burst that produces the vortex dip. These set *why* early responses
  are captured toward the cue; tune against `vortex_depth` and `t_vortex`.
- **Decision threshold / commitment (`task.threshold`, `task.commit_temp`):**
  the accumulation-to-bound analogue; governs speed–accuracy tradeoff and `t_rise`.
- **Internal noise (`task.sigma_noise`):** maps to trial variability and the
  depth/sharpness of the vortex (`D`).
- **Maturation axis (`m` / `eval.m_values`):** young↔adult is the scientific
  contrast. Once both endpoints fit, verify the intermediate `m` interpolation
  is monotone and the mechanism (not just the numbers) matures sensibly.
- **Geometry & modes (post-hoc):** with `n_rank = 2` fixed, use the geometry and
  spatial-signal analyses to check that the *mechanism* (mode activations, SI
  correlate) matches physiology — not just the behavioral summary stats.

The endpoint is not a low `score`; it is a low score achieved **for the right
mechanistic reasons**, verified by the neural/geometry read-outs.
