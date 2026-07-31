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
cd /Users/brycerogers/Documents/antisaccade-maturation
MPLBACKEND=Agg python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke
```

The `smoke` preset intentionally trims everything so a full train+evaluate cycle
is cheap. It is a **diagnostic**, not a fit: the vortex either starts to appear
within ~50 epochs or it doesn't, which tells you immediately whether the
architecture + exogenous burst are wired correctly.

| Knob | `smoke` | `full` | Why smoke shrinks it |
|---|---|---|---|
| `model.n_hidden` | 64 | 200 | fewer units → fast forward/backward |
| `task.t_pre` / `task.t_post` | 100 / 500 | 100 / 500 | full biological timeline in both presets |
| `task.gap_max` | 350 | 350 | full experimental gap range in both presets |
| `task.rpt_step` / `task.rpt_bin_width` | 10 / 20 | 10 / 20 | full analysis grid and training kernel in both presets |
| `train.epochs` | 50 | 1000 | just enough to see a trend |
| `train.batch_size` | 60 | 250 | smaller batches, divisible across five gap strata |
| `train.warmup_epochs` | 10 | 10 | legacy setting; ignored by primary stratified training |
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
  - `task.*`  → `TaskParams`  (biology/task: `threshold`, `a_exo`, `tau_exo`, `sigma_noise`, `sigma_init_shared`, `sigma_init_private`, `commit_temp`, `tau`, `t_post`, `rpt_*`, …)
  - `model.*` → `ModelParams` (`n_hidden`, `n_rank`, `phi`, `init_rec_scale`, `lambda_reg`, `lapse_young_init`, `lapse_adult_init`)
  - `train.*` → `TrainConfig` (`lr`, `epochs`, `batch_size`, `grad_clip`, `resume_checkpoint`, `hard_eval_trials_per_gap`, `plateau_patience`, `plateau_factor`, `seed`, …)
  - `eval.*`  → `EvalConfig`  (`trials_per_gap`, `m_values`)
- **Explicit flags** exist for the common knobs (e.g. `--lr`, `--n-hidden`, `--threshold`, `--a-exo`); see `--help`.
- **`--set task.foo=bar`** is the generic escape hatch for any field without a flag.
- **`--m-values 0,1`** sets both `train.m_choices` and `eval.m_values` together.

Types are coerced automatically from the target field's type (int/float/bool/tuple).

### Lapse and initial-state controls

Training now runs a normal antisaccade branch and a lapse branch. The lapse
branch receives the same cue, go signal, and initial state but has both rule and
maturation inputs removed. The branches are mixed by a learned lapse rate,
`lambda(m)`, whose young and adult endpoints are sigmoid-constrained model
parameters. Tachometric evaluation samples the same lapse process, so it is not
a deterministic replay of the normal branch alone.

Two task parameters control stochastic initial states:

- `task.sigma_init_shared` — one scalar fluctuation shared by all hidden units
  on a trial; principally controls correlation between the two output plans;
- `task.sigma_init_private` — unit-specific initial-state noise; principally
  broadens trial-to-trial variability independently across the hidden state.

Current defaults are `0.3` shared and `0.05` private. Treat them as a coupled
pair: increase shared variability before increasing private variability when
the output plans are insufficiently correlated. The initial values
`model.lapse_young_init` and `model.lapse_adult_init` are legitimate sweep
knobs, but they initialize learned logits rather than fixing lapse rates.
Sweep them only after a healthy race is established, and compare final behavior
rather than treating the initial values as the fitted endpoints.

### Training sampling and scheduler

Primary training samples equal trial counts from five contiguous, equal-width
gap strata spanning `task.gap_min` through `task.gap_max`. Sampling is uniform
within each stratum and shuffled across the batch. Consequently,
`train.batch_size` must be divisible by five. This stratification controls
finite-batch coverage of the full gap range; it does not itself preferentially
weight the vortex region.

`train.warmup_epochs` is retained for compatibility with the legacy
central-hole curriculum helpers, but the primary training path does not use
that curriculum. The `ReduceLROnPlateau` object is also retained, while its
defaults (`plateau_patience=99999`, `plateau_factor=0.99999`) effectively keep
the learning rate fixed. The tachometric loss is too noisy for ordinary
plateau settings to provide reliable reductions.

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

- **behavioral MSE** of the four hard-curve summary stats vs. the Zhu et al.
  targets (`t_rise` and `t_vortex` in ms are down-weighted `1e-4`; `A` and `D`
  weighted `1.0`);
- a **crossing penalty** `5·max(0, 0.4 − frac_crossed)` that punishes configs
  where the decision race rarely reaches threshold θ (a degenerate, untrained
  regime where the curve is meaningless).

`score` is deliberately **not** the raw training loss — a low training loss with
`frac_crossed ≈ 0` is a false victory. Always read `frac_crossed_*` alongside score.

Key per-`m` metrics in `metrics.json` / `results.csv`:
`A_m*`, `t_rise_m*` (75% crossing), `t_vortex_m*`, `D_m*`,
`frac_crossed_m*`, `vortex_depth_m*`, each with a `*_target_*` companion.

### Soft training, hard fitness

The gradient-bearing loss is an rPT-weighted binary cross-entropy computed from
the differentiable mixed-branch trial probabilities (`p_goal_mix`) at their
emergent rPTs. The target parametric tachometric curve is evaluated at each
trial's rPT to provide its probability-correct target. The weights are `3.0`
for 70–200 ms, `1.0` for >200–300 ms, and `0.5` elsewhere. Five-stratum gap
sampling reduces coverage variance, whereas these explicit loss weights are
what upweight the vortex and recovery region.

Hard first-threshold crossings are used separately for periodic training
fitness and for post-training `score` calculation. The hard path applies the
deadline fallback when no option crosses threshold, then fits summary
statistics from the resulting empirical tachometric curve.

The crossing penalty belongs only to post-training ranking; it is not included
in the gradient-bearing loss. It can reject a collapsed race but cannot train
the network away from one. Periodic hard evaluation must therefore track both
summary-statistic error and `frac_crossed`, and viable checkpoint selection
must enforce race health rather than follow the soft loss alone.

Do not compare raw soft loss against the hard `score` as if they were the same
quantity. The soft loss shapes individual gradient updates; the hard curve is
the behavioral measurement. A run with low soft loss but `frac_crossed ≈ 0`
has reached the known false minimum and is not a viable configuration.

---

## 7. Principled sweep order (generalizable framework)

The governing rule: **make the computation exist before you make it pretty, and
make it train stably before you make it biological.** Fix the thing that most
invalidates every downstream measurement first. Concretely, in phases:

### Phase 0 — Plumbing / "does anything cross?"
Nothing else matters until the decision race reaches threshold on a healthy
fraction of trials (`frac_crossed` well above the 0.4 penalty knee, ideally
0.5–0.9). This is a **dynamic-range** problem, not a learning problem.
- Primary knobs: `task.threshold`, readout gain, `task.a_exo`, and initial-state
  variability. `model.init_rec_scale` only changes the initial low-rank factors
  and is not a reliable recovery lever once training dominates initialization.
- After the lapse/initial-state architecture change, re-confirm this gate before
  reusing a previously viable learning rate or rPT grid. `sigma_init_shared` /
  `sigma_init_private` are Phase-0 recovery knobs when their new variability
  prevents the race from reaching threshold.
- Success gate: `frac_crossed_m* ≳ 0.5` at init / very early training.

### Phase 1 — Optimization stability ("can it learn at all?")
Now that gradients flow through a live race, get training to descend cleanly.
- Order: after a crossing check, **learning rate first** (the most impactful
  interacting knob), then `grad_clip` and `batch_size`. `warmup_epochs` does
  not affect primary training.
- Use smoke runs for wiring and short-horizon diagnostics, but re-establish the
  learning rate whenever architecture, capacity, timeline, or training-loss
  resolution changes. Learning-rate viability does not reliably transfer
  across those boundaries.
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
  `task.sigma_noise` (recurrent trial variability → vortex depth `D`),
  `task.sigma_init_shared` / `task.sigma_init_private` (correlated and private
  initial-state variability), and learned-lapse endpoint initializers when the
  crossing gate is already healthy.
- `eval.trials_per_gap` changes only the Monte Carlo precision of hard
  evaluation. `task.rpt_step` and `task.rpt_bin_width` also define the
  differentiable soft-binning objective during training, so treat either as a
  training-loss change: re-confirm crossing and learning-rate stability after
  changing them.
- Success gate: `score` dominated by genuine stat error, not the crossing penalty.

### Phase 4 — Biological / mechanistic tuning (later; see §9)
Once a config reliably fits, interpret and constrain it biologically.

**Sweep hygiene throughout:** sweep **one phase at a time**; keep grids small
(2–3 values per axis, ≤ ~8 combos); freeze the winner via `best_config.json`
before opening the next phase; log every accepted move in
[opt_progress.md](opt_progress.md).

---

## 8. Validation and promotion plan

The harness now operates at the intended timeline, gap range, rPT resolution,
rank, and target hidden size. Further work should preserve that full-scale
problem while separating optimization questions from behavioral measurement:

1. **Establish the live race:** require healthy `frac_crossed` for both
   maturation states before interpreting curve statistics.
2. **Tune at fixed scale:** change one optimization or sampling axis at a time,
   then re-check both the soft loss and hard behavioral diagnostics.
3. **Use stable hard validation:** compare checkpoints with fixed held-out seeds
   and adequate `hard_eval_trials_per_gap`; confirm finalists with fresh,
   larger Monte Carlo evaluation.
4. **Select viable checkpoints:** minimize hard summary-statistic error only
   among checkpoints that pass the crossing gate. Do not select by soft BCE
   alone.
5. **Promote explicitly:** the resolved `config.json` is the reproducibility
   record for the optimization harness. The canonical `run_behavior_fit.py`
   currently reconstructs dataclass defaults rather than loading that file, so
   promotion requires either transferring the resolved values into its runner
   or adding config-loading support before treating the canonical run as an
   exact reproduction.

Rule of thumb: **change one axis per step and require the hard validation gate
to survive it.**

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
- **Initial-state variability (`task.sigma_init_shared`,
  `task.sigma_init_private`):** controls correlated build-up variability and
  therefore rPT spread. Verify output-plan correlation after a substantial
  training run before treating these as fit knobs.
- **Lapse priors (`model.lapse_young_init`, `model.lapse_adult_init`):** set the
  initialization of learned lapse endpoints. Use narrow, ordered pairs with
  young greater than adult; assess final behavior rather than assuming the
  initialized probabilities persist.
- **Maturation axis (`m` / `eval.m_values`):** young↔adult is the scientific
  contrast. Once both endpoints fit, verify the intermediate `m` interpolation
  is monotone and the mechanism (not just the numbers) matures sensibly.
- **Geometry & modes (post-hoc):** with `n_rank = 2` fixed, use the geometry and
  spatial-signal analyses to check that the *mechanism* (mode activations, SI
  correlate) matches physiology — not just the behavioral summary stats.

The endpoint is not a low `score`; it is a low score achieved **for the right
mechanistic reasons**, verified by the neural/geometry read-outs.

---

## 10. Current state of affairs

This section is the progressively maintained snapshot of the active
optimization frontier. Detailed commands and the complete decision history
remain in [opt_progress.md](opt_progress.md).

The current full-scale comparison exposes a tension between behavioral shape
and race health:

- **Run 32 is the best viable operating point.** At `n_hidden=200`, `lr=3e-5`,
  and 200 epochs, it maintains healthy crossing fractions, reproduces the
  maturation-dependent asymptotes, and produces genuine below-chance vortex
  depth. Its vortex and rise timing remain late and its fitted `D` values are
  too shallow.
- **Run 34 is the strongest vortex-shape diagnostic, not a viable fit.**
  Five-stratum gap sampling produces the closest `D` values so far and moves
  vortex/rise timing through the target region, but prolonged training
  collapses `frac_crossed` to 6–9%. The resulting curve is estimated from a
  small, selected subset of crossing trials and is behaviorally degenerate.

Together these runs show that sufficient vortex-region gradient signal exists,
but the current surrogate objective can improve curve shape while sacrificing
the threshold-crossing race. The immediate optimization problem is therefore
to retain Run 32's race health while recovering the timing and depth signal
revealed by Run 34. Hard checkpoint selection must treat `frac_crossed` as a
viability constraint, not merely another score component.
