# Antisaccade Maturation Modeling

A low-rank recurrent neural network (LR-RNN) conditioned on a scalar *maturation
state* that is trained **only** on behavioral tachometric curves (young vs.
adult monkeys) and then used to **predict** the FEF/dlPFC spatial signal
`SI(t, rPT)` post-hoc.

Reference: Zhu, Zhou, Constantinidis, Salinas & Stanford (2024), *iScience*
27(8):110488. Architecture follows Mastrogiuseppe & Ostojic (2018).

See [`antisaccade_maturation_modeling_gameplan_v2.md`](antisaccade_maturation_modeling_gameplan_v2.md)
for the full scientific and implementation plan.

## Fixed design choices

| Choice | Value |
|---|---|
| Connectivity rank | R = 2 |
| Hidden units | N = 200 |
| Maturation conditioning | Mechanism 1 (m as a constant input channel) |
| Non-differentiable decision | Straight-through estimator (hard forward / soft backward) |
| rPT | Emergent: gap imposed, threshold crossing sets `t_commit`, `rPT = t_commit − t_cue` |
| Behavioral objective | rPT-weighted BCE against target tachometric curves; hard summary statistics for evaluation |
| Training maturation states | Discrete m ∈ {0, 1} |
| Cross-prediction | Direction 1 (fit behavior → predict SI) |
| Compute | CPU only |

Training mixes a normal antisaccade branch with a rule- and
maturation-ablated lapse branch using learned young/adult lapse endpoints.
Shared and private stochastic initial-state components provide trial-level RT
variability. The differentiable BCE objective trains this mixture, while hard
threshold crossings, empirical tachometric curves, and `frac_crossed` determine
whether behavior is scientifically viable.

## Repository layout

```
antisaccade_model/
├── task/            trial generation and behavioral (tachometric) targets
├── model/           low-rank RNN and readout
├── training/        training loop, losses, curriculum
├── analysis/        tachometric fit, spatial signal (SI), population geometry
├── visualization/   plotting utilities
├── experiments/     runnable entry points
└── requirements.txt
```

## 1. Environment setup

Python ≥ 3.10 is required (CPU only — no GPU packages).

```bash
# from the workspace root
python -m venv .venv            # optional but recommended
source .venv/bin/activate

pip install -r antisaccade_model/requirements.txt
```

Dependencies: `torch`, `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `tqdm`.

> All commands below are run **from the workspace root**
> (`/Users/brycerogers/Documents/antisaccade-maturation/`) and invoke the
> package with `python -m ...` so that the relative imports resolve.

## 2. Sanity check (optional)

Confirm every module byte-compiles before a full run:

```bash
python -m py_compile antisaccade_model/**/*.py antisaccade_model/*.py && echo "OK"
```

## 3. Main experiment — fit behavior, predict SI (Direction 1)

This is the primary study path. It trains the model, extracts and plots the
tachometric curves, prints the summary-statistic table (model vs. target),
predicts `SI(t, rPT)` for young and adult, and runs the population-geometry
analyses.

```bash
python -m antisaccade_model.experiments.run_behavior_fit
```

Outputs:

- `checkpoints/behavior_fit.pt` — trained model + config + training history.
- `results/behavior_fit/tachometric.png` — model vs. target tachometric curves.
- `results/behavior_fit/si_heatmaps.png` — predicted `SI(t, rPT)` (young, adult).
- `results/behavior_fit/mode_activations.png` — low-rank mode time courses.
- Console: summary-statistic table, SI correlation, participation ratio, PCA
  variance explained.

To adjust epochs or reuse an existing checkpoint, edit the call at the bottom of
[`run_behavior_fit.py`](antisaccade_model/experiments/run_behavior_fit.py) or
call `main(epochs=..., retrain=False)` from a Python session:

```bash
python -c "from antisaccade_model.experiments.run_behavior_fit import main; main(epochs=2000)"
```

`run_behavior_fit.py` constructs `DEFAULT_TASK`, `DEFAULT_MODEL`, and a fresh
`TrainConfig`; it does not load an optimization-harness `config.json`.
Reproducing a promoted optimizer run therefore requires transferring its
resolved values into the canonical runner or adding config-loading support.

## 4. Ablation analyses

Requires a checkpoint from step 3. Lesions the maturation input channel and each
low-rank mode, and reports the effect on behavior and on the adult spatial
signal (mechanism-1-appropriate versions of the gameplan Section 5.2 ablations).

```bash
python -m antisaccade_model.experiments.run_ablations
```

## 5. Maturation sweep (intermediate m)

Evaluate the trained model at intermediate maturation states m ∈ {0, 0.1, …, 1}
to check that behavioral and neural statistics vary monotonically (gameplan
Section 5.1). This is a library call rather than a standalone script:

```bash
python -c "
from antisaccade_model.training.train import load_checkpoint
from antisaccade_model.analysis.maturation_sweep import maturation_sweep
model, ckpt = load_checkpoint('checkpoints/behavior_fit.pt')
out = maturation_sweep(model, ckpt['task'])
print('m       :', out['m_values'])
print('t_rise  :', out['t_rise'])
print('A       :', out['A'])
print('t_vortex:', out['t_vortex'])
print('D       :', out['D'])
"
```

## 6. Optional — Direction 2 (fit SI, predict behavior)

Not part of the primary study. This is a documented scaffold: it requires an
empirical `SI(t, rPT)` target (Zhu et al. 2024 data) and a differentiable SI
surrogate before it can be run end to end. Running it prints setup guidance and
performs a forward-pass sanity check.

```bash
python -m antisaccade_model.experiments.run_neural_fit
```

## Typical end-to-end sequence

```bash
source .venv/bin/activate
pip install -r antisaccade_model/requirements.txt

python -m antisaccade_model.experiments.run_behavior_fit   # train + behavior + SI + geometry
python -m antisaccade_model.experiments.run_ablations      # ablations (needs checkpoint)
python -c "from antisaccade_model.training.train import load_checkpoint; \
from antisaccade_model.analysis.maturation_sweep import maturation_sweep; \
m,c=load_checkpoint('checkpoints/behavior_fit.pt'); print(maturation_sweep(m,c['task'])['t_rise'])"
```

## Using real behavioral data

The young/adult target parameters in
[`antisaccade_model/task/tachometric_targets.py`](antisaccade_model/task/tachometric_targets.py)
(`YOUNG_PARAMS` / `ADULT_PARAMS`) are the gameplan's approximations of Zhu et al.
(2024) Figs. 3–4. Replace these constants with fitted values from the empirical
per-subject tachometric curves to fit the real data.
