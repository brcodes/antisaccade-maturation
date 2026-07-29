# code_v2_delta

This document records the code changes that were implemented in the repo after reading the v2 gameplan and the delta note. It is written so a separate model can check whether the implementation matches the intended mechanism, where it is exact, and where it is an approximation.

## What was changed

### 1. Learned lapse mechanism was added to the model

Files:
- [antisaccade_model/model/model_params.py](antisaccade_model/model/model_params.py)
- [antisaccade_model/model/lrrnn.py](antisaccade_model/model/lrrnn.py)

What the code now contains:
- `ModelParams.lapse_young_init = 0.08`
- `ModelParams.lapse_adult_init = 0.02`
- Two trainable logits on the model:
  - `lapse_young_logit`
  - `lapse_adult_logit`
- A new model method:
  - `LRRNN.lapse_rate(m)`

Implemented formula:

```python
λ(m) = λ_adult + (λ_young - λ_adult) * (1 - m)
```

where the endpoints are learned parameters constrained through a sigmoid.

What this means in practice:
- The model can learn different lapse rates for young and adult conditions.
- The interpolation is continuous in `m` and bounded in `[0, 1]`.
- `m = 0` uses the young endpoint, `m = 1` uses the adult endpoint.

### 2. Stochastic initial hidden state was added

Files:
- [antisaccade_model/task/task_params.py](antisaccade_model/task/task_params.py)
- [antisaccade_model/task/trial_generator.py](antisaccade_model/task/trial_generator.py)

New task parameters:
- `sigma_init_shared = 0.3`
- `sigma_init_private = 0.1`

New helper:
- `sample_initial_state(batch_size, n_hidden, task, generator=None)`

Implemented sampling rule:

```python
h0 = shared + private
shared  ~ N(0, sigma_init_shared^2)
private ~ N(0, sigma_init_private^2 I)
```

The shared term is scalar per trial and broadcasts across units.

What this means in practice:
- Training and analysis now have a trial-level initial-state source of RT variability.
- This is intended to mimic correlated build-up variability, not to model an exact biological latent state.

### 3. Trial generation now supports lapse masking

File:
- [antisaccade_model/task/trial_generator.py](antisaccade_model/task/trial_generator.py)

Change:
- `build_inputs(...)` now accepts `lapse_mask`.
- When a trial is marked as a lapse, the rule input channel is zeroed for that trial.

Implemented behavior:
- Go and cue channels are still constructed normally.
- Maturation scalar `m` is still passed through normally.
- Only the antisaccade rule channel is suppressed on lapse trials.

Important note:
- This is the exact code path used by the training loss to create a lapse branch.
- It is a proxy for the intended cognitive lapse mechanism, not a full mechanistic re-implementation of Salinas et al.'s circuit dynamics.

### 4. Behavioral loss now has two pieces

File:
- [antisaccade_model/training/losses.py](antisaccade_model/training/losses.py)

The loss now combines:
- summary-statistic MSE on the extracted tachometric curve
- an auxiliary per-trial curve pressure term weighted by rPT
- regularization

New helper:
- `rpt_weight(rpt)`

Weighting rule implemented:

```python
70 <= rPT <= 200   -> weight 3.0
200 < rPT <= 300   -> weight 1.0
otherwise          -> weight 0.5
```

Training loss flow:
1. Run the normal branch with the trial batch.
2. Run a lapse branch with the same initial state and cue/go inputs, but with the rule input zeroed.
3. Mix the two branches using `λ(m)`.
4. Build the soft tachometric curve from the mixed branch output.
5. Compute summary-stat MSE and the rPT-weighted auxiliary loss.

New variables used in the loss:
- `soft_commit`
- `soft_lapse`
- `p_goal_mix`
- `rpt_mix`
- `curve_loss`

### 5. Training now supplies `h0`

File:
- [antisaccade_model/training/train.py](antisaccade_model/training/train.py)

Change:
- `make_batch(...)` now samples `h0` and includes it in each training batch.

The model is trained with:
- `model(batch["u"], h0=batch.get("h0"), add_noise=True)`

Checkpoint loading change:
- `load_checkpoint(...)` now uses `strict=False` so older checkpoints can still load even though the model has new lapse parameters.

### 6. Analysis now uses the new lapse and initial-state machinery

Files:
- [antisaccade_model/analysis/tachometric_analysis.py](antisaccade_model/analysis/tachometric_analysis.py)
- [antisaccade_model/analysis/spatial_signal.py](antisaccade_model/analysis/spatial_signal.py)

Changes:
- Tachometric sweeps now pass `h0` into the model.
- Tachometric sweeps also sample lapse trials using `model.lapse_rate(m)`.
- Spatial-signal analysis uses the same machinery for consistency.

Important consequence:
- The analysis path is no longer a pure deterministic replay of the old code; it now includes the learned lapse probability and stochastic initial hidden state.

## New variables and parameters

### Model-level
- `lapse_young_logit`
- `lapse_adult_logit`

### ModelParams
- `lapse_young_init`
- `lapse_adult_init`

### TaskParams
- `sigma_init_shared`
- `sigma_init_private`

### Loss helpers
- `rpt_weight(rpt)`

### Batch fields now used by training / analysis
- `h0`
- `lapse`

## Assumptions encoded in the implementation

### 1. The lapse mechanism is modeled as a branch mixture

The code does not simulate a separate full biological lapse circuit. Instead, it treats lapse trials as a branch where the antisaccade rule input is suppressed, then mixes the lapse and non-lapse branches using `λ(m)`.

This is an approximation, but it keeps the lapse mechanism explicit and trainable.

### 2. The rule input is the lapse-sensitive control channel

The implementation assumes the clearest way to express a lapse in the current architecture is to suppress the rule channel, not to alter all inputs or add a separate decision module.

### 3. Initial-state variability is additive and factorized

The code assumes trial-to-trial RT variability can be approximated by one shared scalar fluctuation plus independent unit-wise noise.

This is a compact proxy for the correlated build-up variability discussed in the gameplan.

### 4. The analysis path uses the learned lapse rate

Tachometric and SI analyses now sample lapse trials with `model.lapse_rate(m)`.

That means the post-hoc behavior prediction is no longer based only on the raw deterministic recurrent dynamics.

### 5. `m` is still a direct input channel

The existing mechanism-1 design is unchanged:
- maturation is concatenated to the input stream as a constant channel
- no maturation-specific recurrent subnetwork was added

## What remains approximate or simplified

- The lapse implementation is not a full Salinas-style detailed circuit model; it is a learnable branch mixture.
- The initial-state variance is not derived from a fitted covariance model for build-up rates; it is a shared-plus-private heuristic.
- The rPT-weighted auxiliary term is a training pressure term, not the sole objective.
- The code does not currently force all analysis helpers to use identical RNG seeds, so exact numeric outputs may vary between runs.

## Validation status

Implemented code paths were byte-compiled successfully with:

```bash
python -m compileall antisaccade_model
```

## Practical interpretation for a reviewer

If you are checking whether the implementation matches the gameplan at a high level, the answer is:
- yes for emergent rPT handling
- yes for mechanism-1 maturation conditioning
- yes for a trainable lapse-rate interpolation
- yes for stochastic initial-state variability
- yes for rPT-weighted training pressure

If you are checking whether it is a literal one-to-one reproduction of the prose mechanism, the answer is:
- not exactly
- it is a compact differentiable approximation intended to preserve the modeled behavior and the intended training signal