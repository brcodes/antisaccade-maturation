# opt_progress.md — Behavior-Fit Optimization Log

Live results + decision log for the optimization harness. Reference doc (modes, artifacts, sweep philosophy, scale-up) is [OPT_README.md](OPT_README.md).

Append one dated entry per accepted finding. Record the command, the key numbers, the interpretation, and the resulting decision. Keep the current best config and next steps sections at the top up to date.

---

## Current best config (living)

| field | value | source |
|---|---|---|
| `task.threshold` | 0.75 | sweep 3 |
| `task.a_exo` | 3.0 | sweep 3 |
| `task.tau_exo` | 30.0 (confirmed = library default) | sweep 4 |
| `train.lr` | 3e-3 | sweep 11 |
| `model.n_hidden` | 100 | sweep 11 |
| `train.epochs` | 300 | sweep 9 |
| `train.batch_size` | 256 (library default) | confirmed |
| `task.t_pre` | 100 (restoring full) | pending |
| `task.t_post` | 500 (restoring full) | pending |
| `task.rpt_step` | 10 (restoring full) | pending |

**Current phase:** Phase 2 capacity confirmed at n_hidden=100. At practical limit of smoke resolution — scaling up to full timeline and bin resolution before further hyperparameter search.

---

## Next steps (specific, ordered)

- Restore full timeline and measurement resolution. Single diagnostic run, no sweep. Share training log to confirm loss curve cleans up:
  ```bash
  python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
      --set task.threshold=0.75 \
      --set task.a_exo=3 \
      --set task.tau_exo=30 \
      --set model.n_hidden=100 \
      --set train.lr=3e-3 \
      --set train.epochs=300 \
      --set task.t_pre=100 \
      --set task.t_post=500 \
      --set task.rpt_step=10 \
      --no-plots
  ```
  Success gate: loss shows genuine downward trend rather than 3.5–9.2 pinball. Score improves meaningfully vs 0.132.

- If full resolution stabilizes training, step n_hidden to 128 at the new resolution, re-confirming LR stability. Then 200.

- Phase 3 (biological, deferred): `commit_temp`/`option_temp`, `sigma_noise` → vortex depth D, `a_exo`/`tau_exo` → vortex timing/depth, maturation interpolation across m. Do not touch until a config trains and fits at full resolution.

---

## 2026-07-28

### Targets bug fix

Before any sweeps, discovered that `tachometric_targets.py` was not the source of truth feeding into the scorer. The actual coded targets are:

```python
YOUNG_PARAMS = TCParams(A=0.92, t_rise=155.0, sigma_rise=25.0,
                        t_vortex=105.0, D=0.28, sigma_vortex=25.0)
ADULT_PARAMS = TCParams(A=0.97, t_rise=140.0, sigma_rise=15.0,
                        t_vortex=106.0, D=0.27, sigma_vortex=20.0)
```

The results.csv target columns in sweep 1 showed t_vortex≈93.9 (young) and D≈0.205 — meaningfully off from the above. Fix was applied before sweep 2. All sweeps from sweep 2 onward use correct targets. Sweep 1 rankings are directionally useful but not trusted for final comparison.

---

### Sweep 1 — θ grid, smoke, wrong targets

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --sweep task.threshold=0.2,0.3,0.5,0.7 --no-plots
```

θ=0.2 won on score (0.250) but only because the wrong D target (0.205) made its D=0.45 output look acceptable. θ=0.3 and θ=0.2 produced nan t_rise for m=0 — likely rpt_step=30 too coarse for interpolation. θ=0.7 scored second (0.383) with frac_crossed≈0.50. Rankings invalidated by wrong targets; re-run as sweep 2.

---

### Sweep 2 — θ grid, smoke, corrected targets

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --sweep task.threshold=0.2,0.3,0.5,0.7 --no-plots
```

Target fix completely reshuffled rankings. θ=0.7 wins (score 1.522), θ=0.2 drops to 3rd (score 2.550). θ=0.7 is the only config that hit D≈0.28 for m=0 nearly exactly — mechanistically correct, since higher θ gives the exogenous burst more time to drag the accumulator down. However t_rise=230ms (+75ms late) and t_vortex=200ms (nearly double target). The model finds the right shape but timing is badly stretched. Nan plague on t_rise for three of four configs — rpt_step coarseness flagged but not addressed yet. θ=0.7 adopted as working baseline.

---

### Sweep 3 — θ × a_exo grid

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --sweep task.threshold=0.6,0.65,0.7,0.75,0.8 \
    --sweep task.a_exo=3,5 \
    --no-plots --top 5
```

θ=0.75, a_exo=3 wins (score 0.297) with t_vortex=106ms nearly on target, but D collapses to ~0. θ=0.7, a_exo=3 holds D≈0.28 (exact) but t_vortex=200ms. a_exo=5 universally worse across all θ — overpowers the race and distorts A. Key finding: D and t_vortex are anti-correlated across θ at fixed a_exo. Need a second degree of freedom to decouple them. τ_exo identified as the candidate. a_exo locked at 3.0.

---

### Sweep 4 — θ × τ_exo grid

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.a_exo=3 \
    --sweep task.threshold=0.7,0.72,0.75,0.78 \
    --sweep task.tau_exo=10,20,30 \
    --no-plots --top 6
```

τ_exo=10 catastrophic — kills frac_crossed to 0.00 on m0 for nearly all θ. τ_exo=20 marginal. τ_exo=30 (library default) clearly correct. The D vs t_vortex anti-correlation is structural and not fixable with τ_exo — the θ tradeoff is the same regardless of burst timescale. τ_exo locked at 30. Winner is again θ=0.75/τ_exo=30 (score 0.297, same as sweep 3). D=0 at θ=0.75 flagged for investigation.

---

### D=0 diagnosis — vortex is real, Gaussian fit is failing

Code inspection confirmed: model D is the Gaussian vortex amplitude from a parametric fit to the binned tachometric curve, not the raw minimum dip. The extractor defines D as 0.5 minus the soft minimum in the vortex region, but if the dip is too narrow or the bins are sparse, `curve_fit` fails to estimate covariance and returns D≈0 even when `vortex_depth` (the raw minimum) is genuinely below chance.

In sweep 4 results, D_m0≈0 while vortex_depth_m0 is negative — confirming the empirical binned curve is crossing below chance but the Gaussian fit cannot recover D from it. This is a measurement artifact at smoke resolution, not an absent mechanism.

---

### Sweep 5 — epoch count, vortex emergence check

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 \
    --set task.a_exo=3 \
    --set task.tau_exo=30 \
    --sweep train.epochs=50,150,300,500 \
    --no-plots --top 4
```

Critical finding: vortex_depth < 0 at all four epoch counts — the curve crosses below chance from the very start. The vortex mechanism is working. D≈0 at 50 and 500 epochs is purely a Gaussian fit failure on a real but shallow/narrow dip. Best genuine fit at 300 epochs (D=0.143, t_rise≈159ms). 500 epochs shows t_vortex drifting to 70ms — possible instability past an optimum. Smoke epoch budget upgraded from 50 → 300.

Also confirmed: a_exo is implemented as a positive side-specific input (not subtractive). The below-chance dip must emerge from learned recurrent weights inverting the burst into a wrong-direction bias — it cannot be guaranteed architecturally and must be learned. This means D cannot appear at initialization; it requires sufficient training.

---

### Sweep 6 — θ × a_exo at 300 epochs (honest ranking)

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set train.epochs=300 \
    --set task.tau_exo=30 \
    --sweep task.threshold=0.7,0.72,0.75,0.78 \
    --sweep task.a_exo=3,5,8 \
    --no-plots --top 6
```

θ=0.75, a_exo=3 holds as winner (score 0.531). t_vortex has converged to ~115–118ms across nearly all viable configs — the timing is no longer a free variable, τ_exo=30 is the dominant controller. a_exo=8 eliminated — kills frac_crossed at θ=0.72 and 0.78. θ=0.78 consistently problematic. θ=0.7/a_exo=5 is a competitive second (t_rise=152ms nearly perfect, D=0.167). Remaining variance is in A (undertrained, ~0.78–0.80) and D (real but shallow). Phase 1 gate reached.

---

### Sweep 7 — LR search, Phase 1

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 \
    --set task.a_exo=3 \
    --set task.tau_exo=30 \
    --set train.epochs=300 \
    --sweep train.lr=1e-4,3e-4,1e-3,3e-3 \
    --no-plots --top 4
```

LR=1e-3 wins on score (0.531). LR=3e-3 produces best A (0.835) and best D (0.362) but overshoots t_rise (134ms). LR≤3e-4 catastrophically dead — frac_crossed=0.00, loss artificially low. LR=3e-4 achieved the lowest training loss while producing a completely degenerate curve — textbook false minimum, confirms OPT_README §6. Sweet spot is between 1e-3 and 3e-3.

---

### Sweep 8 — fine LR search

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 \
    --set task.a_exo=3 \
    --set task.tau_exo=30 \
    --set train.epochs=300 \
    --sweep train.lr=1e-3,1.5e-3,2e-3,3e-3 \
    --no-plots --top 4
```

No clean optimum in the gap. LR=2e-3 scores third despite sitting between the two better configs — t_rise blows to 229ms, worse than either neighbor. Different LRs landing in different basins rather than smoothly interpolating. LR=1e-3 locked as Phase 1 conservative choice (best score, most stable). D m1 weak across all configs — flagged as systemic. Note: LR tuning is model-size dependent; re-confirm after any n_hidden change.

---

### Sweep 9 — epoch scale-up at locked config

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 \
    --set task.a_exo=3 \
    --set task.tau_exo=30 \
    --set train.lr=1e-3 \
    --sweep train.epochs=300,500,750,1000 \
    --no-plots --top 4
```

A m0 stuck between 0.769–0.799 across all epoch counts — no upward trend. Capacity ceiling at n_hidden=64 confirmed: training longer cannot push A toward 0.92. 500 epoch basin instability noted (t_vortex crashes to 70ms, same as sweep 5) — warmup schedule ending may be hitting a saddle point. D m1 degrades with more epochs, consistent with a capacity-constrained model trading off adult state quality to seek young state gains. Phase 2 signal: raise n_hidden.

---

### Sweep 10 — n_hidden capacity sweep

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 \
    --set task.a_exo=3 \
    --set task.tau_exo=30 \
    --set train.lr=1e-3 \
    --set train.epochs=300 \
    --sweep model.n_hidden=64,100,128,200 \
    --no-plots --top 4
```

All n_hidden > 64 fell into the dead-race false minimum (frac_crossed=0.00) at LR=1e-3. n_hidden=128 additionally diverged (loss=11, A=0.5, D=0.6 — collapsed readout). Root cause: LR=1e-3 was tuned for n_hidden=64. Larger models have more parameters and different loss landscape curvature; the same LR is too small relative to gradient scale and the optimizer slides into the dead-race basin before the race is established.

**Key principle:** model size increases may require LR adjustment in either direction — decreases if greater capacity causes instability, increases if greater capacity does not outweigh the now-larger initialization inertia. Always re-confirm LR after any n_hidden change.

Proceeded conservatively: re-run LR search at n_hidden=100 (smallest increment) before jumping to 128.

---

### Sweep 11 — LR re-search at n_hidden=100

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 \
    --set task.a_exo=3 \
    --set task.tau_exo=30 \
    --set train.epochs=300 \
    --set model.n_hidden=100 \
    --sweep train.lr=1e-3,2e-3,3e-3,5e-3 \
    --no-plots --top 4
```

Viable LR threshold is between 2e-3 and 3e-3. LR=3e-3 and LR=5e-3 both achieve frac_crossed=1.00 and converge to identical results (score=0.132, A=0.850, t_rise=161ms, t_vortex=108ms, D=0.450) — confirmed different seeds, genuine shared attractor. LR=3e-3 locked as minimum viable LR for n_hidden=100 (conservative choice, no benefit to going higher). Score 0.132 vs 0.531 at n_hidden=64 — genuine capacity improvement. A jumped from 0.799 to 0.850, t_vortex hit 108ms (nearly on target). D=0.450 with vortex_depth=nan flagged as likely Gaussian fit artifact at 300 epochs rather than true value.

LR=2e-3 curiosity: frac_crossed=0.00 but A=0.932 — model learned a high asymptote without the race ever crossing threshold. Classic false minimum with well-trained readout but broken decision mechanism.

---

### Sweep 12 — grad_clip sweep (ruled out)

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 \
    --set task.a_exo=3 \
    --set task.tau_exo=30 \
    --set model.n_hidden=100 \
    --set train.lr=3e-3 \
    --set train.epochs=300 \
    --sweep train.grad_clip=0.5,1.0,2.0,5.0 \
    --no-plots --top 4
```

grad_clip=1.0 produced identical loss values to the unclipped sweep 11 run — library default is already 1.0, sweeping it changed nothing. Spikes persist (loss oscillating 3.5–9.2) across all clip values while crossed=1.00 throughout. Gradient explosion ruled out: a true explosion would destabilize crossed, but it stays locked. grad_clip=5.0 was actually the cleanest curve, suggesting the current default clip is already slightly too tight.

Root cause of spikes: the behavioral summary-statistic loss is computed from a stochastic Monte Carlo tachometric curve. Different random trial batches give different curve estimates, so the loss has inherent epoch-to-epoch variance. This is not a gradient problem — it is loss noise from the objective itself. batch_size confirmed at library default of 256 — already large enough that further increases would not materially help.

**Diagnosis: at the practical limit of smoke resolution.** The shortened timeline (t_post=250 vs full 500) truncates the gradient on A and t_rise. The coarse bins (rpt_step=30 vs full 10) give ~6 bins across the rPT range instead of ~18, making t_vortex and D estimates unreliable. The optimizer is working with a blurry objective. The loss landscape degeneracy (two different LRs, different loss trajectories, identical behavioral outputs) is a direct consequence of low resolution blurring the loss surface. Next step is restoring full timeline and bin resolution before any further hyperparameter search.

python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 \
    --set task.a_exo=3 \
    --set task.tau_exo=30 \
    --set model.n_hidden=100 \
    --set train.lr=3e-3 \
    --set train.epochs=300 \
    --set task.t_pre=100 \
    --set task.t_post=500 \
    --set task.rpt_step=10 \
    --no-plots 

Run 13 — Dead race. Same false minimum as sweep 10's n_hidden=128 divergence.
The culprit is the resolution change, not the model. When you restored t_post=500 and rpt_step=10, the loss landscape shifted under the same LR=3e-3. The longer timeline means more Euler steps per trial, which changes gradient scale. The finer bins (18 vs ~6) mean the behavioral objective is more sensitive — small deviations now matter more, making the penalty basin deeper and wider relative to the learning signal at epoch 0.

This is the same phenomenon as sweep 10 (LR tuned for n_hidden=64 failed at n_hidden=100): the locked LR doesn't transfer across resolution changes.

LR re-search at full resolution

Same principle as sweep 11, but now at n_hidden=100 + full timeline. The dead-race false minimum at LR=3e-3 means we need to push LR higher to escape initialization inertia at the new resolution. In sweep 11 the viable threshold was between 2e-3 and 3e-3 at smoke resolution — expect it to shift upward here.

Sweep 14 — Partial escape at LR=1.2e-2, but still not clean.

What happened: LR=1.2e-2 is the only config that broke out of the dead-race lock — crossed recovered to ~0.5 around epoch 190–210, then wandered between 0.25–0.60 for the rest of training. Loss dropped from 11.047 to ~7.6–8.1 in that window. But the metrics tell a sobering story: m0 and m1 are still identical (t_rise, A, t_vortex, D all the same for both states), and frac_crossed_m0=0.00 while frac_crossed_m1=0.60. The model is crossing on m1 but not m0 — the race is alive for one maturation state and dead for the other.

The three lower LRs (3e-3, 5e-3, 8e-3) all locked at the false minimum from epoch 10 onward. The viable LR threshold at full resolution is somewhere above 8e-3.

Key observations:

The race escapes around epoch 180–190 at LR=1.2e-2, not during warmup. This suggests the warmup schedule isn't helping establish the race early enough at full resolution — the model drifts into the dead basin first, then eventually escapes via noise at high LR.
m0 never crossing (frac_crossed_m0=0.00) while m1 crosses (0.60) is a new asymmetry. The young state (m=0) is harder to train than the adult state — possibly because the young tachometric curve shape (larger D, shallower A at 0.92) is harder to achieve with the current initialization.
The 7.59 loss at epoch 299 vs 8.10 in the middle suggests the curve hasn't stabilized — it's still moving at termination.

The loss is still noisy and the race is asymmetric. We need to push LR higher to see if we can get both states crossing reliably, and run longer to see if the instability resolves.

Why 500 epochs: at LR=1.2e-2 the race didn't escape until epoch ~190, meaning only ~110 epochs of actual learning happened before termination. We need to see whether crossed stabilizes and loss trends down after the escape, or whether it keeps wandering. 500 gives ~300 post-escape epochs to evaluate.

Why push LR higher: LRs 5e-3 and 8e-3 never escaped at all. 1.2e-2 escaped late and asymmetrically. The pattern from sweeps 7–8 and 11 suggests the viable basin has a sharp lower edge — we may need to get comfortably above it to get clean symmetric crossing from early in training.

What to watch in the log: the critical signal is whether crossed recovers during the warmup window (epochs 0–30ish) rather than at epoch 180. Early crossing means the LR is high enough to establish the race before the dead basin captures it. Also watch whether m0 and m1 cross together or whether the asymmetry persists.

Success gate: at least one config with both frac_crossed_m0 ≥ 0.4 and frac_crossed_m1 ≥ 0.4, loss trending downward in the final 100 epochs, score meaningfully below 4.13.

python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 \
    --set task.a_exo=3 \
    --set task.tau_exo=30 \
    --set model.n_hidden=100 \
    --set task.t_pre=100 \
    --set task.t_post=500 \
    --set task.rpt_step=10 \
    --set train.epochs=500 \
    --sweep train.lr=1.2e-2,1.5e-2,2e-2,3e-2 \
    --no-plots --top 4

The pattern is now clear and damning. LR=1.2e-2 is the only value that ever gets any crossing, and even then only for m1, never m0, and only transiently (frac_crossed_m0=0.00 the whole time). LRs from 1.5e-2 upward are more dead than 1.2e-2 — the reg column ticking up (0.00002 → 0.00003) as LR increases just means the optimizer is growing the weights while running in place in the dead basin. That's not a signal, it's noise.

The LR window that actually works at full resolution is narrower than a single order of magnitude and sits around 1.2e-2 with unreliable, asymmetric crossing. There's no clean viable basin here. Chasing it further is diminishing returns.

The diagnosis: n_hidden=100 at full resolution has a worse Phase 0 problem than n_hidden=64 did at smoke resolution. The longer timeline (more Euler steps) means the accumulator has more time to drift away from threshold before the learning signal can establish the race. The initialization that worked at smoke (crossed=0.11 at epoch 0, then warmup carries it) can't survive 500 timesteps of unguided integration.

Plan: Fall back to n_hidden=64, full resolution, fresh LR search

The logic: n_hidden=64 did train at full resolution in principle — we just never verified it because we jumped to n_hidden=100 (sweep 11) while still on smoke. The smoke score of 0.531 at n_hidden=64/LR=1e-3 is real. The question is whether that LR transfers to full resolution or needs adjustment, just as it did for n_hidden=100.

Sweep 16 — LR search at n_hidden=64, full resolution:
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 \
    --set task.a_exo=3 \
    --set task.tau_exo=30 \
    --set model.n_hidden=64 \
    --set train.epochs=300 \
    --set task.t_pre=100 \
    --set task.t_post=500 \
    --set task.rpt_step=10 \
    --sweep train.lr=1e-3,3e-3,5e-3,8e-3 \
    --no-plots --top 4

    Success gate: at least one config with both frac_crossed_m0 ≥ 0.4 and frac_crossed_m1 ≥ 0.4, crossing established during warmup (not at epoch 180), and score below 2.0. If LR=1e-3 survives the resolution change, we accept it and move on. If it dies and something higher works, we lock the new LR and proceed to epoch scale-up at full resolution.

    If nothing crosses at full resolution even at n_hidden=64, that is a Phase 0 problem with the full timeline itself — at that point we'd look at init_rec_scale or warmup length before touching anything else.