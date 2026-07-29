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
| `model.n_hidden` | 64 | sweep 18 — n_hidden=100 abandoned at full resolution (sweeps 14–15) |
| `train.lr` | unknown — re-search required post-architecture | sweep 18 note |
| `train.epochs` | 300 | sweep 9 |
| `train.batch_size` | 256 (library default) | confirmed |
| `task.t_pre` | 50 (smoke default) | low priority; see sweep 18 note |
| `task.t_post` | 250 | sweep 18 |
| `task.rpt_step` | 30 | sweep 19 attempted; both 30 and 10 ended in a dead race |
| `task.sigma_init_shared` / `task.sigma_init_private` | 0.7 / 0.05 | architecture validation; next crossing-recovery sweep |
| `model.lapse_young_init` / `model.lapse_adult_init` | 0.08 / 0.02 | learned endpoint initializers; not yet swept |

**Note on n_hidden:** The original config table listed n_hidden=100/LR=3e-3 from sweep 11, but sweep 11 was smoke-resolution only. Sweeps 14–15 showed n_hidden=100 is not viable at full resolution — no LR produces a functioning race at t_post=500. Sweep 16 explicitly fell back to n_hidden=64 and sweeps 17–18 continued there. The operative model is n_hidden=64. n_hidden=100 at smoke resolution remains a useful data point but is not the current baseline.

**Note on LR:** At n_hidden=64/t_post=250 the LR was never directly tested post-sweep-18 (sweep 18 used LR=8e-3 specifically for the t_post isolation — it was chosen because it was the only survivor at t_post=500, not because it was optimal at t_post=250). LR must be re-searched at n_hidden=64/t_post=250 after architectural changes. At smoke resolution n_hidden=64 used LR=1e-3 (sweeps 7–9). At full resolution n_hidden=64 needed LR=8e-3 to hold the race at t_post=500. At t_post=250 with new architecture: unknown.

**Current phase:** Architecture is complete, but sweep 19 ended with a dead
race at both rPT resolutions and at two previously viable learning rates. This
is a Phase-0/1 recovery problem, not evidence for or against `rpt_step=10`.
Restore healthy hard crossings before comparing bin resolution, re-searching
learning rate, or retesting `t_post=500`.

---

## Next steps (specific, ordered)

1. **Sweep 20a — crossing recovery** — hold the last known viable timeline and
grid (`n_hidden=64`, `t_post=250`, `rpt_step=30`) fixed. Sweep the new
initial-state pair around its current setting, starting with lower total
variance: `sigma_init_shared=0.3,0.5,0.7` and
`sigma_init_private=0.0,0.05`. Run the paired grid at `lr=1e-3` and rank by
`frac_crossed_*`, then score. Do not open lapse-endpoint sweeps until both
states cross healthily.

2. **Sweep 20b — LR re-search** — use the recovered initial-state setting,
then sweep `lr=1e-3,3e-3,5e-3,8e-3` at `rpt_step=30`. The lapse branch and h0
change loss geometry, so pre-architecture rankings do not transfer. Watch hard
`frac_crossed_*` first; raw loss is only a secondary diagnostic.

3. **Sweep 20c — rPT-grid retry** — after a viable initial-state/LR pair is
found, repeat `rpt_step=30,10`. `rpt_step=10` changes the soft-binning loss, so
accept it only if both hard crossing fractions remain healthy and score is
comparable at the same evaluation budget.

4. **Sweep 21 — t_post=500 retest** — use the recovered LR and rPT setting.
The prior `t_post=500` failures predate the lapse branch, but this test is only
meaningful once the short-window race is healthy.

5. **Lapse and capacity follow-up** — after the crossing/LR/grid gates pass,
test narrow young/adult lapse-initializer pairs, then reconsider `n_hidden=100`.
The endpoint values are learned, so assess their behavioral effect rather than
assuming the initialized probabilities persist.

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

θ=0.2 won on score (0.250) but only because the wrong D target (0.205) made its D=0.45 output look acceptable. θ=0.3 and θ=0.2 produced nan t_rise for m=0 — rpt_step=30 too coarse for interpolation. θ=0.7 scored second (0.383) with frac_crossed≈0.50. Rankings invalidated by wrong targets; re-run as sweep 2.

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

Code inspection confirmed: model D is the Gaussian vortex amplitude from a parametric fit to the binned tachometric curve. If the dip is too narrow or the bins are sparse, `curve_fit` fails and returns D≈0 even when `vortex_depth` (the raw minimum) is genuinely below chance. In sweep 4 results, D_m0≈0 while vortex_depth_m0 is negative — confirming the empirical binned curve is crossing below chance but the Gaussian fit cannot recover D from it. This is a measurement artifact at smoke resolution, not an absent mechanism.

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

Critical finding: vortex_depth < 0 at all four epoch counts — the curve crosses below chance from the very start. The vortex mechanism is working. D≈0 at 50 and 500 epochs is purely a Gaussian fit failure on a real but shallow/narrow dip. Best genuine fit at 300 epochs (D=0.143, t_rise≈159ms). 500 epochs shows t_vortex drifting to 70ms — possible instability past an optimum. Smoke epoch baseline upgraded from 50 → 300.

Also confirmed: a_exo is implemented as a positive side-specific input (not subtractive). The below-chance dip must emerge from learned recurrent weights inverting the burst into a wrong-direction bias — it cannot be guaranteed architecturally and must be learned.

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

All n_hidden > 64 fell into the dead-race false minimum (frac_crossed=0.00) at LR=1e-3. n_hidden=128 additionally diverged (loss=11, A=0.5, D=0.6 — collapsed readout). Root cause: LR=1e-3 was tuned for n_hidden=64. Larger models have more parameters and different loss landscape curvature. Key principle: model size increases may require LR adjustment in either direction. Always re-confirm LR after any n_hidden change. Proceeded conservatively: re-run LR search at n_hidden=100 (smallest increment) before jumping to 128.

---

### Sweep 11 — LR re-search at n_hidden=100 (smoke resolution only)

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

Viable LR threshold is between 2e-3 and 3e-3. LR=3e-3 and LR=5e-3 both achieve frac_crossed=1.00 and converge to identical results (score=0.132, A=0.850, t_rise=161ms, t_vortex=108ms, D=0.450) — confirmed different seeds, genuine shared attractor. LR=3e-3 locked as minimum viable for n_hidden=100 **at smoke resolution**. Score 0.132 vs 0.531 at n_hidden=64 — genuine capacity improvement. A jumped from 0.799 to 0.850, t_vortex hit 108ms (nearly on target). D=0.450 with vortex_depth=nan flagged as likely Gaussian fit artifact at 300 epochs.

LR=2e-3 curiosity: frac_crossed=0.00 but A=0.932 — model learned a high asymptote without the race ever crossing threshold. Classic false minimum.

**Important:** n_hidden=100 result here is smoke-resolution only (t_post=250, rpt_step=30). Full-resolution viability tested and failed in sweeps 14–15. n_hidden=100 is not the current operative baseline.

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

grad_clip=1.0 produced identical loss values to the unclipped sweep 11 run — library default is already 1.0. Spikes persist (loss oscillating 3.5–9.2) across all clip values while crossed=1.00 throughout. Gradient explosion ruled out: a true explosion would destabilize crossed, but it stays locked. Root cause of spikes: the behavioral summary-statistic loss is computed from a stochastic Monte Carlo tachometric curve — inherent epoch-to-epoch variance in the objective itself. batch_size confirmed at library default of 256.

Diagnosis: at the practical limit of smoke resolution. The shortened timeline (t_post=250 vs full 500) truncates the gradient on A and t_rise. The coarse bins (rpt_step=30 vs full 10) give ~6 bins across the rPT range instead of ~18, making t_vortex and D estimates unreliable.

**Note (added after sweep 18):** rpt_step affects the training objective, not just evaluation. It defines task.rpt_grid, which is used in losses.py when soft-binning trial outputs into the tachometric curve. Changing rpt_step changes what the model trains against.

---

### Run 13 — full resolution diagnostic (failed, n_hidden=100)

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

Dead race from epoch 10 onward. frac_crossed=0.00 for both m0 and m1, loss locked at 11.047 for 290 epochs. Score=4.13. The locked LR=3e-3 (tuned for n_hidden=100 at smoke resolution) does not transfer to full timeline — same principle as sweep 10. Changing resolution shifts the loss landscape. LR re-search required at full resolution.

---

### Sweep 14 — LR re-search at n_hidden=100, full resolution (failed)

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 \
    --set task.a_exo=3 \
    --set task.tau_exo=30 \
    --set model.n_hidden=100 \
    --set train.epochs=300 \
    --set task.t_pre=100 \
    --set task.t_post=500 \
    --set task.rpt_step=10 \
    --sweep train.lr=3e-3,5e-3,8e-3,1.2e-2 \
    --no-plots --top 4
```

LR=1.2e-2 is the only config that escapes the dead-race lock — crossed recovers to ~0.5 around epoch 190, but only for m1, never m0. Three lower LRs locked at false minimum from epoch 10. The viable LR window at full resolution is narrower than a single order of magnitude.

---

### Sweep 15 — higher LR range at n_hidden=100, full resolution (failed, abandoned)

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 \
    --set task.a_exo=3 \
    --set task.tau_exo=30 \
    --set model.n_hidden=100 \
    --set train.epochs=500 \
    --set task.t_pre=100 \
    --set task.t_post=500 \
    --set task.rpt_step=10 \
    --sweep train.lr=1.2e-2,1.5e-2,2e-2,3e-2 \
    --no-plots --top 4
```

LR=1.5e-2 through 3e-2 all dead from epoch 10, never escape. LR=1.2e-2 repeats sweep 14 behavior — m1 only, never m0. No viable config found. **n_hidden=100 abandoned at full resolution. Fell back to n_hidden=64.**

---

### Sweep 16 — LR search at n_hidden=64, full resolution

```bash
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
```

LR=1e-3: race establishes (crossed=0.78 by epoch 40), then slowly dies by epoch 210. LR=3e-3: peaks at crossed=0.94 by epoch 50, dies by epoch 120, partially recovers, collapses at termination. LR=5e-3: crossed=0.98 at epoch 20, crashes dead by epoch 50. LR=8e-3: crossed=1.00 from epoch 10, stays 0.55–1.00 all 300 epochs, but loss wanders 7.4–11.2 with no descending trend. Score=1.72 with m0/m1 asymmetry (frac_crossed_m0=0.08, frac_crossed_m1=1.00).

No single LR that establishes the race and then descends. Pattern: race finds a good basin early, then walks out regardless of LR. Two-stage training implemented as a result.

---

### Two-stage training — harness modification

Added `resume_checkpoint: str | None = None` to `TrainConfig` dataclass and a resume branch in `train.py` before the epoch loop:

```python
if train_cfg.resume_checkpoint is not None:
    ckpt = torch.load(train_cfg.resume_checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
```

Optimizer state deliberately not restored — fresh Adam at the new LR is the intent.

---

### Sweeps 17a/b/c — two-stage training at n_hidden=64, t_post=500 (failed)

**17a** — Stage 1: 100 epochs at LR=8e-3 to establish race.
```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 --set task.a_exo=3 --set task.tau_exo=30 \
    --set model.n_hidden=64 --set train.epochs=100 --set train.lr=8e-3 \
    --set task.t_pre=100 --set task.t_post=500 --set task.rpt_step=10 \
    --no-plots
# Checkpoint: results/opt/single_20260728_163950/model.pt
```

Race alive at termination (crossed ~0.5–1.0 throughout). Checkpoint saved.

**17b** — Stage 2: 200 epochs at LR=1e-3 resuming from 17a.
```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 --set task.a_exo=3 --set task.tau_exo=30 \
    --set model.n_hidden=64 --set train.epochs=200 --set train.lr=1e-3 \
    --set train.resume_checkpoint=results/opt/single_20260728_163950/model.pt \
    --set task.t_pre=100 --set task.t_post=500 --set task.rpt_step=10 \
    --no-plots
```

Race survived all 200 epochs (crossed=0.56–0.98). Two-stage loading confirmed working. But loss did not descend — wandered 7.3–11.0, same floor as stage 1. m0/m1 asymmetry persisted (frac_crossed_m0=0.33, frac_crossed_m1=1.00). Score=0.485. LR=1e-3 too low to navigate the basin.

**17c** — Stage 2 LR search: resume from same 17a checkpoint, sweep LR=2e-3,3e-3,5e-3.

LR=2e-3: loss drops to 4.52 at epoch 130 (best full-resolution loss seen), then race dies by epoch 199. LR=3e-3: drops to 6.27 by epoch 60, dies by epoch 120. LR=5e-3: drops to 4.51 at epoch 30(!), immediately destabilizes, never recovers. Consistent pattern across all: model finds a good basin early, then walks out. Two-stage approach delays but does not prevent collapse.

**Conclusion:** the instability at t_post=500 is not an LR problem. Something structural. Curriculum (warmup_epochs=10) ruled out — collapse happens 100+ epochs after curriculum stabilizes. STE gradient accumulation hypothesized.

---

### Sweep 18 — t_post isolation (STE hypothesis tested and ruled out)

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 --set task.a_exo=3 --set task.tau_exo=30 \
    --set model.n_hidden=64 --set train.epochs=300 --set train.lr=8e-3 \
    --set task.t_pre=100 --set task.rpt_step=10 \
    --sweep task.t_post=150,250,350,500 \
    --no-plots --top 4
```

Results by t_post:
- t_post=150: dead by epoch 60, loss descends to ~1.9 with crossed=0.00 — false minimum, model learns constant below-threshold output.
- t_post=250: crossed=1.00 both states all 300 epochs, score=0.132. Reproduces smoke result exactly.
- t_post=350: dead at epoch 40, loss collapses to ~1.0–1.3, A=0.9999 (constant readout output, no race). reg ticks to 0.00004 — weights growing while race dead.
- t_post=500: crossed 0.55–1.00 all 300 epochs, score=1.72. Race survives but loss doesn't descend.

STE hypothesis ruled out — the relationship between t_post and survival is non-monotone. A true gradient accumulation problem would produce monotone degradation with t_post.

**Key finding:** t_post=250 is the viable training window. t_post=500 is survivable but non-learning. t_post=150 and t_post=350 fall into false minima. The three-variable resolution change (t_pre, t_post, rpt_step) was a confound — the problem was always t_post specifically. t_post=250 at LR=8e-3/n_hidden=64 reproduces score=0.132 (same as smoke best at n_hidden=64). This is the operative baseline at end of this log.

**t_pre note:** t_pre controls how long the accumulator runs before cue onset. Monkeys fixated ~1000ms; t_pre=50 vs t_pre=100 matters only if the RNN hasn't reached its pre-cue attractor within 50ms. Likely fast convergence at n_hidden=64, so t_pre is low-priority.

**rpt_step note (added after sweep 18):** rpt_step affects the training objective, not just evaluation. It defines task.rpt_grid used in losses.py for soft-binning. Changing rpt_step changes what the model trains against.

---

### Architectural changes — code_v2_delta_edit_delta (implemented and validated)

Both architectural gaps identified at end of sweep 18 are now implemented and validated:

**Lapse mechanism** — branch mixture. Normal and lapse branches run in parallel; mixed with λ(m). Rule input channel zeroed on lapse branch. m-leakage test: max output diff across m on lapse branch = 0.0 (was 0.266 before fix). λ_young=0.08, λ_adult=0.02, both learned via sigmoid-constrained logits. Lapse branch contributes gradient even when normal branch doesn't cross threshold — primary reason t_post=500 may now be viable.

**Stochastic initial state** — shared+private factorization. σ_shared=0.7, σ_private=0.05 (set to contingency values after output correlation check). Provides trial-to-trial RT variability and more realistic vortex depth.

**Hard/soft curve separation** — confirmed correct. Soft mixed-branch curve drives per-step gradients; hard threshold-crossing curve used only for periodic summary-stat fitness. Separate code paths.

**Validation:** full forward pass with lapse+h0 active; all parameters receive gradients; lapse logits at correct initial values. Output correlation (0.230) and curve stability (50–80ms) at 50 epochs reflect untrained W_out geometry, not implementation errors — re-check after full training.

**Loss landscape impact:** both changes alter gradient flow. LR=8e-3 (the last operative value at n_hidden=64/t_post=250) must be re-confirmed.

---

### Original sweep 19 plan 
```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 \
    --set task.a_exo=3 \
    --set task.tau_exo=30 \
    --set model.n_hidden=64 \
    --set train.lr=8e-3 \
    --set train.epochs=300 \
    --set task.t_pre=100 \
    --set task.t_post=250 \
    --sweep task.rpt_step=30,10 \
    --no-plots --top 2
```

ran also 
--set train.lr=1e-3
---

### Sweep 19 result — post-architecture rPT-grid check

The planned `rpt_step=30,10` comparison was run twice at
`n_hidden=64`, `t_post=250`, and 300 epochs. The first run used `lr=8e-3`;
the confirmation used `lr=1e-3`. Both young and adult hard evaluation curves
had `frac_crossed=0.00` at both grid resolutions.

| learning rate | rpt_step | score | frac_crossed m0 / m1 |
|---|---:|---:|---|
| 8e-3 | 30 | 4.398 | 0.00 / 0.00 |
| 8e-3 | 10 | 4.814 | 0.00 / 0.00 |
| 1e-3 | 30 | 5.210 | 0.00 / 0.00 |
| 1e-3 | 10 | 5.670 | 0.00 / 0.00 |

**Decision:** this does not rank rPT resolutions. The new lapse and stochastic
initial-state machinery changed the Phase-0/1 landscape, so the next task is to
recover a live hard race at `rpt_step=30`, not to continue the planned LR or
timeline sweeps.

### Superseding trajectory

1. Hold `n_hidden=64`, `t_post=250`, `rpt_step=30`, and the fixed task knobs.
    Sweep `task.sigma_init_shared=0.3,0.5,0.7` by
    `task.sigma_init_private=0.0,0.05` at `lr=1e-3`. Rank hard
    `frac_crossed_*` before score.
2. With a recovered initial-state pair, re-search
    `train.lr=1e-3,3e-3,5e-3,8e-3`. Keep lapse endpoint initializers fixed.
3. Retry `rpt_step=30,10` only after both endpoint states cross healthily;
    this is a loss change, not an evaluation-only change.
4. Retest `t_post=500` with the recovered LR/grid pair. Sweep learned-lapse
    endpoint initializers and revisit capacity only after these gates pass.

sweep 20

Ran

python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 \
    --set task.a_exo=3 \
    --set task.tau_exo=30 \
    --set model.n_hidden=64 \
    --set train.lr=1e-3 \
    --set train.epochs=300 \
    --set task.t_pre=100 \
    --set task.t_post=250 \
    --set task.rpt_step=30 \
    --sweep task.sigma_init_shared=0.3,0.5,0.7 \
    --sweep task.sigma_init_private=0.0,0.05 \
    --no-plots --top 6
    
    
Res:
This isn't a sigma_init problem — sigma_init barely moved the score (4.6 → 5.6), and the ordering is noisy enough that it's mostly reflecting lapse/curve-fit noise, not a real signal from the parameter being swept.

A few things to pull from the results.csv that are meaningful despite the dead race:

m1 is producing real curves. Look at t_rise_m1 (81–141ms), t_vortex_m1 (60–141ms), D_m1 (0.07–0.19) — these are nonzero and in the right ballpark. frac_crossed_m1=0.00 but vortex_depth_m1 ≈ -0.39 everywhere, meaning the hard curve is dipping below chance. The hard-crossing counter is the problem, not the dynamics.
m0 is partially alive too — t_rise_m0 ranges 176–204ms, A_m0 reaches 0.93, vortex_depth_m0 ≈ -0.37. Real dynamics, just not crossing the threshold on enough trials to register.
The "dead race" classification (frac_crossed=0.00) is being triggered by the hard-threshold counter, but the model is clearly doing something — it's not the all-zero collapse you saw in sweeps 16/17.

What's actually going wrong: vortex_depth < 0 but frac_crossed = 0.00 is a signature of a model where the accumulator is active but the output readout isn't scaling enough to cross threshold=0.75 on enough trials at hard-eval time. The soft training loss is working (you're getting shaped curves), but the hard threshold crossing metric disagrees.

This points to one of two things:

threshold=0.75 is now too high for the post-architecture model — the lapse branch and h0 variability are competing with the signal that drives the accumulator past the threshold.
LR=1e-3 isn't pushing the readout weights hard enough to generate crossings, even though it shapes the soft curve.



Sweep 20 

Res:
This is expected: at θ=0.3 the race crosses too readily, so the exogenous capture doesn't have time to drag it down. The model is finding a valid regime but it's a different one from the target — fast threshold produces fast crossing that washes out the vortex timing. This is the same D vs t_vortex anti-correlation from the pre-architecture sweeps, now re-appearing here.

What this tells you about the root cause

The pre-architecture model trained fine at θ=0.75. Post-architecture, the viable threshold window has shifted dramatically downward. The lapse branch is the most likely explanation: with λ_young=0.08, ~8% of trials have no rule signal and the accumulator drifts randomly. At θ=0.75 those trials almost never cross, which pulls the hard-crossing metric below the penalty knee — but more importantly, the lapse branch also receives gradients, and those gradients pull the readout weights toward lower gain (since lower-gain outputs are more "correct" for lapse trials where no crossing should occur). This suppresses the normal branch's ability to cross.

Sweep 22 — fine threshold scan around the cliff

Before changing anything else, find where exactly the cliff is and whether there's a stable zone between 0.3 and 0.5.

bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.a_exo=3 --set task.tau_exo=30 \
    --set model.n_hidden=64 --set train.lr=1e-3 \
    --set train.epochs=300 --set task.t_pre=100 --set task.t_post=250 \
    --set task.rpt_step=30 \
    --set task.sigma_init_shared=0.3 --set task.sigma_init_private=0.05 \
    --sweep task.threshold=0.3,0.35,0.4,0.45 \
    --no-plots --top 4

If 0.35 or 0.4 survive, you may be able to find a threshold where the race is alive but the vortex timing isn't as washed out.


Res:
Sweep 22 diagnosis

The cliff is now precisely mapped:

threshold	frac_crossed m0 / m1	score	verdict
0.30	0.55 / 0.65	1.223	alive, timing badly late
0.35	0.51 / 0.58	0.839	alive, better
0.40	0.46 / 0.50	0.587	winner — best score, vortex timing on target
0.45	0.09 / 0.01	4.486	dying by epoch 130, dead by 250

The viable zone is 0.30–0.40, with a hard cliff between 0.40 and 0.45. And critically, θ=0.40 wins on score, not just on survival — the behavioral fit is genuinely better than 0.30 or 0.35.

What the θ=0.40 metrics say:

t_vortex_m0 = 115.6ms (target 105ms) — nearly on target, vs 192ms at θ=0.30
A_m0 = 0.908 (target 0.92) — very close
D_m0 = 0.141 (target 0.28) — shallow, but vortex_depth_m0 = −0.19 so the curve dips real, Gaussian fit is probably struggling at smoke resolution (same diagnosis as pre-architecture sweeps)
frac_crossed m0/m1 = 0.46 / 0.50 — right at the 0.4 penalty knee, fragile
m1 collapse: D_m1 ≈ 0, t_rise_m1 = 163ms — adult state isn't fitting well

The one concern: frac_crossed at 0.46/0.50 is barely above the penalty threshold. The race is alive but marginal. At 300 epochs it's holding, but this is likely to drift if you push anything else (larger t_post, higher LR). θ=0.40 is the right working point but it needs LR optimization to stabilize it.

m1 pattern: Both θ=0.35 and θ=0.40 show D_m1 ≈ 0 (or near-zero) and the t_rise_m1 for θ=0.35 collapses to 80ms. This is the same m1 weakness seen pre-architecture. At smoke resolution the adult state is systematically harder to fit. Not a blocker now — it's a capacity/training-length issue, not a dead race.

What to do next

θ=0.40 is the new working baseline. The architecture + sigma_init=(0.3, 0.05) is Phase-0 healthy at this threshold. Now do what the plan called for after recovery: re-search LR.

Sweep 23 — LR search at θ=0.40
bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.a_exo=3 --set task.tau_exo=30 \
    --set model.n_hidden=64 \
    --set train.epochs=300 --set task.t_pre=100 --set task.t_post=250 \
    --set task.rpt_step=30 \
    --set task.sigma_init_shared=0.3 --set task.sigma_init_private=0.05 \
    --set task.threshold=0.40 \
    --sweep train.lr=5e-4,1e-3,3e-3,5e-3 \
    --no-plots --top 4

The current LR=1e-3 has frac_crossed just barely above the gate. A better LR may either (a) push frac_crossed higher for the same threshold, giving headroom, or (b) improve the score by fitting m1's D and t_rise. Watch frac_crossed first — if a higher LR pushes it above 0.6 without killing it, that's the winner. The pre-architecture pattern was that LRs above 3e-3 caused collapse at t_post=250, but the loss landscape is different now, so test the range fresh.

After LR is locked, the sequence from the plan resumes: rpt_step, then t_post=500 retest, then lapse endpoint sweep, then capacity.

One thing worth noting for the opt_progress log: the post-architecture viable threshold (0.40) is meaningfully lower than the pre-architecture optimum (0.75). This needs to be tracked as a scientific consideration, not just an engineering one — if the model that fits the behavioral data requires θ=0.40 rather than 0.75, that's a statement about what accumulation-to-bound value the lapse architecture implies. Worth flagging for the biological/mechanistic tuning phase (OPT_README §9).

Q about threshold- doesn't 0.4 undermine us scientifically given the 0.75, well-established precedent?

Claude:
What the threshold actually represents

In your architecture, θ is the readout threshold — the value that the output unit's activation must cross for a commitment to be detected. This is not directly the biological decision threshold in the accumulator-to-bound sense. It's a parameter of the detection rule applied to the model's output, not a parameter of the underlying recurrent dynamics.

The biological claim in Mastrogiuseppe/Ostojic-style LR-RNNs is that the recurrent dynamics implement the race/accumulation, and the readout just reads out the result. So θ in your code is more like "how large does the output projection need to be before we call it a commitment" — which is partly a function of the readout weight scale, the recurrent gain, and the initialization, not purely a statement about the animal's decision criterion.

This means θ=0.40 vs θ=0.75 is not necessarily a scientific claim about what threshold the monkey uses. It may just reflect that your readout weights are currently initialized/scaled such that the output activations live in a lower range, and θ needs to match that range for the crossing detector to work.

The question to ask is: does the vortex appear at the right rPT, at the right depth, with the right asymptote — and does the SI signal come out correctly — regardless of what numerical value of θ you needed to use to get there? If yes, θ is an internal engineering parameter, not a scientific claim.

When θ would be scientifically non-negotiable

θ would be a hard scientific constraint if your model is explicitly implementing a drift-diffusion / race-to-bound where θ is the bound height, and the bound height has a known empirical value from the physiology. From the Zhu et al. 2024 paper and the Salinas/Stanford circuit model: the relevant physiological quantities are the timing of commitment (t_rise, t_vortex) and the shape of the tachometric curve, not the raw activation scale. The FEF/dlPFC SI signal that you're trying to predict post-hoc is a spatial selectivity index, which is also a relative/normalized quantity.

Sweep 23 diagnosis

Clear result. The LR landscape at θ=0.40 is:

LR	frac_crossed m0/m1	score	verdict
5e-4	0.52 / 0.85	0.831	alive, m1 excellent, m0 timing late
1e-3	0.46 / 0.50	0.587	winner on score
3e-3	0.00 / 0.00	4.327	dead by epoch 60
5e-3	0.00 / 0.00	5.401	dead by epoch 90

The viable LR window is narrow: only 5e-4 and 1e-3 survive. 3e-3 kills the race exactly as it did pre-architecture at this t_post — the lapse branch didn't open up the upper end of the LR range.

But the interesting comparison is 5e-4 vs 1e-3:

LR=5e-4 doesn't win on score (0.831 vs 0.587) but it has some genuinely better individual metrics:

frac_crossed_m1 = 0.85 vs 0.50 — the adult race is much more robust
t_rise_m1 = 96ms (target 140ms) — undershooting but the right order of magnitude, vs 163ms at 1e-3
t_vortex_m1 = 91ms (target 106ms) — nearly on target
D_m1 = 0.061 — still weak but nonzero
final_loss = 2.05 vs 2.60 — meaningfully lower, model is actually learning more

LR=1e-3 wins on score primarily because m0 metrics happen to be better, but its m1 state is essentially degenerate (D_m1 ≈ 0, frac_crossed_m1 = 0.50). The score function weights the two maturation states together, so a model that does well on one and poorly on the other can outscore a model that does moderately well on both.

The real signal here: 5e-4 is finding a better basin (lower loss, m1 healthier) but needs more epochs to close the gap on m0. LR=1e-3 is more aggressive but leaving m1 behind.

Sweep 23b — fine LR between 5e-4 and 1e-3

There may be a sweet spot that gives better m1 robustness than 1e-3 while converging faster than 5e-4:

python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.a_exo=3 --set task.tau_exo=30 \
    --set model.n_hidden=64 \
    --set train.epochs=300 \
    --set task.t_pre=100 --set task.t_post=250 \
    --set task.rpt_step=30 \
    --set task.sigma_init_shared=0.3 --set task.sigma_init_private=0.05 \
    --set task.threshold=0.40 \
    --sweep train.lr=5e-4,7e-4,1e-3 \
    --no-plots --top 3

    just ran .0007 by itself.

also ran a .0001. No long epoch test yet. where to


Let me parse what you have: two single runs — LR=7e-4 and LR=1e-4.

LR=7e-4 (first metrics.json/log):

score=0.938, loss=2.177
m0: t_vortex=115ms ✓, A=0.988, frac_crossed=0.50, D=0.036
m1: t_vortex=177ms (badly late), A=1.0, frac_crossed=0.72, D=0.035
crossed hovers 0.55–0.80 all 300 epochs, never collapses

LR=1e-4 (second metrics.json/log):

score=2.637, loss=1.794 (lowest loss yet)
m0: t_vortex=138ms, A=0.982, frac_crossed=0.54, D≈0
m1: t_vortex=108ms ✓✓ (nearly perfect), A=0.607 (collapsed), frac_crossed=0.97, D=0.45, t_rise=NaN
crossed very stable 0.67–0.84 throughout — the most stable race you've seen
What these two runs tell you together

The picture across all five tested LRs is now:

LR	score	loss	m0 t_vortex	m1 t_vortex	frac_crossed m0/m1	stability
1e-4	2.64	1.79	138ms	108ms ✓	0.54/0.97	excellent
5e-4	0.83	2.05	175ms	91ms	0.52/0.85	good
7e-4	0.94	2.18	115ms ✓	177ms	0.50/0.72	good
1e-3	0.59	2.60	115ms ✓	163ms	0.46/0.50	marginal
3e-3	dead	—	—	—	0/0	—

There's a clean inversion: as LR decreases, m1 gets better and m0 gets worse. LR=1e-4 has near-perfect m1 t_vortex but m1 A has collapsed (0.607) and m0 timing is late. LR=1e-3 has good m0 but m1 is marginal. No single LR fits both states well at 300 epochs.

This is a capacity/epoch problem, not a hyperparameter problem. The two maturation states are pulling the model in different directions at 300 epochs of smoke resolution. The model has enough capacity to fit one well at a time, but 300 epochs isn't long enough for it to settle both simultaneously. LR just controls which state gets priority first.

The low loss at LR=1e-4 (1.794 vs 2.60 at 1e-3) with a still-alive race and stable crossings is the key signal — this LR is genuinely learning more, it's just not done yet.

What to do

Skip the fine LR search and go straight to epoch extension. You have enough information.

Sweep 24 — epoch extension at 7e-4 and 1e-4
bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.a_exo=3 --set task.tau_exo=30 \
    --set model.n_hidden=64 \
    --set task.t_pre=100 --set task.t_post=250 \
    --set task.rpt_step=30 \
    --set task.sigma_init_shared=0.3 --set task.sigma_init_private=0.05 \
    --set task.threshold=0.40 \
    --sweep train.lr=7e-4,1e-4 \
    --sweep train.epochs=600,1000 \
    --no-plots --top 4

NOW:
PIVOT N=100

This is a pivotal result. Let me lay out the full picture.

Sweep 24 diagnosis
config	score	loss	m0 t_vortex	m0 frac_crossed	m1 t_vortex	m1 frac_crossed	m1 A
1e-4 / 600ep	0.904	1.106	63ms	0.54	124ms	0.977	1.000
7e-4 / 1000ep	1.463	1.635	115ms ✓	0.50	198ms	0.698	1.000
7e-4 / 600ep	2.021	1.341	61ms	0.50	195ms	0.701	1.000
1e-4 / 1000ep	2.841	1.359	168ms	0.54	117ms	0.968	0.739

The headline finding: LR=1e-4 at 600 epochs wins on score (0.904) with loss=1.106 — the lowest loss you've seen by a wide margin. m1 frac_crossed=0.977 is exceptional. But the m0 t_vortex (63ms) and m1 t_vortex (124ms) are both off target (105/106ms).

The troubling pattern: the LR inversion is now sharper and cleaner than before, and it's not resolving with more epochs:

LR=7e-4 at 1000 epochs has excellent m0 t_vortex (115ms ✓) but m1 t_vortex is getting worse with more epochs (195→198ms), not better. The m0/m1 split is widening, not closing.
LR=1e-4 at 1000 epochs: m0 t_vortex regresses to 168ms (was 63ms at 600), m1 A collapses to 0.739, t_rise_m1=NaN. The model is un-learning something past 600 epochs. This is instability, not slow convergence.

LR=1e-4 at 1000 epochs is worse than at 600 epochs on nearly every metric. That's the clearest signal in the data: there's an optimum around 600 epochs for this LR, and it degrades past that. The model is cycling or drifting in a flat basin rather than converging.

The core problem that won't go away: at every LR and epoch count tried, t_vortex for m0 and m1 refuse to be simultaneously near target. When m0 t_vortex ≈ 115ms (correct), m1 t_vortex ≈ 195–200ms (nearly 2x too late). When m1 t_vortex ≈ 108–124ms (closer), m0 t_vortex ≈ 60–80ms (too early). This is not a training convergence problem — it's a capacity/architecture constraint.

What this means

You've hit the n_hidden=64 capacity ceiling, but in a specific way: the model can't simultaneously fit the temporal dynamics of both maturation states. The two states require different t_vortex timings (both ~105–106ms) but the model's low-rank recurrent dynamics are finding solutions where one state's timing is correct and the other's is wrong, and no amount of LR tuning or epoch extension resolves this.

The D=0 / near-zero issue across nearly all configs is a related symptom — at this capacity the model can't also learn the correct vortex depth while getting timing right.

This is Phase 2 territory: you need more representational capacity. The scale-up plan from OPT_README §8 says n_hidden=64→higher is the next step after confirming a capacity ceiling. You've now confirmed it.

What to do next

The best checkpoint you have going forward is LR=1e-4, 600 epochs (score=0.904, loss=1.106). Lock that as the operative baseline.

Sweep 25 — n_hidden capacity at the new baseline
bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.a_exo=3 --set task.tau_exo=30 \
    --set task.t_pre=100 --set task.t_post=250 \
    --set task.rpt_step=30 \
    --set task.sigma_init_shared=0.3 --set task.sigma_init_private=0.05 \
    --set task.threshold=0.40 \
    --set train.lr=1e-4 \
    --set train.epochs=600 \
    --sweep model.n_hidden=64,100,128 \
    --no-plots --top 3

Pre-architecture, n_hidden=100 needed a higher LR (3e-3) to establish the race. Post-architecture, LR=1e-4 is a very different landscape — the lapse branch provides gradient even when the normal branch isn't crossing, which may make larger models viable at lower LR. Watch frac_crossed closely: if n_hidden=100/128 collapse to 0.00 with LR=1e-4, you'll need a brief LR re-check at the new size before concluding capacity helps. But given the lapse gradient path, there's genuine reason to expect larger models to be more tractable now than they were pre-architecture.

The key question: does n_hidden=100 or 128 allow both m0 and m1 t_vortex to land near 105ms simultaneously? If yes, that's your Phase 2 breakthrough. If both states still split, you have a deeper architectural constraint and the mechanistic interpretation becomes the next thing to interrogate.