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

### Original sweep 19 plan — superseded by the result below

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

**If stable:** lock rpt_step=10. **If not:** keep rpt_step=30 for training; rpt_step=10 for eval only. LR=8e-3 here is the last operative value — result is directionally useful but LR must be re-confirmed in sweep 20 regardless.

---

### Sweep 20 — LR re-search at n_hidden=64/t_post=250 with new architecture (pending)

Run after sweep 19.

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 \
    --set task.a_exo=3 \
    --set task.tau_exo=30 \
    --set model.n_hidden=64 \
    --set train.epochs=300 \
    --set task.t_pre=100 \
    --set task.t_post=250 \
    --set task.rpt_step=[from sweep 19] \
    --sweep train.lr=1e-3,3e-3,5e-3,8e-3 \
    --no-plots --top 4
```

Pre-architecture, t_post=250/n_hidden=64 used LR=1e-3 (smoke, sweeps 7–8) and LR=8e-3 (full resolution, sweep 18). The lapse branch and stochastic h0 both change the loss landscape. Sweep the full range. Watch frac_crossed as primary health metric. If all dead: try two-stage approach.

---

### Sweep 21 — t_post=500 retest at n_hidden=64 with new architecture (pending)

Run after sweep 20 confirms a viable LR.

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 \
    --set task.a_exo=3 \
    --set task.tau_exo=30 \
    --set model.n_hidden=64 \
    --set train.epochs=300 \
    --set task.t_pre=100 \
    --set task.rpt_step=[from sweep 19] \
    --set train.lr=[from sweep 20] \
    --set task.t_post=500 \
    --no-plots
```

The lapse branch provides gradient even when the normal branch doesn't cross threshold — this is the structural change most likely to fix the t_post=500 non-learning problem seen in sweeps 16–17. If race establishes and loss descends: t_post=500 is viable. If not: the t_post failure is not architectural — investigate whether the rPT distribution at t_post=500 overweights uninformative guessing-tail trials.

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
