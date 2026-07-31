# opt_progress.md — Behavior-Fit Optimization Log

Live results + decision log for the optimization harness. Reference doc (modes, artifacts, sweep philosophy, scale-up) is [OPT_README.md](OPT_README.md).

Append one dated entry per accepted finding. Record the command, the key numbers, the interpretation, and the resulting decision. Keep the current best config and next steps sections at the top up to date.

**Ghost-point protocol:** when a sweep is proposed, any config that duplicates a previously-run point is listed as a **ghost point** — noted for context in the analysis but not re-run. Ghost points are marked inline: `// ghost: sweep N — <key metrics>`.

---

## Current best config (living)

See updated table at bottom of document under "Current best config (updated)". Best run: Run 34 (2026-07-30 last run).

**Current phase:** Phase 2/3 boundary. Gap stratification (Run 34) proved vortex gradient signal is the lever for D/t_vortex but degrades frac_crossed over long training. 

---

## Next steps (specific, ordered)

See updated next steps at bottom of document.

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

Results.csv target columns in sweep 1 showed t_vortex≈93.9 (young) and D≈0.205 — meaningfully off. Fix applied before sweep 2. All sweeps from sweep 2 onward use correct targets. Sweep 1 rankings directionally useful but not trusted for final comparison.

---

### Sweep 1 — θ grid, smoke, wrong targets

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --sweep task.threshold=0.2,0.3,0.5,0.7 --no-plots
```

θ=0.2 won (score 0.250) only because wrong D target made its D=0.45 look acceptable. θ=0.3 and θ=0.2 produced nan t_rise for m=0 — rpt_step=30 too coarse. θ=0.7 scored second (0.383) with frac_crossed≈0.50. Rankings invalidated by wrong targets; re-run as sweep 2.

---

### Sweep 2 — θ grid, smoke, corrected targets

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --sweep task.threshold=0.2,0.3,0.5,0.7 --no-plots
```

Target fix reshuffled rankings. θ=0.7 wins (score 1.522) — only config to hit D≈0.28. t_rise=230ms and t_vortex=200ms badly late. **θ=0.7 adopted as working baseline.**

---

### Sweep 3 — θ × a_exo grid

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --sweep task.threshold=0.6,0.65,0.7,0.75,0.8 \
    --sweep task.a_exo=3,5 \
    --no-plots --top 5
```

θ=0.75, a_exo=3 wins (score 0.297), t_vortex=106ms nearly on target, but D≈0. θ=0.7, a_exo=3 holds D≈0.28 but t_vortex=200ms. a_exo=5 universally worse. D and t_vortex anti-correlated across θ at fixed a_exo — structural. **a_exo locked at 3.0.**

---

### Sweep 4 — θ × τ_exo grid

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.a_exo=3 \
    --sweep task.threshold=0.7,0.72,0.75,0.78 \
    --sweep task.tau_exo=10,20,30 \
    --no-plots --top 6
```

τ_exo=10 catastrophic (kills frac_crossed). τ_exo=20 marginal. τ_exo=30 (library default) correct. D vs t_vortex anti-correlation not fixable with τ_exo. **τ_exo locked at 30.** Winner: θ=0.75/τ_exo=30 (score 0.297).

**D=0 diagnosis:** D is the Gaussian vortex amplitude from a parametric fit to binned tachometric curves. If the dip is too narrow or bins too sparse, `curve_fit` returns D≈0 even when `vortex_depth` (raw minimum) is genuinely below chance. This is a measurement artifact at smoke resolution. **Always read vortex_depth alongside D.**

---

### Sweep 5 — epoch count, vortex emergence check

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 --set task.a_exo=3 --set task.tau_exo=30 \
    --sweep train.epochs=50,150,300,500 \
    --no-plots --top 4
```

vortex_depth < 0 at all epoch counts — vortex mechanism works from the start. D≈0 at 50/500 epochs is Gaussian fit failure, not absent mechanism. Best fit at 300 epochs. 500 epochs: t_vortex drifts to 70ms — instability past an optimum. Also confirmed: a_exo is additive; below-chance dip must be learned via recurrent weights. **Smoke epoch baseline upgraded to 300.**

---

### Sweep 6 — θ × a_exo at 300 epochs (honest ranking)

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set train.epochs=300 --set task.tau_exo=30 \
    --sweep task.threshold=0.7,0.72,0.75,0.78 \
    --sweep task.a_exo=3,5,8 \
    --no-plots --top 6
```

θ=0.75, a_exo=3 holds as winner (score 0.531). t_vortex converged to ~115–118ms across viable configs — τ_exo=30 is the dominant timing controller. a_exo=8 eliminated. Phase 1 gate reached.

---

### Sweep 7 — LR search, Phase 1

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 --set task.a_exo=3 --set task.tau_exo=30 \
    --set train.epochs=300 \
    --sweep train.lr=1e-4,3e-4,1e-3,3e-3 \
    --no-plots --top 4
```

LR=1e-3 wins (score 0.531). LR=3e-3: best A (0.835) and D (0.362) but overshoots t_rise. LR≤3e-4: dead (frac_crossed=0.00) with artificially low loss — textbook false minimum. Never trust raw training loss alone.

---

### Sweep 8 — fine LR search

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 --set task.a_exo=3 --set task.tau_exo=30 \
    --set train.epochs=300 \
    --sweep train.lr=1e-3,1.5e-3,2e-3,3e-3 \
    --no-plots --top 4
```

No clean optimum in the gap — LR=2e-3 lands in a different basin (t_rise=229ms). **LR=1e-3 locked for n_hidden=64, pre-architecture.** LR tuning is model-size dependent; re-confirm after any n_hidden change.

---

### Sweep 9 — epoch scale-up at locked config

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 --set task.a_exo=3 --set task.tau_exo=30 \
    --set train.lr=1e-3 \
    --sweep train.epochs=300,500,750,1000 \
    --no-plots --top 4
```

A m0 stuck 0.769–0.799 across all epoch counts. **Capacity ceiling at n_hidden=64 confirmed (pre-architecture).** 500-epoch instability: t_vortex crashes to 70ms (same as sweep 5). D m1 degrades with more epochs. Phase 2 signal: raise n_hidden.

---

### Sweep 10 — n_hidden capacity sweep (pre-architecture)

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 --set task.a_exo=3 --set task.tau_exo=30 \
    --set train.lr=1e-3 --set train.epochs=300 \
    --sweep model.n_hidden=64,100,128,200 \
    --no-plots --top 4
```

All n_hidden > 64 dead (frac_crossed=0.00) at LR=1e-3. n_hidden=128 additionally diverged. **Key principle: always re-confirm LR after any n_hidden change.** Proceeded to LR re-search at n_hidden=100.

---

### Sweep 11 — LR re-search at n_hidden=100 (smoke resolution only)

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 --set task.a_exo=3 --set task.tau_exo=30 \
    --set train.epochs=300 --set model.n_hidden=100 \
    --sweep train.lr=1e-3,2e-3,3e-3,5e-3 \
    --no-plots --top 4
```

LR=3e-3 and LR=5e-3 both viable (frac_crossed=1.00), converge to identical results (score=0.132, A=0.850, t_vortex=108ms) — confirmed different seeds, genuine shared attractor. LR=3e-3 locked as minimum viable for n_hidden=100 **at smoke resolution only**. Score 0.132 vs 0.531 at n_hidden=64 — genuine capacity improvement. LR=2e-3: frac_crossed=0.00 but A=0.932 — classic false minimum.

**Important:** n_hidden=100 here is smoke-resolution only. Full-resolution viability tested and failed in sweeps 14–15.

---

### Sweep 12 — grad_clip sweep (ruled out)

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 --set task.a_exo=3 --set task.tau_exo=30 \
    --set model.n_hidden=100 --set train.lr=3e-3 --set train.epochs=300 \
    --sweep train.grad_clip=0.5,1.0,2.0,5.0 \
    --no-plots --top 4
```

Library default already 1.0 — sweeping changed nothing. Loss spikes (3.5–9.2) persist across all clip values while crossed=1.00 throughout. Gradient explosion ruled out. Root cause: stochastic Monte Carlo behavioral objective inherently noisy. batch_size confirmed at library default 256. At practical limit of smoke resolution.

---

### Run 13 — full resolution diagnostic, n_hidden=100 (failed)

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 --set task.a_exo=3 --set task.tau_exo=30 \
    --set model.n_hidden=100 --set train.lr=3e-3 --set train.epochs=300 \
    --set task.t_pre=100 --set task.t_post=500 --set task.rpt_step=10 \
    --no-plots
```

Dead race from epoch 10. frac_crossed=0.00 both states, loss locked at 11.047, score=4.13. LR=3e-3 (tuned at smoke resolution) does not transfer to full timeline.

---

### Sweep 14 — LR re-search at n_hidden=100, full resolution (failed)

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 --set task.a_exo=3 --set task.tau_exo=30 \
    --set model.n_hidden=100 --set train.epochs=300 \
    --set task.t_pre=100 --set task.t_post=500 --set task.rpt_step=10 \
    --sweep train.lr=3e-3,5e-3,8e-3,1.2e-2 \
    --no-plots --top 4
```

LR=1.2e-2 only partial escape — crossed recovers ~0.5 around epoch 190 for m1 only, never m0. Three lower LRs dead from epoch 10. No viable config at n_hidden=100/t_post=500.

---

### Sweep 15 — higher LR range at n_hidden=100, full resolution (failed, abandoned)

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 --set task.a_exo=3 --set task.tau_exo=30 \
    --set model.n_hidden=100 --set train.epochs=500 \
    --set task.t_pre=100 --set task.t_post=500 --set task.rpt_step=10 \
    --sweep train.lr=1.2e-2,1.5e-2,2e-2,3e-2 \
    --no-plots --top 4
```

All dead or m1-only. **n_hidden=100 abandoned at full resolution. Fell back to n_hidden=64.**

---

### Sweep 16 — LR search at n_hidden=64, full resolution

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 --set task.a_exo=3 --set task.tau_exo=30 \
    --set model.n_hidden=64 --set train.epochs=300 \
    --set task.t_pre=100 --set task.t_post=500 --set task.rpt_step=10 \
    --sweep train.lr=1e-3,3e-3,5e-3,8e-3 \
    --no-plots --top 4
```

LR=8e-3 only survivor (crossed=0.55–1.00 all 300 epochs) but loss wanders 7.4–11.2 with no descent. All lower LRs: race establishes then dies. Pattern: race finds a good basin early, then walks out regardless of LR. Two-stage training implemented.

---

### Two-stage training — harness modification

Added `resume_checkpoint: str | None = None` to `TrainConfig`. Loads `state_dict` only — fresh Adam at new LR.

---

### Sweeps 17a/b/c — two-stage training at n_hidden=64, t_post=500 (failed)

**17a** — Stage 1: 100 epochs at LR=8e-3.
```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 --set task.a_exo=3 --set task.tau_exo=30 \
    --set model.n_hidden=64 --set train.epochs=100 --set train.lr=8e-3 \
    --set task.t_pre=100 --set task.t_post=500 --set task.rpt_step=10 \
    --no-plots
# Checkpoint: results/opt/single_20260728_163950/model.pt
```

Race alive at termination.

**17b** — Stage 2: 200 epochs at LR=1e-3 from 17a checkpoint. Race survived but loss did not descend (7.3–11.0). LR=1e-3 too low to navigate the basin.

**17c** — Stage 2 LR search: LR=2e-3,3e-3,5e-3 resuming from 17a. All find a good basin briefly then walk out. LR=2e-3 best (loss 4.52 at epoch 130), dead by epoch 199. Two-stage delays collapse but does not prevent it. Not an LR problem — structural. Warmup ruled out. STE gradient accumulation hypothesized.

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

| t_post | outcome |
|---|---|
| 150 | dead by epoch 60, false minimum (loss ~1.9, crossed=0.00) |
| **250** | **crossed=1.00 both states all 300 epochs, score=0.132** |
| 350 | dead at epoch 40, A=0.9999 (constant readout), no race |
| 500 | crossed 0.55–1.00 all 300 epochs, score=1.72 — survives, loss never descends |

STE hypothesis ruled out — non-monotone relationship between t_post and survival. **Key finding: t_post=250 is the viable training window.** The three-variable resolution change in earlier sweeps was a confound — the problem was always t_post specifically.

**t_pre note:** pre-cue attractor window. Low priority — matters only if RNN hasn't converged within 50ms. Set explicitly to 100 since this sweep.

**rpt_step note:** affects the soft-binning training objective (task.rpt_grid in losses.py), not just evaluation. Treat any rpt_step change as a training-loss change; re-confirm crossing and LR stability after.

---

### Architectural changes — code_v2_delta_edit_delta (implemented and validated, post-sweep-18)

Both architectural gaps identified at end of sweep 18 now implemented:

**Lapse mechanism** — branch mixture. Rule input (and maturation input) zeroed on lapse branch. m-leakage test passed: max output diff across m on lapse branch = 0.0 (was 0.266). λ_young=0.08, λ_adult=0.02, both learned via sigmoid-constrained logits. Lapse branch contributes gradient even when normal branch doesn't cross threshold.

**Stochastic initial state** — shared+private factorization. σ_shared=0.7, σ_private=0.05 (contingency values, updated to 0.3/0.05 in sweep 20).

**Hard/soft curve separation** — confirmed. Soft mixed-branch curve drives gradients; hard threshold-crossing curve used for periodic fitness only.

**Validation:** all parameters receive gradients; lapse logits at correct initial values (0.08/0.02). Output correlation (0.230) and curve stability at 50 epochs reflect untrained W_out geometry, not implementation errors.

**Loss landscape impact:** both changes alter gradient flow. All pre-architecture LR findings must be re-confirmed.

---

## 2026-07-29

### Sweep 19 — post-architecture rPT-grid check (Phase-0 failure)

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 --set task.a_exo=3 --set task.tau_exo=30 \
    --set model.n_hidden=64 --set train.epochs=300 \
    --set task.t_pre=100 --set task.t_post=250 \
    --sweep task.rpt_step=30,10 \
    --no-plots --top 2
# Run twice: once at lr=8e-3, once at lr=1e-3
```

| lr | rpt_step | score | frac_crossed m0 / m1 |
|---|---:|---:|---|
| 8e-3 | 30 | 4.398 | 0.00 / 0.00 |
| 8e-3 | 10 | 4.814 | 0.00 / 0.00 |
| 1e-3 | 30 | 5.210 | 0.00 / 0.00 |
| 1e-3 | 10 | 5.670 | 0.00 / 0.00 |

All frac_crossed=0.00. This does not rank rPT resolutions — it is a Phase-0/1 recovery failure. vortex_depth genuinely negative across configs despite frac_crossed=0.00 — dynamics active, readout not reaching threshold at θ=0.75 post-architecture. rpt_step comparison deferred; crossing recovery is the immediate task.

---

### Sweep 20 — sigma_init crossing recovery

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 --set task.a_exo=3 --set task.tau_exo=30 \
    --set model.n_hidden=64 --set train.lr=1e-3 --set train.epochs=300 \
    --set task.t_pre=100 --set task.t_post=250 --set task.rpt_step=30 \
    --sweep task.sigma_init_shared=0.3,0.5,0.7 \
    --sweep task.sigma_init_private=0.0,0.05 \
    --no-plots --top 6
```

| sigma_shared / sigma_private | score | frac_crossed m0/m1 |
|---|---|---|
| **0.3 / 0.05** | **4.613** | **0.00 / 0.00** |
| 0.5 / 0.00 | 5.087 | 0.00 / 0.00 |
| 0.7 / 0.00 | 5.149 | 0.00 / 0.00 |
| 0.7 / 0.05 | 5.210 | 0.00 / 0.00 |
| 0.3 / 0.00 | 5.244 | 0.00 / 0.00 |
| 0.5 / 0.05 | 5.616 | 0.00 / 0.00 |

Still all dead at θ=0.75. sigma_init barely moved the score — this is not a sigma_init problem. vortex_depth genuinely negative across all configs. Root cause: lapse branch gradient suppresses readout gain, making θ=0.75 structurally unreachable. Threshold must be lowered. **sigma_init=(0.3, 0.05) locked as best performer.**

---

### Sweep 21 — threshold cliff mapping

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.a_exo=3 --set task.tau_exo=30 \
    --set model.n_hidden=64 --set train.lr=1e-3 --set train.epochs=300 \
    --set task.t_pre=100 --set task.t_post=250 --set task.rpt_step=30 \
    --set task.sigma_init_shared=0.3 --set task.sigma_init_private=0.05 \
    --sweep task.threshold=0.3,0.5,0.6,0.75 \
    --no-plots --top 4
```

| threshold | frac_crossed m0/m1 | score |
|---|---|---|
| **0.3** | **0.55 / 0.65** | **1.223** |
| 0.5 | 0.00 / 0.00 | 4.327 |
| 0.6 | 0.00 / 0.00 | 5.305 |
| 0.75 | 0.00 / 0.00 | 4.613 |

Hard cliff between 0.3 and 0.5. Only θ=0.3 survives. Root cause: with λ_young=0.08, lapse trials pull readout weights toward lower gain; at θ≥0.5 the normal branch can't overcome this suppression at current scale. Phase-0 gate re-established at θ=0.3. Fine cliff scan next.

**Threshold scientific note:** θ in this architecture is a detection threshold on output activations, not the biological accumulator bound. The biological quantities of interest are t_rise, t_vortex, A, D, and the SI signal shape — all relative/normalized quantities. θ=0.40 vs θ=0.75 is an engineering parameter reflecting readout weight scale and lapse gradient interaction, not a statement about the monkey's decision criterion. Once a working config is found, validate on mechanistic diagnostics (mode activations, SI prediction, intermediate-m sweep) rather than θ value.

---

### Sweep 22 — fine threshold cliff scan

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.a_exo=3 --set task.tau_exo=30 \
    --set model.n_hidden=64 --set train.lr=1e-3 --set train.epochs=300 \
    --set task.t_pre=100 --set task.t_post=250 --set task.rpt_step=30 \
    --set task.sigma_init_shared=0.3 --set task.sigma_init_private=0.05 \
    --sweep task.threshold=0.3,0.35,0.4,0.45 \
    --no-plots --top 4
```

| threshold | frac_crossed m0/m1 | score | verdict |
|---|---|---|---|
| 0.30 | 0.55 / 0.65 | 1.223 | alive, timing badly late |
| 0.35 | 0.51 / 0.58 | 0.839 | alive, better |
| **0.40** | **0.46 / 0.50** | **0.587** | **winner — best score, vortex timing near target** |
| 0.45 | 0.09 / 0.01 | 4.486 | dying by epoch 130, dead by 250 |

Hard cliff between 0.40 and 0.45. θ=0.40 wins on score, not just survival: t_vortex_m0=115.6ms (target 105ms), A_m0=0.908 (target 0.92). frac_crossed at 0.46/0.50 — barely above the 0.4 penalty knee, fragile. m1 still weak (D_m1≈0, t_rise_m1=163ms). **θ=0.40 adopted as post-architecture baseline.**

---

### Sweep 23 — LR search at θ=0.40

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.a_exo=3 --set task.tau_exo=30 \
    --set model.n_hidden=64 --set train.epochs=300 \
    --set task.t_pre=100 --set task.t_post=250 --set task.rpt_step=30 \
    --set task.sigma_init_shared=0.3 --set task.sigma_init_private=0.05 \
    --set task.threshold=0.40 \
    --sweep train.lr=5e-4,1e-3,3e-3,5e-3 \
    --no-plots --top 4
```

| LR | frac_crossed m0/m1 | score | verdict |
|---|---|---|---|
| 5e-4 | 0.52 / 0.85 | 0.831 | alive, m1 excellent, m0 timing late (175ms) |
| **1e-3** | **0.46 / 0.50** | **0.587** | **winner on score, m1 marginal** |
| 3e-3 | 0.00 / 0.00 | 4.327 | dead by epoch 60 |
| 5e-3 | 0.00 / 0.00 | 5.401 | dead by epoch 90 |

Additional singles run outside sweep grid:
- **LR=7e-4:** score=0.938, loss=2.177 — m0 t_vortex=115ms ✓, m1 t_vortex=177ms (late), frac_crossed=0.50/0.72
- **LR=1e-4:** score=2.637, loss=1.794 (lowest seen to this point) — m0 t_vortex=138ms, m1 t_vortex=108ms ✓✓, frac_crossed=0.54/0.97, m1 A=0.607 (collapsed), race stable throughout

Full picture across all tested LRs:

| LR | score | loss | m0 t_vortex | m1 t_vortex | frac_crossed m0/m1 |
|---|---|---|---|---|---|
| 1e-4 | 2.64 | **1.79** | 138ms | **108ms** ✓ | 0.54 / 0.97 |
| 5e-4 | 0.83 | 2.05 | 175ms | 91ms | 0.52 / 0.85 |
| 7e-4 | 0.94 | 2.18 | **115ms** ✓ | 177ms | 0.50 / 0.72 |
| 1e-3 | **0.59** | 2.60 | **115ms** ✓ | 163ms | 0.46 / 0.50 |
| ≥3e-3 | dead | — | — | — | 0 / 0 |

Clean LR inversion: as LR decreases, m1 improves and m0 degrades. No single LR fits both states at 300 epochs — this is a capacity/epoch problem, not a hyperparameter problem.

---

### Sweep 24 — epoch extension at LR=7e-4 and LR=1e-4

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.a_exo=3 --set task.tau_exo=30 \
    --set model.n_hidden=64 \
    --set task.t_pre=100 --set task.t_post=250 --set task.rpt_step=30 \
    --set task.sigma_init_shared=0.3 --set task.sigma_init_private=0.05 \
    --set task.threshold=0.40 \
    --sweep train.lr=7e-4,1e-4 \
    --sweep train.epochs=600,1000 \
    --no-plots --top 4
```

| config | score | loss | m0 t_vortex | m1 t_vortex | frac_crossed m0/m1 | m1 A |
|---|---|---|---|---|---|---|
| **1e-4 / 600ep** | **0.904** | **1.106** | 63ms | 124ms | 0.54 / 0.977 | 1.000 |
| 7e-4 / 1000ep | 1.463 | 1.635 | 115ms ✓ | 198ms | 0.50 / 0.698 | 1.000 |
| 7e-4 / 600ep | 2.021 | 1.341 | 61ms | 195ms | 0.50 / 0.701 | 1.000 |
| 1e-4 / 1000ep | 2.841 | 1.359 | 168ms | 117ms | 0.54 / 0.968 | 0.739 |

**LR=1e-4 / 600 epochs wins** (score=0.904, loss=1.106 — lowest loss seen by wide margin).

Key findings:
- LR=1e-4 at 1000 epochs is **worse** than at 600 (m0 regresses to 168ms, m1 A collapses to 0.739) — model is un-learning past 600 epochs, cycling in a flat basin.
- LR=7e-4 at 1000 epochs: m1 t_vortex worsens (195→198ms). The m0/m1 split widens with more epochs, not closes.
- The m0/m1 t_vortex split is permanent at n_hidden=64: when m0 t_vortex≈115ms, m1 t_vortex≈195–200ms; when m1 t_vortex≈108–124ms, m0 t_vortex≈60–80ms. No epoch count or LR resolves this simultaneously.

**Conclusion: n_hidden=64 capacity ceiling confirmed** — model cannot fit both maturation states' temporal dynamics simultaneously. D≈0 across nearly all configs is a related symptom. **Locked baseline: LR=1e-4, 600 epochs. Moving to Phase 2: n_hidden scale-up.**

---

### Sweep 25 — n_hidden capacity

```bash
# Ghost point: n_hidden=64
//   ghost: sweep 24 (1e-4/600ep) — score=0.904, loss=1.106,
//   m0 t_vortex=63ms, m1 t_vortex=124ms, frac_crossed=0.54/0.977, D≈0/0.109

python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.a_exo=3 --set task.tau_exo=30 \
    --set task.t_pre=100 --set task.t_post=250 --set task.rpt_step=30 \
    --set task.sigma_init_shared=0.3 --set task.sigma_init_private=0.05 \
    --set task.threshold=0.40 \
    --set train.lr=1e-4 \
    --set train.epochs=600 \
    --sweep model.n_hidden=100,128 \
    --no-plots --top 2
```


Key question: does either size allow both m0 and m1 t_vortex to land near 105ms simultaneously?

| n_hidden | score | loss | frac_crossed m0/m1 | m0 t_vortex | m1 t_vortex | D_m0/m1 | verdict |
|---|---|---|---|---|---|---|---|
| 100 | 1.144 | 1.492 | 0.71 / 0.93 | 82ms | 82ms | ≈0 / ≈0 | alive, m0/m1 collapsed to identical |
| 128 | 5.015 | 3.309 | 0.01 / 0.02 | 195ms | 141ms | ≈0 / 0.306 | Phase-0 failure — dead |

n_hidden=100: Phase-0 gate passed (best frac_crossed seen post-architecture) but m0/m1 t_vortex collapsed to identical 82ms — maturation states not differentiated. n_hidden=128: hard Phase-0 failure, frac_crossed ≈0 throughout training.

**Conclusion:** n_hidden=100 is alive but shows m-collapse; n_hidden=128 is Phase-0 dead at θ=0.40. LR re-check at n_hidden=100 next.

---

### Sweep 26 — LR re-check at n_hidden=100

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.a_exo=3 --set task.tau_exo=30 \
    --set task.t_pre=100 --set task.t_post=250 --set task.rpt_step=30 \
    --set task.sigma_init_shared=0.3 --set task.sigma_init_private=0.05 \
    --set task.threshold=0.40 \
    --set model.n_hidden=100 \
    --set train.epochs=600 \
    --sweep train.lr=3e-4,5e-4,7e-4 \
    --no-plots --top 3
```

| LR | score | frac_crossed m0/m1 | m0 t_vortex | m1 t_vortex | pattern |
|---|---|---|---|---|---|
| 3e-4 | 4.50 | 0.24 / 0.16 | 200ms | 200ms | Phase-0 collapse during training |
| 5e-4 | 3.99 | 0.04 / 0.03 | 139ms | 126ms | fast Phase-0 collapse |
| 7e-4 | 5.14 | 0.01 / 0.004 | 197ms | 127ms | fastest collapse |

All three start healthy (crossed=1.00 epoch 0) then training kills the race. LR hypothesis falsified — this is active destabilization during training, not wrong LR basin. Lapse branch gradient suppressing readout gain mid-training at all LRs > 1e-4. LR=1e-4 (sweep 25) survived only by staying near initialization — too slow to differentiate m states.

**Conclusion:** LR is not the lever at n_hidden=100. θ re-evaluation next.

---

### Sweep 27 — θ re-evaluation at n_hidden=100

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.a_exo=3 --set task.tau_exo=30 \
    --set task.t_pre=100 --set task.t_post=250 --set task.rpt_step=30 \
    --set task.sigma_init_shared=0.3 --set task.sigma_init_private=0.05 \
    --set model.n_hidden=100 \
    --set train.epochs=600 \
    --sweep task.threshold=0.25,0.30,0.35 \
    --sweep train.lr=1e-4,3e-4 \
    --no-plots --top 6
```

| θ / LR | score | frac_crossed m0/m1 | m0 t_vortex | m1 t_vortex | D | pattern |
|---|---|---|---|---|---|---|
| 0.25 / 1e-4 | 1.200 | 0.97 / 0.99 | 80ms | 80ms | ≈0/≈0 | A=1.0, early-commit degenerate |
| 0.25 / 3e-4 | 1.189 | 0.97 / 0.96 | 80ms | 80ms | ≈0/≈0 | same |
| 0.30 / 1e-4 | 1.144 | 0.94 / 0.98 | 82ms | 82ms | ≈0/≈0 | same |
| 0.30 / 3e-4 | 1.242 | 0.93 / 0.93 | 80ms | 74ms | ≈0/≈0 | slight split |

Uniform failure mode: A≈1.0, D≈0, t_vortex ~80ms for both states — model over-commits extremely early. At θ≤0.35 with n_hidden=100 the race crosses threshold before exogenous signal can act. Sweep stopped early (4/6 runs) — θ=0.35 points would not reverse trend.

**Conclusion:** n_hidden=100 has no viable operating window — too high θ kills frac_crossed (sweep 26), too low θ causes early-commit degenerate solution. init_rec_scale investigated as potential lever.

---

### Diagnostic — W_in column norms and init_rec_scale wiring

W_in column norms inspected across three checkpoints (init_rec_scale=0.01, 0.03, 0.1):

```
maturation_state: 1.417  (vs go_signal: 1.422, antisaccade_rule: 1.491)
```

- m-input weight is healthy and competitive — not zeroed, not regularized away. Other model's "W_in near zero" hypothesis falsified.
- init_rec_scale sweep (0.01, 0.03, 0.1) produced **identical** W_in, readout, and lapse logit norms across all three checkpoints. Only M and N (low-rank factors) differed slightly.
- **Finding:** in rank-2 LR-RNN, init_rec_scale only affects M and N initialization. With LR=1e-4 over 600 epochs, gradient signal dominates initialization — M/N converge to same place regardless of starting scale. init_rec_scale is a dead knob at these training settings. This is correct behavior for the architecture, not a bug.

**Decision:** pivot to n_hidden=200 with full-scale settings rather than continuing to probe n_hidden=100.

---

### Run 28 — n_hidden=200, smoke preset, 150 epochs, rpt_step=10

First full-capacity diagnostic run.

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.a_exo=3 --set task.tau_exo=30 \
    --set task.t_pre=100 --set task.t_post=250 \
    --set task.rpt_step=10 \
    --set task.sigma_init_shared=0.3 --set task.sigma_init_private=0.05 \
    --set task.threshold=0.40 \
    --set train.lr=1e-4 \
    --set train.epochs=150 \
    --set model.n_hidden=200 \
    --no-plots
```

| | m0 | m1 | target |
|---|---|---|---|
| t_vortex | 184ms | 176ms | 105/106ms |
| D | 0.085 | 0.077 | 0.28/0.27 |
| A | 1.0 | 1.0 | 0.92/0.97 |
| frac_crossed | 0.67 | 0.76 | — |
| vortex_depth | +0.278 | +0.112 | negative |

**Key findings:**
- m0 ≠ m1 for first time at n_hidden=200 — maturation signal differentiating
- D > 0 for both states — genuine vortex structure emerging
- vortex_depth positive (curve above chance) — timing too late, exogenous capture not yet occurring
- gap_max=350 with t_post=250 identified as mismatch — trials with gap>250ms have negative available post-cue time; must fix before extending

**Discussions:** gap_max/t_post mismatch analysis; Salinas 2019 reviewed confirming gap 0-350ms with 450ms RT deadline; lapse branch architecture re-examined against Zhu 2024 — lapse branch setup confirmed correct (lapses are rule-application failures, not maturation-dependent dynamics; m-differentiation belongs in normal branch only); intermediate saccade (40%/23%) finding from Zhou 2016 PNAS confirmed to be outside tachometric curve scope and not a valid loss constraint.

---

### Run 29 — n_hidden=200, 300 epochs, gap_max=350, t_post=250 (mismatched)

```bash
# gap_max=350, t_post=250 — mismatched; ~30% of training trials cut off
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set model.n_hidden=200 \
    --set task.t_post=250 --set task.gap_max=350 \
    --set task.rpt_step=10 \
    --set task.threshold=0.40 \
    --set train.lr=1e-4 \
    --set train.epochs=300 \
    --no-plots
```

| | m0 | m1 | target |
|---|---|---|---|
| t_vortex | 199ms | 156ms | 105/106ms |
| D | 0.104 | 0.154 | 0.28/0.27 |
| A | 1.0 | 1.0 | 0.92/0.97 |
| frac_crossed | 0.46 | 0.58 | — |
| score | 1.397 | | — |

m1 t_vortex moved 176→156ms in 150 additional epochs — trending correctly. D growing. m0/m1 split widening (199 vs 156ms). Loss noisy but descending.

**Discussions:** rpt_max analysis — 240ms ceiling clips t_rise for m0 (199ms at ceiling); rpt_bin_width=12 vs Zhu 2024 bin width of 20ms; decision to move to biologically correct settings: t_post=500, gap_max=350, rpt_max=350, rpt_bin_width=20. Salinas 2019 re-read confirming gap 0-350ms with correct rPT mechanics. ReduceLROnPlateau scheduler analyzed — patience=50 with noisy loss causing erratic firing; plateau_factor=0.99999 and patience=99999 adopted to effectively disable. hard_eval_trials_per_gap=10 identified as a source of noisy hard loss readings.

---

### Run 30 — n_hidden=200, full biological settings, 600 epochs, batch_size=64

First run with correct t_post=500, gap_max=350, rpt_max=350, rpt_bin_width=20.

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set model.n_hidden=200 \
    --set task.t_post=500 --set task.gap_max=350 \
    --set task.rpt_max=350 --set task.rpt_bin_width=20 \
    --set task.rpt_step=10 \
    --set task.threshold=0.40 \
    --set train.lr=1e-4 \
    --set train.epochs=600 \
    --no-plots
```

| | m0 | m1 | target |
|---|---|---|---|
| t_vortex | 200ms | 164ms | 105/106ms |
| D | 0.044 | ≈0 | 0.28/0.27 |
| A | 1.0 | 1.0 | 0.92/0.97 |
| frac_crossed | 0.74 | 0.83 | — |
| vortex_depth | -0.069 | -0.188 | negative ✓ |
| score | 1.635 | | — |
| final_train_loss | 0.966 | | — |

Both vortex_depths genuinely negative for first time at n_hidden=200. frac_crossed healthy. m0/m1 differentiated (200 vs 164ms). Loss (0.966) lowest seen at this scale. Timing still ~60-80ms too late.

**Discussions:** batch_size 64→256 analysis — vortex region underrepresented in small batches (4-5 bins across ~40ms rPT range may have 0-1 trials per batch); 256 gives 4× more vortex-region gradient signal per update; one gradient update per epoch confirmed (train.py:71 — no steps_per_epoch loop). Scheduler erratic firing confirmed as contaminating prior runs. Scheduler effectively disabled with (plateau_patience=99999, plateau_factor=0.99999).

---

### Run 31 — full preset, batch_size=200, scheduler active (bug shown-- revealed argv plateau_patience (etc.) not consumed by training mechanism.)

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset full \
    --set task.a_exo=3 --set task.tau_exo=30 \
    --set task.threshold=0.40 \
    --set task.sigma_init_shared=0.3 --set task.sigma_init_private=0.05 \
    --set train.lr=1e-4 \
    --set train.epochs=500 \
    --set train.batch_size=200 \
    --no-plots
```

**Killed at epoch ~400.** Scheduler fired 6 times in 400 epochs: 1e-4 → 5e-5 → 2.5e-5 → 1.25e-5 → 6.25e-6 → 3.125e-6 → 1.56e-6. By epoch 400 LR was 64× smaller than start. Loss stagnating at 0.82-0.97 with no descent. Loss noise (0.75-1.64 swings) caused scheduler to anchor on early lucky low point (best=0.7434) and keep halving.

**Conclusion:** ReduceLROnPlateau with patience=50 incompatible with noisy tachometric loss. Scheduler param consumption fixed, and LR scheduler was disabled for all subsequent runs.

---

### Run 32 — n_hidden=200, full preset, lr=3e-5, 200 epochs (key diagnostic)

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset full \
    --set model.n_hidden=200 \
    --set train.batch_size=200 \
    --set train.lr=3e-5 \
    --set train.epochs=200 \
    --set task.rpt_bin_width=20 \
    --set train.hard_eval_trials_per_gap=50 \
    --set train.plateau_patience=99999 \
    --set train.plateau_factor=0.99999 \
    --set eval.trials_per_gap=200 \
    --no-plots
```

| | m0 | m1 | target |
|---|---|---|---|
| t_vortex | 165ms | 180ms | 105/106ms |
| t_rise | 165ms | 180ms | 155/140ms |
| D | 0.122 | 0.055 | 0.28/0.27 |
| **A** | **0.900** | **0.964** | **0.92/0.97** |
| frac_crossed | 0.81 | 0.89 | — |
| vortex_depth | -0.184 | -0.278 | negative ✓ |
| score | 1.149 | | — |
| final_train_loss | 0.822 | | — |

**Best result up until this point.** Key achievements:
- A no longer 1.0 — lapse mechanism working: A_m0=0.900 vs target 0.920, A_m1=0.964 vs target 0.970
- Loss stable throughout — no degradation — lr=3e-5 fixed the instability problem
- m0/m1 differentiated, vortex_depth negative both states
- Timing still ~60ms too late

m=0/m=1 convention verified post-run — correct (m0=young, m1=adult). Inverted t_vortex direction (m0 faster than m1) is a genuine model behavior at this training stage, not a labeling error.

---

### Run 33 — n_hidden=200, 2000 epochs from scratch, lr=3e-5 (degradation study)

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset full \
    --set model.n_hidden=200 \
    --set train.batch_size=200 \
    --set train.lr=3e-5 \
    --set train.epochs=2000 \
    --set train.hard_eval_trials_per_gap=50 \
    --set train.plateau_patience=99999 \
    --set train.plateau_factor=0.99999 \
    --set eval.trials_per_gap=200 \
    --no-plots
```

Stopped by user at epoch 1060. Loss trajectory:
- Epochs 0-300: healthy (0.75-0.85, frac_crossed 0.83-0.92) — same good basin as Run 32
- Epochs 300-500: slow upward drift, frac_crossed to 0.77-0.81
- Epochs 500-700: accelerating degradation, loss 0.88-0.95
- Epochs 700-1060: full degradation, loss 0.90-1.15, frac_crossed 0.62-0.80
- Final metrics: A=1.0, D≈0, score=1.651

**Finding:** good basin found reliably at epoch 100-300, but not stable under continued gradient updates at lr=3e-5. Optimizer walks out of basin. The 200-epoch result is the best the model achieves at this LR; more epochs hurt. Discussion: gap sampling change; no longer uniform in 0-350ms, but split into /n_strata (5) bins: 0-70..280-350. At batch_size=200, effective num trials per bin is 40. Batch_size/n_strata determines num trials per sampling bin.

---

### Run 34 — n_hidden=200, 2000 epochs, gap stratification (5 coarse bins)

Gap sampling changed from uniform continuous to stratified over 5 coarse bins covering 0-350ms (equal trials per bin). Same hyperparameters as Single Run 6.

```bash
# gap stratification: 5 bins over 0-350ms, equal trials per bin
python -m antisaccade_model.experiments.opt_behavior_fit --preset full \
    --set model.n_hidden=200 \
    --set train.batch_size=200 \
    --set train.lr=3e-5 \
    --set train.epochs=2000 \
    --set train.hard_eval_trials_per_gap=50 \
    --set train.plateau_patience=99999 \
    --set train.plateau_factor=0.99999 \
    --set eval.trials_per_gap=200 \
    --no-plots
```

Stopped by user at epoch 1060 (frac_crossed=0.62).

Log trajectory:
- Epochs 0-420: healthy (loss 0.75-0.87, frac_crossed 0.81-0.92) — stable longer than Run 33 (420 vs ~300 epochs)
- Epochs ~500-1430: gradual frac_crossed degradation (0.23 by epoch 1430)
- Epochs 1430-2000: frac_crossed collapses to 0.04-0.14, loss 0.94-1.21

Final metrics despite collapsed frac_crossed:
| | m0 | m1 | target |
|---|---|---|---|
| t_vortex | **83ms** | **87ms** | 105/106ms |
| t_rise | 138ms | 114ms | 155/140ms |
| D | **0.219** | **0.249** | 0.28/0.27 |
| A | 0.994 | 0.999 | 0.92/0.97 |
| frac_crossed | 0.090 | 0.062 | — |
| score | 3.432 | | — |

**Key findings:**
- D closest to target ever (0.219/0.249 vs 0.28/0.27) — gap stratification dramatically improved vortex gradient signal
- t_vortex now **too early** (83-87ms) rather than too late — direction flipped by stratification
- t_rise in correct ballpark for first time (138ms vs 155ms target for m0)
- frac_crossed catastrophically low (6-9%) — model learned to only fire on vortex-region trials
- Score inflated by massive crossing penalty (5 × max(0, 0.4 - 0.062) ≈ 1.69 per state)
- Degenerate solution: correct curve shape from tiny fraction of trials

**Interpretation:** two separable problems now visible:
1. Run 32 (no stratification): correct A, stable race, timing too late, shallow D
2. Run 34 (stratification): correct D, timing close, but dead race

Gap stratification fixed the D/t_vortex problem but amplified the lapse-gradient-vs-race-health tension. Crossing penalty weight (5×) insufficient to prevent optimizer from trading race health for curve shape over long training.

**Discussion:** Salinas-inspired gap strata (boundaries at 0,75,100,125,150,175,200,250,350ms) discussed as more biologically grounded alternative to equal-width bins — denser sampling in vortex-producing gap range (75-200ms) matching experimental design. Next run: resume Run 32 checkpoint with lr=1e-5.

---

## Current best config (updated)

| field | value | source |
|---|---|---|
| `model.n_hidden` | 200 | Run 32 |
| `model.n_rank` | 2 | Locked (architectural) |
| `task.threshold` | 0.40 | Sweep 22 |
| `task.a_exo` | 3.0 | Sweep 3 |
| `task.tau_exo` | 30.0 | Sweep 4 |
| `task.t_pre` | 100 | Sweep 13 |
| `task.t_post` | 500 | Biological target (Salinas 2019) |
| `task.gap_max` | 350 | Biological target (Salinas 2019) |
| `task.rpt_max` | 350 | Biological target (Zhu 2024 curve range) |
| `task.rpt_step` | 10 | Sweep 28 |
| `task.rpt_bin_width` | 20 | Zhu 2024 binning |
| `task.sigma_init_shared` | 0.3 | Sweep 20 |
| `task.sigma_init_private` | 0.05 | Sweep 20 |
| `task.sigma_noise` | 0.1 | Default; not yet swept |
| `task.commit_temp` | 0.2 | Default; Phase 3 knob |
| `train.lr` | 3e-5 | Run 32 |
| `train.batch_size` | 200 | Run 32 |
| `train.epochs` | 200 (good basin)  Run 32 |
| `train.hard_eval_trials_per_gap` | 50 | Run 32 |
| `train.plateau_patience` | 99999 | Run 31 (disable scheduler) |
| `train.plateau_factor` | 0.99999 | Run 31 (disable scheduler) |
| `eval.trials_per_gap` | 200 | Run 32 |
| `model.lapse_young_init` | 0.08 | Salinas 2019 + Zhu 2024 inference |
| `model.lapse_adult_init` | 0.02 | Salinas 2019 |

**Best run:** Run 34.

---

## Considerations moving forward

1. **The gradient-bearing loss explicitly upweights the vortex/recovery region.** Per-trial BCE is weighted `3.0` for emergent rPTs from 70–200 ms, `1.0` for >200–300 ms, and `0.5` elsewhere. Run 34 therefore reflects the combined effect of balanced five-stratum gap coverage and an already vortex-heavy optimization objective. Before making sampling denser in the same region, **COULD** reduce/rebalance the explicit rPT weights to try and preserve Run 32's race health while retaining Run 34's gains in `D` and timing.
2. **Current five-stratum gap sampling controls finite-batch coverage; it is not itself a vortex-specific weighting scheme.** The current sampler draws equal counts from five equal-width gap intervals over 0–350 ms and samples uniformly within each interval. Salinas-inspired nonuniform boundaries would be a distinct experimental-sampling choice that further concentrates coverage around vortex-producing gaps.
3. **The existing no-crossing penalty cannot prevent training collapse.** The `5 * max(0, 0.4 - frac_crossed)` term is applied only to post-training evaluation/ranking and supplies no gradient to Adam. Increasing it would demote collapsed runs but would not change their training trajectory. **COULD** require a minimum `frac_crossed` gate for checkpoint eligibility, and select the best viable checkpoint rather than the lowest BCE or final epoch (but this seems hacky right now). If collapse remains common, **COULD** add a differentiable race-health or survival-mass term to the training objective.
4. **Phase 3 biological tuning follows a stable viable fit.** Once both maturation states retain healthy `frac_crossed`, can tune parameters such as `tau_exo` (timing) and `commit_temp` (recovery sharpness). Consider `n_hidden=256` only after the `n_hidden=200` configuration remains stable under the hard behavioral gate.
