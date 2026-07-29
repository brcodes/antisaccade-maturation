# opt_progress.md — Behavior-Fit Optimization Log

Live results + decision log for the optimization harness. Reference doc (modes, artifacts, sweep philosophy, scale-up) is [OPT_README.md](OPT_README.md).

Append one dated entry per accepted finding. Record the command, the key numbers, the interpretation, and the resulting decision. Keep the current best config and next steps sections at the top up to date.

**Ghost-point protocol:** when a sweep is proposed, any config that duplicates a previously-run point is listed as a **ghost point** — noted for context in the analysis but not re-run. Ghost points are marked inline: `// ghost: sweep N — <key metrics>`.

---

## Current best config (living)

| field | value | source |
|---|---|---|
| `task.threshold` | **0.40** | sweep 22 — post-architecture cliff mapping |
| `task.a_exo` | 3.0 | sweep 3 |
| `task.tau_exo` | 30.0 | sweep 4 (confirmed = library default) |
| `model.n_hidden` | **pending sweep 25** | sweep 24 confirmed n_hidden=64 capacity ceiling |
| `train.lr` | **1e-4** | sweep 24 winner |
| `train.epochs` | **600** | sweep 24 winner |
| `train.batch_size` | 256 | library default confirmed |
| `task.t_pre` | 100 | set explicitly since sweep 13 |
| `task.t_post` | 250 | sweep 18 — viable training window |
| `task.rpt_step` | 30 | sweep 19 — rpt_step=10 not yet retested post-architecture |
| `task.sigma_init_shared` / `task.sigma_init_private` | 0.3 / 0.05 | sweep 20 winner |
| `model.lapse_young_init` / `model.lapse_adult_init` | 0.08 / 0.02 | architecture defaults; not yet swept |

**Note on threshold:** θ=0.40 is meaningfully lower than the pre-architecture optimum (0.75). This is an engineering consequence of the lapse branch suppressing readout gain, not a scientific claim about the biological decision bound. θ is a detection threshold on the model's output activations, not the accumulator bound height. The biological quantities of interest (t_rise, t_vortex, A, D, SI signal shape) are what validate the model, not the θ value itself. See threshold scientific note under sweep 21.

**Note on n_hidden:** sweep 24 confirmed n_hidden=64 cannot simultaneously fit m0 and m1 t_vortex timing — a clean capacity ceiling. Sweep 25 probes n_hidden=100/128 at the new LR/epoch baseline. Pre-architecture n_hidden=100 at smoke resolution (sweep 11) achieved score=0.132 at LR=3e-3; post-architecture LR landscape is different and must be re-confirmed if frac_crossed collapses.

**Current phase:** Phase 2 — capacity scale-up. Phase-0 gate re-established at θ=0.40 (sweep 22). LR and epoch baseline confirmed (sweeps 23–24). n_hidden=64 ceiling confirmed (sweep 24). Sweep 25 in progress.

---

## Next steps (specific, ordered)

1. **Sweep 25 — n_hidden capacity** — in progress. n_hidden=100 and 128 at lr=1e-4, epochs=600. Ghost: n_hidden=64 (sweep 24, score=0.904). If new sizes collapse → LR re-check at that size before concluding capacity doesn't help.
2. **LR re-check at new n_hidden** — if sweep 25 shows frac_crossed collapse, sweep lr=5e-4,1e-4,1e-3 at the viable size.
3. **rpt_step retry** — after stable n_hidden/LR locked, test rpt_step=10. This changes the soft-binning training objective; re-confirm frac_crossed after.
4. **t_post=500 retest** — only after rpt_step confirmed. Lapse branch may now provide gradient where pure normal branch couldn't.
5. **Lapse endpoint sweep** — narrow pairs around (0.08, 0.02); after the above gates pass.

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

### Sweep 25 — n_hidden capacity (in progress)

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

Key question: does either size allow both m0 and m1 t_vortex to land near 105ms simultaneously? If yes: Phase 2 breakthrough. If frac_crossed collapses → LR re-check at that size before concluding capacity doesn't help. Lapse branch now provides gradient through dead-race episodes, so the pre-architecture pattern (n_hidden>64 requiring higher LR) may not hold here.

*Results pending.*
