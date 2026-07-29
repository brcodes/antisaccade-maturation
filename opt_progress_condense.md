# opt_progress_condense.md — context handoff

Targets (ground truth, confirmed correct from sweep 2 onward):
- young: A=0.92, t_rise=155ms, t_vortex=105ms, D=0.28
- adult: A=0.97, t_rise=140ms, t_vortex=106ms, D=0.27
- score = weighted behavioral MSE + crossing penalty; lower is better; always read alongside frac_crossed

**Architectural status:** Both identified gaps are implemented and validated:
lapse branch mixture λ(m), and stochastic h0 via shared+private factorization.
The newly sweepable controls are `task.sigma_init_shared`,
`task.sigma_init_private`, `model.lapse_young_init`, and
`model.lapse_adult_init`. Initial lapse values are priors for learned endpoints,
not fixed probabilities. The architecture changed the loss landscape.

**rpt_step note:** affects the soft-binning training objective, not just
evaluation. `eval.trials_per_gap` improves hard-evaluation precision only;
changing `rpt_step` or `rpt_bin_width` requires another crossing/LR check.

**t_pre note:** pre-cue attractor establishment window. Low-priority variable — matters only if RNN hasn't converged to baseline within 50ms of simulation time. Likely fast at n_hidden=64.

---

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --sweep task.threshold=0.2,0.3,0.5,0.7 --no-plots
```
- Targets were wrong (t_vortex/D off); rankings invalid. Re-run after fix = sweep 2.

---

```bash
# sweep 2 — same command, corrected targets
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --sweep task.threshold=0.2,0.3,0.5,0.7 --no-plots
```
- θ=0.7 wins (score 1.522); only config to hit D≈0.28. t_rise=230ms and t_vortex=200ms badly late. θ is the Phase-0 dynamic-range knob. θ=0.7 adopted as baseline.

---

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --sweep task.threshold=0.6,0.65,0.7,0.75,0.8 \
    --sweep task.a_exo=3,5 \
    --no-plots --top 5
```
- θ=0.75/a_exo=3 wins (score 0.297, t_vortex=106ms) but D≈0. θ=0.7 holds D≈0.28 but t_vortex=200ms. D and t_vortex anti-correlated across θ. a_exo=5 universally worse — eliminated. a_exo locked at 3.

---

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.a_exo=3 \
    --sweep task.threshold=0.7,0.72,0.75,0.78 \
    --sweep task.tau_exo=10,20,30 \
    --no-plots --top 6
```
- τ_exo=10 kills frac_crossed; τ_exo=20 marginal; τ_exo=30 correct. D vs t_vortex tradeoff is structural, not fixable with τ_exo. τ_exo locked at 30 (= library default).

**Measurement artifact diagnostic (sweep 4):** When D collapses to ~0 while vortex_depth is genuinely negative (e.g., sweep 3 at θ=0.75/a_exo=3), this is a Gaussian fit failure at smoke resolution. The empirical binned curve crosses below chance but bins are too sparse for `curve_fit` to recover amplitude. This artifact re-appears in later configs and should not be misinterpreted as mechanism failure.

---

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 \
    --set task.a_exo=3 \
    --set task.tau_exo=30 \
    --sweep train.epochs=50,150,300,500 \
    --no-plots --top 4
```
- vortex_depth < 0 at all epoch counts — vortex mechanism works from the start. D≈0 was Gaussian fit failure on a real but shallow dip, not absent mechanism. a_exo is additive (not subtractive); below-chance dip must be learned via recurrent weights. Best fit at 300 epochs. Smoke epoch baseline set to 300.

---

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set train.epochs=300 \
    --set task.tau_exo=30 \
    --sweep task.threshold=0.7,0.72,0.75,0.78 \
    --sweep task.a_exo=3,5,8 \
    --no-plots --top 6
```
- θ=0.75/a_exo=3 wins (score 0.531) at honest 300-epoch ranking. t_vortex converged to ~115–118ms across viable configs — τ_exo=30 is the dominant timing controller. a_exo=8 eliminated. A≈0.78–0.80 and D shallow — undertrained, not wrong parameters. Phase 1 gate reached.

---

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 \
    --set task.a_exo=3 \
    --set task.tau_exo=30 \
    --set train.epochs=300 \
    --sweep train.lr=1e-4,3e-4,1e-3,3e-3 \
    --no-plots --top 4
```
- LR=1e-3 wins (score 0.531). LR=3e-3 best A and D but t_rise overshoots. LR≤3e-4 dead — false minimum with lowest training loss. Never trust raw training loss alone.

---

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 \
    --set task.a_exo=3 \
    --set task.tau_exo=30 \
    --set train.epochs=300 \
    --sweep train.lr=1e-3,1.5e-3,2e-3,3e-3 \
    --no-plots --top 4
```
- No clean optimum in gap; 2e-3 worse than both neighbors (different basins). LR=1e-3 locked for n_hidden=64.

---

```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 \
    --set task.a_exo=3 \
    --set task.tau_exo=30 \
    --set train.lr=1e-3 \
    --sweep train.epochs=300,500,750,1000 \
    --no-plots --top 4
```
- A stuck at 0.769–0.799 regardless of epoch count — capacity ceiling at n_hidden=64 confirmed. 500-epoch instability noted (t_vortex crashes). D m1 degrades with more training. Phase 2: raise n_hidden.

---

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
- All n_hidden > 64 dead at LR=1e-3 (frac_crossed=0.00); n_hidden=128 diverged. LR must be re-tuned after any n_hidden change — larger models may need higher or lower LR depending on whether capacity or initialization inertia dominates.

---

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
- LR=3e-3 and 5e-3 both viable (frac_crossed=1.00), converge to same attractor (confirmed different seeds). LR=3e-3 locked as minimum viable for n_hidden=100. Score 0.132 vs 0.531 at n_hidden=64 — genuine capacity gain. A=0.850, t_vortex=108ms. D=0.450 flagged as likely Gaussian fit artifact.

---

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
- Library default grad_clip already = 1.0; sweeping it changed nothing. Spikes are loss noise from stochastic Monte Carlo behavioral objective — not gradient explosion. batch_size confirmed at library default 256. At practical limit of smoke resolution.

---

```bash
# Run 13 — full resolution diagnostic (n_hidden=100)
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 --set task.a_exo=3 --set task.tau_exo=30 \
    --set model.n_hidden=100 --set train.lr=3e-3 --set train.epochs=300 \
    --set task.t_pre=100 --set task.t_post=500 --set task.rpt_step=10 \
    --no-plots
```
- Dead race from epoch 10. Score=4.13, frac_crossed=0.00 both states. LR=3e-3 tuned for smoke resolution does not transfer to full timeline — same principle as sweep 10 (LR is resolution-dependent).

---

```bash
# Sweeps 14–15 — LR search at n_hidden=100, t_post=500 (abandoned)
--sweep train.lr=3e-3,5e-3,8e-3,1.2e-2   # sweep 14
--sweep train.lr=1.2e-2,1.5e-2,2e-2,3e-2  # sweep 15
```
- LR=1.2e-2 only partial escape (m1 only, never m0). All others dead. No viable config at n_hidden=100/t_post=500. Abandoned; fell back to n_hidden=64.

---

```bash
# Sweep 16 — LR search at n_hidden=64, t_post=500
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 --set task.a_exo=3 --set task.tau_exo=30 \
    --set model.n_hidden=64 --set train.epochs=300 \
    --set task.t_pre=100 --set task.t_post=500 --set task.rpt_step=10 \
    --sweep train.lr=1e-3,3e-3,5e-3,8e-3 --no-plots --top 4
```
- LR=8e-3 only survivor (crossed 0.55–1.00 all 300 epochs) but loss doesn't descend. Lower LRs establish race then die. Pattern: race finds good basin early, then walks out at every LR. Two-stage training implemented.

---

```bash
# Two-stage training — harness modification
# Added resume_checkpoint field to TrainConfig; loads state_dict only (not optimizer state)
# Stage 1: establish race at high LR; Stage 2: resume at low LR to descend

# Sweeps 17a/b/c — two-stage at n_hidden=64, t_post=500 (failed)
# 17a: 100 epochs LR=8e-3 → checkpoint results/opt/single_20260728_163950/model.pt
# 17b: 200 epochs LR=1e-3 resume — race alive but loss flat (7.3–11.0), no descent
# 17c: sweep LR=2e-3,3e-3,5e-3 resume — all find good basin briefly then die
#   LR=2e-3: loss 4.52 at epoch 130, dead by epoch 199
#   LR=3e-3: loss 6.27 by epoch 60, dead by epoch 120
#   LR=5e-3: loss 4.51 at epoch 30(!), immediately destabilizes
```
- Two-stage delays collapse but doesn't prevent it. Not an LR problem — structural. Warmup ruled out (collapse 100+ epochs after curriculum stabilizes). STE hypothesis raised, then tested in sweep 18.

---

```bash
# Sweep 18 — t_post isolation (STE hypothesis test)
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 --set task.a_exo=3 --set task.tau_exo=30 \
    --set model.n_hidden=64 --set train.epochs=300 --set train.lr=8e-3 \
    --set task.t_pre=100 --set task.rpt_step=10 \
    --sweep task.t_post=150,250,350,500 --no-plots --top 4
```
- t_post=150: dead by epoch 60, false minimum (loss ~1.9, crossed=0.00)
- t_post=250: crossed=1.00 both states all 300 epochs, score=0.132 — reproduces smoke result exactly
- t_post=350: dead at epoch 40, loss ~1.0–1.3, A=0.9999 (constant readout, no race)
- t_post=500: crossed 0.55–1.00, score=1.72, but loss never descends

STE hypothesis ruled out — non-monotone relationship between t_post and survival. Key finding: **t_post=250 is the viable training window**. The full-resolution confound was always t_post specifically, not t_pre or rpt_step.

---

```bash
# Architectural changes — code_v2_delta_edit_delta (implemented and validated post-sweep 18)
```

Both architectural gaps identified as required before further fitting are now implemented:

**Lapse mechanism** — branch mixture. Normal and lapse branches run in parallel; mixed with λ(m). Rule input channel zeroed on lapse branch (m-leakage test passed: max output diff across m on lapse branch = 0.0). λ_young=0.08, λ_adult=0.02, both learned via sigmoid-constrained logits. Lapse branch contributes gradient even when normal branch doesn't cross threshold — this is the primary reason t_post=500 may now be viable.

**Stochastic initial state** — shared+private factorization. σ_shared=0.7, σ_private=0.05. Provides trial-to-trial RT variability and more realistic vortex depth.

**Validation complete:** full forward pass with lapse+h0 active; all parameters receive gradients; lapse logits at correct initial values. Output correlation (0.230) and curve stability (50–80ms) at 50 epochs reflect untrained W_out geometry, not implementation errors.

**Loss landscape impact:** both changes alter gradient flow. LR must be re-confirmed post-architecture.

---

**Current locked config (pre-architecture):**
`θ=0.75, a_exo=3, τ_exo=30, n_hidden=64, LR=8e-3, epochs=300, batch_size=256, t_post=250`

**Status:** Architecture (lapse mechanism + stochastic h0) now implemented and validated. Loss landscape altered — LR and rpt_step must be re-confirmed post-architecture.

**Original next-sweep plan (superseded by the sweep-19 result below):**

**Sweep 19** — rpt_step viability at t_post=250/n_hidden=64:
```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 --set task.a_exo=3 --set task.tau_exo=30 \
    --set model.n_hidden=64 --set train.lr=8e-3 --set train.epochs=300 \
    --set task.t_pre=100 --set task.t_post=250 \
    --sweep task.rpt_step=30,10 --no-plots --top 2
```
If rpt_step=10 stable → lock it. If not → keep rpt_step=30 for training. LR=8e-3 is baseline, will be re-confirmed in sweep 20.

**Sweep 20** — LR re-search at n_hidden=64/t_post=250 with new architecture:
```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 --set task.a_exo=3 --set task.tau_exo=30 \
    --set model.n_hidden=64 --set train.epochs=300 \
    --set task.t_pre=100 --set task.t_post=250 \
    --set task.rpt_step=[from sweep 19] \
    --sweep train.lr=1e-3,3e-3,5e-3,8e-3 --no-plots --top 4
```
Lapse + h0 both alter loss landscape. Pre-architecture used LR=1e-3 (smoke) or LR=8e-3 (t_post=500). Sweep full range; watch frac_crossed as health metric.

**Sweep 21** — t_post=500 retest with new architecture:
```bash
python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 --set task.a_exo=3 --set task.tau_exo=30 \
    --set model.n_hidden=64 --set train.epochs=300 \
    --set task.t_pre=100 --set task.rpt_step=[from sweep 19] \
    --set train.lr=[from sweep 20] --set task.t_post=500 --no-plots
```
Lapse branch now provides gradient when normal branch doesn't cross — likely fixes t_post=500 non-learning. If stable and loss descends: t_post=500 viable. If not: investigate rPT distribution.

---

## Sweep 19 result and superseding trajectory

**Sweep 19 result:** the planned `rpt_step=30,10` comparison is invalid as a
resolution ranking. At `n_hidden=64`, `t_post=250`, and 300 epochs, both hard
endpoint curves had `frac_crossed=0.00` at both grid sizes under both tested
learning rates:

| lr | rpt_step | score | frac_crossed m0/m1 |
|---|---:|---:|---|
| 8e-3 | 30 | 4.398 | 0.00 / 0.00 |
| 8e-3 | 10 | 4.814 | 0.00 / 0.00 |
| 1e-3 | 30 | 5.210 | 0.00 / 0.00 |
| 1e-3 | 10 | 5.670 | 0.00 / 0.00 |

The post-architecture issue is Phase 0/1 recovery, not rPT resolution. The
soft loss is not a substitute for hard `frac_crossed` here.

**Next sweep 20a — recover a live hard race:** hold
`n_hidden=64`, `t_post=250`, `rpt_step=30`, and fixed task knobs. At `lr=1e-3`,
sweep `task.sigma_init_shared=0.3,0.5,0.7` and
`task.sigma_init_private=0.0,0.05`; rank by `frac_crossed_*` before score.
Keep lapse endpoint initializers fixed.

**Then 20b:** re-search `lr=1e-3,3e-3,5e-3,8e-3` with the recovered initial
state pair. **Then 20c:** retry `rpt_step=30,10`; accept 10 only if both states
retain healthy hard crossings. **Then sweep 21:** retest `t_post=500` with the
recovered LR/grid pair. Lapse prior sweeps and n_hidden re-evaluation follow
only after these gates pass.
