# opt_progress_condense.md — context handoff

Targets (ground truth, confirmed correct from sweep 2 onward):
- young: A=0.92, t_rise=155ms, t_vortex=105ms, D=0.28
- adult: A=0.97, t_rise=140ms, t_vortex=106ms, D=0.27
- score = weighted behavioral MSE + crossing penalty; lower is better; always read alongside frac_crossed

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
- Library default grad_clip already = 1.0; sweeping it changed nothing (identical loss values confirmed). Spikes are loss noise from stochastic Monte Carlo behavioral objective, not gradient explosion — crossed=1.00 throughout proves race is stable. batch_size confirmed at library default 256. At practical limit of smoke resolution: t_post=250 (vs full 500) truncates A/t_rise gradient; rpt_step=30 (vs full 10) gives ~6 bins instead of ~18. Loss landscape degeneracy confirmed (different LRs → same behavioral output). Next: restore full timeline and bin resolution.

---

**Current locked config:**
`θ=0.75, a_exo=3, τ_exo=30, n_hidden=100, LR=3e-3, epochs=300, batch_size=256`

**Next run:**
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
Single diagnostic run. Success gate: loss shows genuine downward trend (not 3.5–9.2 pinball). Score improves vs 0.132.
