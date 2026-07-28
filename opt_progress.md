# opt_progress.md — Behavior-Fit Optimization Log

Live results + decision log for the optimization harness. Reference doc (modes,
artifacts, sweep philosophy, scale-up) is [OPT_README.md](OPT_README.md).

**How to use this file:** append one dated entry per accepted finding. Record the
command, the key numbers, the interpretation, and the resulting decision. Keep
the **Current best config** and **Next steps** sections at the top up to date.

---

## Current best config (living)

| Field | Value | Source |
|---|---|---|
| `task.threshold` | **0.5** (was 1.0) | Entry 1 |
| `task.a_exo` | 3.0 | Entry 1 |
| everything else | `smoke` preset defaults | — |

**Current phase:** Phase 0 → 1 (plumbing verified; optimization not yet tuned).

---

## Next steps (specific, ordered)

Following the [OPT_README §7](OPT_README.md) phase framework. Do these top-down;
freeze each result here before moving on.

1. **Finish Phase 0 — pin the crossing knob.** Confirm the threshold with a
   finer 1-D sweep and pick the smallest θ that keeps `frac_crossed` in
   0.5–0.9 **without** wrecking behavioral stats:
   ```bash
   python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
       --sweep task.threshold=0.4,0.5,0.6,0.7 --no-plots --top 4
   ```
   - Watch: `frac_crossed_m*` (want 0.5–0.9) **and** `score`.
   - Open question from Entry 1: θ=0.3 crossed more but scored **worse** (2.5 vs
     0.10). Hypothesis: too-low θ commits too early → bad `t_rise` / vortex.
     This sweep should confirm the non-monotonic θ↔score relationship.

2. **Phase 1 — learning rate.** With θ fixed, sweep LR first:
   ```bash
   python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
       --set task.threshold=0.5 \
       --sweep train.lr=3e-4,1e-3,3e-3 --no-plots --top 3
   ```
   - Success gate: clean, monotone-ish loss decrease over 50 epochs; no NaNs.

3. **Phase 1 — stability follow-ups (only if LR alone is unstable).**
   `train.grad_clip`, then `train.batch_size`, then `train.warmup_epochs`.

4. **Phase 2 — capacity.** Once LR is locked, sweep `model.n_hidden`
   (e.g. 64,128) then `model.lambda_reg`. Look for `A` and a real vortex.

5. **Then scale up** per [OPT_README §8](OPT_README.md): epochs → resolution →
   full timeline → `n_hidden`=200 → full batch/epochs → promote to
   `run_behavior_fit.py`.

**Deferred (Phase 3–4, biological):** `commit_temp`/`option_temp`,
`sigma_noise` → vortex depth `D`, `a_exo`/`tau_exo` → vortex timing/depth,
maturation interpolation across `m`. Do not touch until a config trains + fits.

---

## Results log

### Entry 1 — 2026-07-28 — Threshold is the Phase-0 gate

**Context.** First runs of the new harness on the `smoke` preset. With the
library-default `task.threshold = 1.0`, the decision race never reached the bound.

**Observations.**
- Default θ=1.0, single run (5 epochs): `frac_crossed = 0.00` for both `m`;
  `t_rise` came back `nan` (untrained/degenerate curve). Score is meaningless in
  this regime — the curve never forms.
- Sweep `task.threshold=0.3,0.5 × task.a_exo=3,5` (3 epochs each), ranked:

  | rank | threshold | a_exo | score | final_loss | frac_crossed (m0) |
  |---|---|---|---|---|---|
  | 1 | **0.5** | 3 | **0.1041** | 5.3923 | ~0.50 |
  | 2 | 0.5 | 5 | 0.3793 | 5.1403 | — |
  | 3 | 0.3 | 3 | 2.5051 | 5.5598 | — |
  | 4 | 0.3 | 5 | 2.5078 | 5.5598 | — |

**Interpretation.**
- **`task.threshold` is the Phase-0 dynamic-range knob.** Dropping θ from 1.0 to
  0.5 moved `frac_crossed` from 0.00 → ~0.50 and the score from meaningless →
  0.10. This is the single change that made every downstream metric valid.
- **θ is non-monotonic w.r.t. fit:** θ=0.3 crossed *more* but scored *worse*
  (2.5). Lower θ ⇒ earlier commitment ⇒ distorted `t_rise` / vortex. So the goal
  is not "maximize crossings" but "smallest θ that keeps crossings healthy
  without corrupting timing." Flagged for the Next-step 1 finer sweep.
- **`a_exo` (3 vs 5) was second-order** at this stage — real but small relative
  to θ. Leave at 3.0 for now; revisit in the biological phase (vortex tuning).
- `final_loss` and `score` disagree in ranking (θ=0.3 had lower loss but worse
  score). Confirms the [OPT_README §6](OPT_README.md) point: **read
  `frac_crossed` + `score`, not raw training loss.**

**Decision.** Adopt `task.threshold = 0.5` as the working smoke baseline.
Proceed to Next-step 1 (finer θ sweep) to lock Phase 0, then LR (Phase 1).

**Open questions carried forward.**
- Exact θ that best trades crossing vs. timing (Next-step 1).
- Should the *library default* `TaskParams.threshold` be lowered, or should θ
  stay a swept knob only? (Decide after Next-step 1.)


### 2026-07-28

Thresh 0.7 looked the best, (0.2,0.3...)
Didn't nan out the t_rise for m=0
(that nan could have been task. or eval.'s rpt_step=30 (too coarse) or trials_per_gap=100 (not enough data for smooth interpolation) (of 75% point))

Stuck with 0.7
Ran

(tcia-lung1-seg-class-cpu) brycerogers@Bryces-MacBook-Pro antisaccade_maturation % python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --sweep task.threshold=0.6,0.65,0.7,0.75,0.8 \
    --sweep task.a_exo=3,5 \
    --no-plots --top 5

This keeps θ near the winner while testing whether a_exo can pull the vortex timing back toward 105ms without wrecking D. Those two interact directly — a_exo sets the strength of the burst, threshold sets how long the race runs before it matters. You want that 10-combo grid to tell you if there's a (θ, a_exo) pair that keeps D≈0.28 while pulling t_vortex down from 200ms toward 105ms.

Res:
θ=0.75, a_exo=3 wins clearly — but it's a Pyrrhic victory. The score (0.297) looks great on paper, but look at why: it nailed t_vortex (106ms ≈ 105 target) while D collapsed to essentially zero (3.75e-10). The scorer is rewarding the timing hit but the vortex depth is completely dead. The model found a configuration where the timing is right but there's no actual dip — which is mechanistically wrong. It's the opposite problem from θ=0.7 (which had perfect D but broken timing).

The a_exo=5 pattern is revealing. Across every θ, raising a_exo from 3→5 made scores worse, often dramatically. Higher exogenous burst strength is hurting rather than helping at this stage — likely because it's overpowering the race and distorting A. Leave a_exo at 3 and don't revisit until Phase 3.

What you're seeing is a θ vs D tradeoff:

θ too low (0.6–0.7): t_vortex blows out to 200ms, timing broken
θ too high (0.75–0.8): D collapses toward zero, vortex disappears
The sweet spot — if it exists at this smoke scale — is somewhere in between, and neither 0.7 nor 0.75 fully gets there

The core tension: D and t_vortex are being controlled by the same knob (θ) in opposite directions right now. This is a signal that you need a second degree of freedom to decouple them. tau_exo (the timescale of the exogenous burst) is the natural candidate — it governs how quickly the burst decays, which affects timing without necessarily killing depth the way θ does.

Kept a_exo=3.

Searched tau_exo (originally 30). 

python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.a_exo=3 \
    --sweep task.threshold=0.7,0.72,0.75,0.78 \
    --sweep task.tau_exo=10,20,30 \
    --no-plots --top 6

What this sweep really tells you: the D≈0 problem at θ=0.75 isn't something τ_exo can rescue. The burst decays at the right timescale but the threshold is still killing the vortex depth. This is a structural tension in the current architecture — the next lever to try is sigma_noise, which governs trial-to-trial variability and directly produces D in the tachometric curve (more noise → deeper apparent dip via trial averaging). That's a Phase 3 knob per the OPT_README, but you may need to peek at it earlier given the D problem is showing up so persistently.

Alternatively, before going there — are you confident the D extraction is correct? With frac_crossed ≈ 0.50 everywhere, the curve near the vortex is computed from very thin trial counts and the dip can genuinely vanish as a measurement artifact rather than a model failure.