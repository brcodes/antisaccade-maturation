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
