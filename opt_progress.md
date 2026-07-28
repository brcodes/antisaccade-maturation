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

Evaluated that. Gpt 5.4mini, home repo.
Claude response (first optimization conversation, Web):

This reframes where you are. The problem isn't θ vs τ_exo — it's that the exogenous capture mechanism isn't strong enough to pull proportion-correct below chance at early rPTs, which is the biological signature of the vortex.

what governs whether that below-chance dip forms at all?

The three levers that directly control whether the curve crosses below chance:

a_exo — amplitude of the exogenous burst. This is your primary suspect. You fixed it at 3 across sweeps 3 and 4 after seeing a_exo=5 hurt in sweep 3, but that finding was confounded: sweep 3 was still using wrong targets, and the a_exo=5 runs may have been penalized for other reasons (A distortion, frac_crossed effects). With correct targets and τ_exo now confirmed at 30, a_exo deserves a fresh isolated look.

sigma_noise — trial-to-trial noise. More noise widens the distribution of outcomes at each rPT, which deepens the apparent dip in the averaged curve. This is Phase 3 normally but you're effectively already there given D=0 is your blocker.

commit_temp — softness of the commitment decision. A very sharp commitment (low temp) means once the accumulator tips toward the exogenous direction it fully commits, which can deepen the vortex. Worth knowing the current default.

Your current defaults are commit_temp = 0.2 and sigma_noise = 0.1, defined in task_params.py:57 and task_params.py:61.


One thing to verify in code before running: confirm that a_exo scales the burst subtractively at the accumulator (i.e. it pushes the decision variable toward the wrong response, not just adds noise). If it's implemented as an additive perturbation in the wrong direction it should work; if it's something else, the mechanism may not be capable of producing below-chance performance regardless of magnitude.

Gpt-5.4mini

It is not implemented as a subtractive accumulator term. In trial_generator.py:63, a_exo multiplies an exponential burst, that burst is added to a sustained cue drive, and the result is written into either the left or right cue input channel after cue onset. That means the model gets an extra positive side-specific input, not a built-in negative push toward the wrong choice.

The decision variable itself is formed later in lrrnn.py:43 from learned recurrent and input weights, and the readout in readout.py:1 is linear. So the only way a_exo can produce below-chance performance is indirectly, through the learned weights turning that side-specific burst into a bias toward the reflexive/wrong response. There is no code-level guarantee that the burst is “wrong-direction” in the accumulator; that behavior has to emerge from training.

Claude:

The vortex cannot exist at initialization — it has to be learned. The burst is a positive side-specific input, and whether that drives the accumulator toward the wrong response depends entirely on what the recurrent weights learn to do with it. At 50 smoke epochs with a fresh random init, the weights almost certainly haven't learned to invert the burst into a wrong-direction bias yet. So D≈0 and vortex_depth≥0 may not mean the mechanism is broken — it may just mean the model hasn't trained long enough to discover that the burst should produce capture errors.

This reframes the Phase 0 problem entirely. You've been interpreting D=0 as a dynamic-range / architectural failure, but it's more likely a training horizon problem. The plumbing is correct; the weights just need more epochs to learn the inversion.

ran:

python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 \
    --set task.tau_exo=30 \
    --set task.a_exo=3 \
    --sweep train.epochs=50,150,300,500 \
    --no-plots --top 4

goal- Watch for when vortex_depth_m0 first goes negative. If it crosses below chance somewhere between 150–500 epochs, you have your answer: the smoke epoch budget was the bottleneck, not the parameters. If it's still ≥0 at 500 epochs, then the weight learning genuinely isn't discovering the capture mechanism and you need to look at the loss landscape — specifically whether the behavioral loss has any gradient signal pointing toward wrong-direction responses at short rPT, or whether the loss is blind to the vortex until A and t_rise are already fit.

One other thing worth checking in the loss function: is D explicitly in the behavioral loss, or is it only implicit through the tachometric curve shape? If the loss only penalizes A and t_rise directly and treats D as emergent, the gradient toward vortex formation may be very weak early in training.

Results:
Claude:

The big news: the vortex IS there at all epoch counts. Every single run has vortex_depth < 0 — the curve is genuinely crossing below chance. The D≈0 problem you've been chasing was purely a Gaussian fit failure on a real-but-shallow dip, not an absent mechanism. The architecture is working.

The score being non-monotone in epochs is the key diagnostic. 50 wins on score but for the wrong reason (shallow dip → D=0 → low D penalty + good t_vortex by accident). 300 is the genuine best. This means the smoke epoch budget of 50 has been giving you misleading rankings throughout sweeps 1–4 — configs that scored well may have just been accidentally producing narrow unfittable dips.

Recommended next step — re-run your best θ/a_exo configs at 300 epochs to get honest rankings:

python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set train.epochs=300 \
    --set task.tau_exo=30 \
    --sweep task.threshold=0.7,0.72,0.75,0.78 \
    --sweep task.a_exo=3,5,8 \
    --no-plots --top 6


Res:
θ=0.75, a_exo=3 holds as winner — and now it's a trustworthy ranking. These are honest 300-epoch scores, not 50-epoch artifacts.

The most important pattern in this sweep: t_vortex has essentially converged to ~115–118ms across every viable config. It's no longer a free variable — the model is finding the same vortex timing regardless of θ or a_exo. That's actually good news architecturally; it means the exogenous burst timescale (τ_exo=30) is the dominant controller of when the dip occurs, and that part is working. The ~10ms gap from target (105ms) is small and likely closeable with τ_exo fine-tuning later.

The two remaining problems are A and D, and they're linked. A is sitting around 0.78–0.81 across the board (target 0.92) — the model isn't reaching high enough asymptotic accuracy. D is 0.13–0.17 at best (target 0.28) — the vortex depth is real but shallow. Both of these are undertrained symptoms, not wrong-parameter symptoms. The model knows where to put the vortex and roughly when to rise, but hasn't fully learned the magnitude of either.

a_exo=5 at θ=0.7 is interesting second-place result: t_rise=152ms (nearly perfect), t_vortex=118ms, D=0.167 — a genuinely competitive config that might overtake the winner with more epochs.

The Phase 1 gate is now in reach. You have a stable, non-degenerate training config (θ=0.75, a_exo=3, τ_exo=30)

Recap:
The model knows 
1) where to put the vortex (D)
2) when to put the vortex (t_vortex)
3) and roughly how much to rise (A)
   
but it hasn't fully learned the magnitude of vortex or extent of accuracy rise (D,A), and t_vortex is a little off.

Now:
So Learning Rate search: push A and D toward their correct maxima, minima, but don't push out of convergence basin you have discovered.

Later:
τ_exo fine-tuning for t_vortex

python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 \
    --set task.a_exo=3 \
    --set task.tau_exo=30 \
    --set train.epochs=300 \
    --sweep train.lr=1e-4,3e-4,1e-3,3e-3 \
    --no-plots --top 4


Res:
You need to go higher, not lower.

1e-3 wins on score (your current default, so no change needed there), but look at what 3e-3 is producing: A=0.835 (best in the sweep), D=0.362 (only config to overshoot the target, meaning it's genuinely fitting the vortex depth), and t_vortex=110ms (closest to 105 yet). It scores slightly worse only because t_rise undershoots to 134ms. That's a timing calibration issue, not a capacity issue — 3e-3 is learning more, just slightly fast on the rise.

The sweet spot is between 1e-3 and 3e-3. 3e-3 has better A and D but overshoots t_rise; 1e-3 has better t_rise but weaker A and D. Run a fine search in that gap:

python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 \
    --set task.a_exo=3 \
    --set task.tau_exo=30 \
    --set train.epochs=300 \
    --sweep train.lr=1e-3,1.5e-3,2e-3,3e-3 \
    --no-plots --top 4

If 2e-3 lands with A>0.85, D>0.20, and t_rise closer to 155ms, that's your Phase 1 locked LR. Then one more check: -- grad_clip — 3e-3's higher loss suggests some gradient noise that clipping might stabilize, which could let you run at the higher LR more cleanly.

Res:
LR landscape is non-monotone and the gap between 1e-3 and 3e-3 doesn't contain a clean optimum. That's unusual and suggests training is landing in different basins at different LRs rather than smoothly interpolating.

The real story from sweeps 7+8 combined: 1e-3 and 3e-3 are both legitimate but solving different parts of the problem:

1e-3: good t_rise (159ms), weak A (0.799), weak D (0.143)
3e-3: good A (0.835), good D (0.362), bad t_rise (134ms) and bad D m1 (0.069)

Neither is strictly better. They're on different tradeoff curves. The issue isn't LR alone — it's that D m1 is consistently low or nan across everything. The adult maturation state (m=1) is lagging badly on vortex depth in almost every sweep. That asymmetry between m0 and m1 on D is worth noting as a systemic issue before proceeding.

SYSTEMIC ISSUE: systematic asymmetry between m=0 and m=1 fit (revealing bi-objective tug-of-war)

Phase 1 call: LR is sufficiently explored. The basin structure isn't going to get cleaner with more fine-grained LR search — you've found that 1e-3 and 3e-3 are your two viable operating points with different tradeoffs. Lock 1e-3 as the conservative choice (better score, more stable) and move to Phase 2 capacity.

Per OPT_README §8 scale-up, the next step is extending epochs before touching n_hidden:

python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 \
    --set task.a_exo=3 \
    --set task.tau_exo=30 \
    --set train.lr=1e-3 \
    --sweep train.epochs=300,500,750,1000 \
    --no-plots --top 4

The question is whether more epochs at the locked LR can push A toward 0.92 and D toward 0.28 simultaneously, or whether the model has hit a capacity ceiling at n_hidden=64 that more training time can't fix. If A plateaus below 0.87 regardless of epoch count, that's your signal to raise n_hidden.

Res: 
Capacity ceiling confirmed. A m0 wanders between 0.769 and 0.799 across 300→1000 epochs with no upward trend — training longer isn't moving the needle. The model at n_hidden=64 simply doesn't have enough representational room to push A toward 0.92. This is the Phase 2 signal the OPT_README was waiting for.

Possible concern at 500 eps:500 epochs is a bad basin — t_vortex crashes to 70ms and D collapses to ~0, same pathology seen in sweep 5. There's something unstable happening around the 500-epoch mark specifically at this LR. Could be the warmup schedule ending and the optimizer hitting a saddle point. Worth keeping in mind but not worth chasing now.

D m1 is also telling a story. It's highest at 300 epochs (0.322) and deteriorates with more training — the adult vortex gets worse as epochs increase. Model has reached capacity at n_hidden=64, and can't jointly maintain both m=0, m=1 states and starts lowering m1 representation quality in order to seek m0 gains.

Time to raise n_hidden. Per OPT_README §8, go from 64 toward 200 in steps:

python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 \
    --set task.a_exo=3 \
    --set task.tau_exo=30 \
    --set train.lr=1e-3 \
    --set train.epochs=300 \
    --sweep model.n_hidden=64,100,128,200 \
    --no-plots --top 4

Res:
Every larger model fell into the dead-race false minimum. This is the same pathology as LR=3e-4 in sweep 7 — frac_crossed=0.00, loss artificially low, curve completely degenerate. n_hidden=128 additionally diverged (loss=11, A=0.5, D=0.6 are all hallmarks of a collapsed readout, not a real fit).

The diagnosis is straightforward: LR=1e-3 was tuned for n_hidden=64. Larger models have more parameters and a different loss landscape curvature — the same LR is effectively too small relative to the gradient scale, and the optimizer slides into the dead-race basin before the race even gets established. This is exactly the OPT_README §8 warning: "Re-confirm the LR from Phase 1 still trains stably after raising n_hidden — usually yes; nudge down if not." In your case you need to nudge up, not down.

NOTE FOR FUTURE: MODEL SIZE INCREASES MAY REQUIRE LEARNING RATE DECREASES (IF GREATER COST SIZE CAUSES INSTABILITY) OR INCREASES (IF GREATER COST SIZE DOES NOT OUTWEIGH THE NOW-LARGER PRIOR INIT INERTIA).

Next sweep — re-run Phase 1 LR check at n_hidden=100:

Search UP from 1e-3 x5:

python -m antisaccade_model.experiments.opt_behavior_fit --preset smoke \
    --set task.threshold=0.75 \
    --set task.a_exo=3 \
    --set task.tau_exo=30 \
    --set train.epochs=300 \
    --set model.n_hidden=100 \
    --sweep train.lr=1e-3,2e-3,3e-3,5e-3 \
    --no-plots --top 4

Goal:
Then find the lowest LR that keeps frac_crossed alive, and if 100 shows a genuine A improvement over 64 you step to 128, then 200. One size at a time.

The big headline: score=0.132 at n_hidden=100 vs 0.531 at n_hidden=64. That's a genuine capacity improvement — A jumped from 0.799 to 0.850, t_vortex hit 108ms (nearly on target), and t_rise=161ms is solid. The capacity ceiling was real and raising n_hidden broke through it.