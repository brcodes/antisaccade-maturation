# Gameplan v2 Delta

What changed from v1 to v2. Excludes implementation decisions made by the user (rank, mechanism, straight-through, direction 1, reference repo) — those are recorded in the v2 header table. This document covers only new scientific and technical content.

---

## 1. rPT: corrected definition and source of variance

**What changed**: v1 described rPT as emergent but then said to "generate trials at evenly spaced rPTs" as if rPT were a directly settable parameter. This was contradictory. v2 corrects and unifies the definition throughout.

**The correct statement**: rPT = t_commit − t_cue, computed post-hoc per trial. It is never imposed. Gap (cue onset time) is the controllable simulation parameter. The network's threshold crossing determines t_commit, and rPT falls out of their interaction. This exactly replicates the experimental logic: in Zhu et al. 2024, rPT = RT − gap, where RT is the monkey's saccade onset time.

**Why the continuous gap approximation is valid**: Zhu et al. 2024 used three discrete gap conditions (overlap, zero-gap, 100 ms, with 200 ms in some sessions). The tachometric curve you are fitting is pooled across all conditions. The continuous 0–350 ms gap sweep used in Salinas et al. 2019 and adopted here samples the same rPT space more densely and is the established CAS modeling convention. The approximation is not a limitation — it is an improvement for fitting the pooled curve.

**Where the vortex actually comes from in Zhu et al.**: The paper states that 95% of cue viewing times in the overlap and zero-gap conditions exceed 150 ms — those conditions contribute only to the asymptotic portion of the curve. The vortex and recovery region are driven by the 100 ms gap condition, where rPT variance arises from trial-to-trial RT variability (not gap variation). The model must reproduce this RT variability through its stochastic initial state (see Addendum 2 below).

**Tachometric curve generation** (corrected in Section 3.4): Trials are binned by their **emergent rPT** (t_commit − t_cue), not by the imposed gap. This mirrors exactly how Zhu et al. 2024 computed the tachometric curve.

---

## 2. Stochastic initial hidden state (correlated build-up variability)

**What changed**: v1 included a generic `noise(t)` term in the RNN dynamics but did not address trial-to-trial RT variability as a separate structural concern. v2 adds a stochastic initial hidden state with a shared noise component.

**Why it matters**: In the CAS model (Salinas et al. 2019, Table 1), initial build-up rates b_C and b_A are drawn from a joint Gaussian with inter-plan correlation ρ = 0.95. This high correlation means when the monkey is in a "fast" motor state, both plans build up fast together, producing the narrow unimodal RT distribution seen empirically. Without this correlated variability, the model produces a vortex that is too sharp and RT distributions that are too narrow to match Zhu et al. 2024 Fig. 3D.

**Implementation** (new in Section 2.6):

```python
h(0) = h_mean(m) + σ_shared * ε_shared * 1_N + σ_private * ε_private
```

where ε_shared ~ N(0,1) is a scalar shared across all units (producing correlated variability analogous to the ρ = 0.95 inter-plan correlation) and ε_private ~ N(0, I_N) is independent per unit. Free parameters: σ_shared (~0.3) and σ_private (~0.1).

**Expected effect**: Produces realistic bimodal rPT distributions for correct and error trials, with error mode around rPT = 115–121 ms and correct mode around rPT = 180–198 ms, matching Zhu et al. 2024 Fig. 3D.

**Biological motivation**: Correlated trial-to-trial variability in motor preparation is well-established in FEF and SC recordings. The shared component here corresponds to global fluctuations in motor urgency or arousal that simultaneously modulate both competing motor plans.

---

## 3. rPT-weighted behavioral loss

**What changed**: v1 weighted all rPT bins equally in the training loss. v2 adds an explicit rPT-dependent weighting that concentrates training pressure on the developmentally informative region.

**Why it matters**: The developmental difference between young and adult curves lives almost entirely in the 70–200 ms rPT range (vortex + recovery). Uniform weighting allows the optimizer to fit the asymptote well while underweighting the vortex, producing solutions that look reasonable in aggregate but miss the key developmental signal. Additionally, the Zhu et al. 2024 data is itself unevenly distributed across rPT — the 100 ms gap condition overrepresents short-to-medium rPTs relative to long ones. Equal weighting in the loss would implicitly mismatch this empirical distribution.

**Implementation** (new in Section 3.2):

```python
def rpt_weight(rPT_ms):
    if 70 <= rPT_ms <= 200:
        return 3.0    # vortex + recovery: highest weight
    elif 200 < rPT_ms <= 300:
        return 1.0    # asymptote
    else:
        return 0.5    # guessing tail

loss = (weights * per_trial_loss).mean()
```

The specific weight values (3.0 / 1.0 / 0.5) are a reasonable starting point; tune if the optimizer consistently underfits the vortex while overfitting the asymptote or vice versa.

**Connection to empirical data**: The 100 ms gap condition in Zhu et al. 2024 is the sole source of short-rPT trials. Its trials are approximately uniformly distributed across rPTs within the condition (because RT variability is the source, not gap variation). Upweighting the 70–200 ms range effectively increases the influence of the 100 ms gap condition's contribution to the loss, matching the empirical structure of the data.

---

## 4. Updated empirical target parameters

**What changed**: v1 used estimated parameter values from predecessor literature. v2 replaces these with values read directly from Zhu et al. 2024 Fig. 3B and the 95% confidence intervals reported in the text.

| Parameter | v1 value (Young / Adult) | v2 value (Young / Adult) | Source |
|---|---|---|---|
| t_rise (ms) | 170 / 145 | 155 / 140 | Zhu et al. 2024 Fig. 3B, text (95% CI) |
| t_vortex (ms) | 110 / 105 | 105 / 106 | Zhu et al. 2024 Fig. 3B, text (95% CI) |
| A (asymptote) | 0.80 / 0.92 | 0.92 / 0.97 | Zhu et al. 2024 Fig. 3B visual |
| D (depth) | 0.50 / 0.42 | 0.28 / 0.27 | Zhu et al. 2024 Fig. 3B; D = 0.5 − min |

Key corrections: t_rise was too high in v1 (especially young: 170 → 155 ms). The vortex locations are nearly identical between young and adult (105 vs. 106 ms) — this was not captured in v1, which had them at 110 and 105 ms. The asymptotes were substantially underestimated (0.80/0.92 → 0.92/0.97). Vortex depth is much shallower than v1 assumed (0.28/0.27, not 0.50/0.42). These corrections meaningfully change the loss landscape: the vortex depth difference between conditions is now negligible (matching the figure), and the developmental difference is concentrated entirely in t_rise and A.

**Note on D**: chance is fixed at 0.5 in the parametric model (Zhu et al. 2024 Methods set A_L = 0.5). Use this convention consistently regardless of the empirical baseline (~0.52 adult, ~0.48 young), which reflects binomial noise.

---

## 5. Response deadline and lapse mechanism (explicit)

**What changed**: v1 mentioned lapses as a free parameter without connecting them to the 450 ms deadline in Zhu et al. 2024. v2 makes the mechanism explicit.

**Implementation** (new in Section 2.4): If neither output crosses θ by T_max = 450 ms, force a response at T_max with direction = argmax(z(T_max)). This is a lapse. Lapse probability λ is initialized to ~0.02 following Salinas et al. 2019 Table 1 (high luminance condition).

**Why it matters**: Lapses are the primary contributor to the difference in asymptotic performance between young and adult (the paper attributes the asymptote difference to "lapses in performance and less consistent ability to apply the antisaccade rule"). Without an explicit deadline mechanism, the model has no way to reproduce the asymptote difference — it will either always reach threshold (no lapses) or fail systematically. The deadline + lapse mechanism is required for the asymptotic A parameter to be a meaningful free quantity.
