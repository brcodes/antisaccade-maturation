# Antisaccade Maturation Modeling: Implementation Gameplan v2

**Reference paper**: Zhu, J., Zhou, X.M., Constantinidis, C., Salinas, E., & Stanford, T.R. (2024).
"Parallel signatures of cognitive maturation in primate antisaccade performance and prefrontal activity."
*iScience*, 27(8), 110488. https://doi.org/10.1016/j.isci.2024.110488

**Predecessor papers required for context**:
- Salinas, E. et al. (2019). "Voluntary and involuntary contributions to perceptually guided saccadic choices resolved with millisecond precision." *eLife* 8:e46359. [Primary CAS model reference; open access]
- Goldstein, B.R. et al. (2022). "Exogenous capture accounts for fundamental differences between pro- and antisaccade performance." *eLife* 11:e76964. [Parametric curve fitting reference; open access]
- Zhou, X. et al. (2016). "Behavioral response inhibition and maturation of goal representation in prefrontal cortex after puberty." *PNAS* 113(12):3353–3358. [Developmental behavioral reference]
- Liu, Y.H. et al. (2021). "Emergence of prefrontal neuron maturation properties by training recurrent neural networks in cognitive tasks." *iScience* 24(10):103178. [Closest methodological precedent for RNN-maturation approach]
- Mastrogiuseppe, F. & Ostojic, S. (2018). "Linking connectivity, dynamics and computations in low-rank recurrent neural networks." *Neuron* 99(3):609–623. [Low-rank RNN architecture; reference implementation at https://github.com/fmastrogiuseppe/LowRank]

---

## Locked implementation decisions

The following choices have been made and are fixed throughout this document:

| Decision | Choice | Rationale |
|---|---|---|
| RNN rank | R = 2 | Biologically motivated; sufficient for two competing spatial modes |
| Maturation conditioning | Mechanism 1 only (m as direct ongoing input) | Simplest; ablations later can test mechanism 2/3 |
| Differentiability | Straight-through estimator (option b) | Preserves threshold crossing as the biological decision event |
| Validation direction | Direction 1 only (fit behavior, predict SI) | Most insight-generating; SI comparison is post-hoc |
| Reference implementation | https://github.com/fmastrogiuseppe/LowRank | Use as architectural foundation |

---

## Scientific goal

Build a rank-2 low-rank recurrent neural network (LR-RNN) conditioned on a scalar maturation state `m` that:

1. Is trained **only** on behavioral tachometric curves from young and adult monkeys (behavioral fit; no neural loss)
2. Spontaneously reproduces — without being fit to them — the neural signatures observed in FEF/dlPFC: faster transition from cue-evoked to goal-directed spatial signal (SI) with maturation (Zhu et al. 2024, Fig. 7)
3. Reveals what internal dynamical differences between young and adult network states produce these behavioral and neural changes
4. Is biologically plausible in its architecture, dynamics, and interpretation of computational units

**Key hypothesis**: Fitting behavior alone is sufficient to constrain internal dynamics, such that SI(t, rPT) changes emerge as a consequence of the behavioral fit rather than as an additional constraint. If they do not, this is equally meaningful — it implies the behavioral and neural changes are not explained by the same mechanistic shift.

---

## Section 1: Task structure and data targets

### 1.1 The compelled antisaccade task (trial structure)

The experiment in Zhu et al. 2024 used three discrete gap conditions: overlap (cue before fixation offset), zero-gap (simultaneous), and 100 ms gap (fixation offset precedes cue by 100 ms). A 200 ms gap was also used in some sessions. These are not continuously varying — the rPT variance within each condition arises from trial-to-trial variability in the monkey's RT, not from gap manipulation.

Critically, the vortex and recovery region of the tachometric curve are driven almost entirely by the 100 ms gap condition. In the overlap and zero-gap conditions, the paper reports that 95% of cue viewing times exceed 150 ms — these conditions contribute only to the asymptotic portion. The pooled tachometric curve you are fitting combines all conditions, but the developmentally informative shape (vortex depth, rise point) comes from the 100 ms gap trials.

For implementation, we use a **continuously varying gap from 0–350 ms**, following Salinas et al. 2019. This is a deliberate departure from Zhu et al.'s discrete conditions but correctly samples the same rPT space and is the established CAS modeling convention (Salinas et al. 2019; Goldstein et al. 2022). The pooled tachometric curve you are fitting is itself an aggregate across conditions; a continuous gap implementation samples that aggregate space more densely and is strictly preferable for curve fitting.

**Trial structure** (values from Zhu et al. 2024 and Salinas et al. 2019):

```
[Fixation: 500–800 ms]
  → Go signal (fixation point offset): t = 0
  → Gap period: drawn uniformly from 0–350 ms each trial
  → Cue onset: t = t_go + gap  (bright stimulus, left or right, ±10° eccentricity)
  → Response deadline: saccade must occur within 450 ms of go signal
  → Correct response: saccade to location OPPOSITE the cue (antisaccade)
  → Error: saccade toward the cue (prosaccade / reflexive capture)
```

**rPT definition**: rPT = t_commit − t_cue, computed post-hoc per trial from the threshold crossing time. rPT is **never imposed directly**. You set gap (cue onset time), the network's threshold crossing determines t_commit, and rPT falls out of their interaction. This replicates the experimental logic exactly: in Zhu et al. 2024, rPT = RT − gap, where RT is the monkey's saccade onset time and gap is the experimental condition.

**rPT variance in simulation**: Trial-to-trial rPT variation in the model arises from two sources, both required for biological fidelity (see Section 2.6):
1. Gap variation across trials (the dominant source in Salinas 2019's human experiment)
2. Stochastic initial hidden state producing RT variability at fixed gap (the dominant source in Zhu 2024's monkey experiment within each gap condition)

Both sources are present in the model. The continuous gap sweep samples rPT space broadly; the stochastic initial state produces RT variability that blurs rPT within each gap value, as in the monkey data.

### 1.2 Target tachometric curve parameters

The empirical tachometric curve (proportion correct vs. rPT) has four distinguishable phases:

| Phase | rPT range | Behavior | Mechanism |
|---|---|---|---|
| Guessing | 0–~90 ms | ~50% correct (chance) | Motor plan commits before cue is processed |
| Vortex | ~90–130 ms | Drops below chance | Exogenous burst captures saccade toward cue |
| Recovery | ~130–200 ms | Steep sigmoidal rise | Endogenous goal signal overtakes exogenous |
| Asymptote | >200 ms | Plateau below 100% | Lapses and rule-adherence failures |

**Empirical fit targets** — values read directly from Zhu et al. 2024, Fig. 3B and Fig. 4B–D. All four summary statistics are computed from the model's simulated tachometric curve post-hoc, **not from sigmoid parameters**:

| Parameter | Young | Adult | Source | Definition |
|---|---|---|---|---|
| t_rise (ms) | 155 [154,157] | 140 [139,141] | Zhu Fig. 3B (95% CI) | rPT at which TC crosses 75% correct |
| t_vortex (ms) | 105 [97,112] | 106 [101,110] | Zhu Fig. 3B (95% CI) | rPT of minimum accuracy |
| A (asymptote) | ~0.92 | ~0.97 | Zhu Fig. 3B visual | Mean accuracy for rPT > 200 ms |
| D (vortex depth) | ~0.28 | ~0.27 | Zhu Fig. 3B visual + chance=0.5 | 0.5 − minimum accuracy |
| σ_rise (ms) | ~25 | ~15 | Visual estimate | SD of cumulative normal fit to recovery |
| σ_vortex (ms) | ~25 | ~20 | Visual estimate | SD of Gaussian subtracted for vortex |

**Important**: t_rise is defined as the rPT at which the fitted tachometric curve crosses 75% correct. Compute it from the simulated curve as:
```python
t_rise = rPT_bins[np.argmin(np.abs(tc_curve - 0.75))]
```
Do **not** use the sigmoid midpoint. The sigmoid midpoint equals the 75% crossing only when the asymptote is 100% and the floor is 0%, which is never the case here. Using the midpoint will produce a systematic mismatch between model and empirical targets.

Similarly:
```python
t_vortex = rPT_bins[np.argmin(tc_curve)]
D        = 0.5 - np.min(tc_curve)        # relative to fixed chance=0.5
A        = np.mean(tc_curve[rPT_bins > 200])
```

**Note on D**: chance is fixed at 0.5 in the parametric model (following Zhu et al. 2024 and Salinas et al. 2019, which set A_L = 0.5 in the fitting function). The empirical baselines deviate slightly (adult ~0.52, young ~0.48) but these reflect binomial noise around true chance, not a meaningful signal. Use 0.5 as the chance reference consistently.

**Note on σ parameters**: σ_rise and σ_vortex are not directly reported in Zhu et al. 2024. Values above are visual estimates. Goldstein et al. 2022 (eLife, open access) reports explicit σ values for adult monkeys on the same task and should be consulted for better-constrained estimates of these parameters.

**Individual monkey data**: Fig. 4A of Zhu et al. 2024 shows tachometric curves for each of the four monkeys separately. If individual data become available (contact stanford@wakehealth.edu per the paper's data availability statement), fit the model to individual curves rather than the pooled curve. This enables testing whether intermediate m values reproduce individual monkey trajectories.

### 1.3 Target neural signature (FEF/dlPFC spatial signal)

The FEF/dlPFC spatial signal (SROC in Zhu et al. 2024; SI here) quantifies which spatial location — cue or goal — is currently dominant in the PFC population. In Zhu et al. 2024 it is computed via ROC analysis on presaccadic spike counts (50 ms window before saccade onset) in visuomotor (VM) neurons only.

**SROC definition in Zhu et al. 2024** (Fig. 7E):
- SROC > 0.5: population more active for cue-in-RF than saccade-into-RF → cue-dominant state
- SROC < 0.5: population more active for saccade-into-RF → goal-dominant state
- SROC = 0.5: no spatial preference

**Empirical values** (Zhu et al. 2024, Fig. 7E, 8A–B):

| Condition | Short rPT (70–170 ms) SROC | Long rPT (170–300 ms) SROC |
|---|---|---|
| Young | 0.58 [0.55,0.61] | 0.53 [0.52,0.55] |
| Adult | 0.54 [0.51,0.56] | 0.47 [0.46,0.48] |

Key findings: (1) adult SI crosses from cue-dominant to goal-dominant at long rPTs (SROC < 0.5); young SI never fully crosses (remains above 0.5 at all rPTs). (2) The adult neurometric curve co-varies tightly with the behavioral tachometric curve as a function of rPT; the young curve does not.

**This is a prediction target, not a fit target.** The model is fit only to behavioral tachometric curves. SI(t, rPT) is extracted post-hoc and compared to the empirical SROC values above.

**Analog SI definition for the model** (Section 4):
```
SI(t, rPT) = [R_preferred(t, rPT) − R_antipreferred(t, rPT)] /
              [R_preferred(t, rPT) + R_antipreferred(t, rPT) + ε]
```
where preferred/antipreferred are defined by the unit's spatial tuning. Sign convention matches Zhu et al.: SI > 0 = cue-dominant, SI < 0 = goal-dominant.

---

## Section 2: Model architecture

### 2.1 Overview

The model is a continuous-time rank-2 low-rank RNN (Mastrogiuseppe & Ostojic 2018) conditioned on a scalar maturation state `m ∈ [0,1]` (0 = young, 1 = adult). Use the reference implementation at https://github.com/fmastrogiuseppe/LowRank as the architectural foundation, adapting it for the antisaccade task structure and maturation conditioning described here.

The low-rank constraint (R = 2) is chosen because:
- Biologically motivated: low-dimensional dynamics emerge from structured cortico-cortical connectivity (Mastrogiuseppe & Ostojic 2018)
- Interpretable: the two rank-2 modes can be identified with cue-encoding and goal-encoding spatial dimensions, directly analogous to the two competing populations in the CAS model
- Constrained: prevents high-dimensional solutions with no neural correlate

**Architecture dimensions**:
```
N_input  = 5    (go, cue_left, cue_right, task_rule=antisaccade, maturation_scalar)
N_hidden = 200  (recurrent units)
N_rank   = 2    (fixed)
N_output = 2    (z_cue, z_goal)
```

### 2.2 Continuous-time RNN dynamics

The hidden state h(t) evolves as:

```
τ * dh/dt = -h(t) + W_rec * r(t) + W_in * u(t) + noise(t)
r(t) = φ(h(t))   [φ = tanh]
```

Parameters:
- `τ = 10 ms` (membrane time constant; biologically motivated for cortical neurons)
- `W_rec = Σ_{k=1}^{2} m_k * n_k^T` (rank-2 connectivity; outer products of learned mode vectors)
- `W_in` = input weight matrix (N_hidden × N_input)
- `noise(t)` ~ N(0, σ²_unit) per unit per timestep; σ_unit is a free parameter

**Low-rank parameterization** (following Mastrogiuseppe & Ostojic 2018, Eqs. 1–5):
The mode vectors m_k and n_k are the learnable parameters of W_rec. The overlap between these modes and the input/output axes defines the low-dimensional manifold of network dynamics and is the primary object of post-training analysis.

### 2.3 Input encoding

Time is discretized at dt = 1 ms. Simulate T = 500 ms per trial (covers fixation, gap, and response window).

```python
u(t) = [
    go_signal(t),          # step: 1.0 from t=0 onward
    cue_left(t),           # exogenous burst + sustained step if cue is left
    cue_right(t),          # exogenous burst + sustained step if cue is right
    task_rule,             # constant 1.0 (antisaccade rule)
    maturation_scalar,     # constant m ∈ [0,1] for the entire trial [MECHANISM 1]
]
```

**Exogenous burst on cue input** (critical for vortex generation; Salinas et al. 2019):

In the real brain, cue onset produces a transient burst in SC and visual cortex (~76 ms after cue onset for high-luminance stimuli; Salinas et al. 2019, Table 1) before the endogenous antisaccade plan can develop. This is what generates the vortex. Implement as:

```python
cue_exo(t)       = A_exo * exp(-(t - t_cue) / τ_exo) * (t >= t_cue)
cue_sustained(t) = 1.0 * (t >= t_cue)
u_cue(t)         = cue_exo(t) + cue_sustained(t)
```

Free parameters: A_exo (~2–5), τ_exo (~30 ms). Initial values from Salinas et al. 2019 Table 1 (high luminance, pooled): ERI onset ~76 ms, duration ~24 ms, exogenous acceleration a_EX = 0.96 AU/ms². These provide biologically grounded initialization priors.

**Maturation as direct ongoing input (Mechanism 1)**: m is concatenated to u(t) as a constant at every timestep. This is the simplest conditioning mechanism and is the only one used for training. It allows the network to modulate its recurrent dynamics continuously as a function of maturation level, without requiring a maturation-dependent initial state or gain modulation.

### 2.4 Output and decision rule

```
z(t) = W_out * r(t)     [W_out: 2 × N_hidden]
z = [z_cue, z_goal]
```

A saccade is committed when either z_cue or z_goal first crosses threshold θ. This is the race-to-threshold mechanism, directly analogous to the ART model (Salinas et al. 2019). The winning output determines saccade direction.

**rPT per trial**: `rPT = t_commit − t_cue`, where t_commit is the timestep of threshold crossing. This is computed post-hoc; it is never imposed as a simulation parameter.

**Response deadline**: If neither output crosses θ by T_max = 450 ms (the monkey's deadline; Zhu et al. 2024), the trial is classified as a lapse. Lapses are handled by forcing a response at T_max with direction determined by argmax(z(T_max)). Lapse probability λ is a free parameter initialized to ~0.02 (Salinas et al. 2019, Table 1).

### 2.5 Straight-through estimator (decision B — locked)

The threshold crossing is non-differentiable. Use the straight-through estimator:

**Forward pass** (hard, mechanistically faithful):
```python
def find_threshold_crossing(z_history, theta=1.0):
    # z_history: [T x 2]
    for t in range(len(z_history)):
        if z_history[t, 0] > theta or z_history[t, 1] > theta:
            winner = int(torch.argmax(z_history[t]))
            return winner, t
    # No crossing before deadline: lapse
    winner = int(torch.argmax(z_history[-1]))
    return winner, len(z_history) - 1
```

**Backward pass** (soft proxy at t_commit, gradient flows):
```python
def compute_loss(z_history, target, theta=1.0, tau_temp=temperature):
    with torch.no_grad():
        _, t_commit = find_threshold_crossing(z_history, theta)
    # Soft readout AT t_commit — gradients flow through z_history[t_commit]
    z_at_commit = z_history[t_commit]
    p = torch.softmax(z_at_commit / tau_temp, dim=0)
    loss = -torch.log(p[target] + 1e-8)
    return loss, t_commit
```

**Temperature annealing**: Start τ = 5.0 (smooth gradients), anneal toward τ = 0.5 over the first 50% of training epochs. Too-rapid annealing causes gradient instability; too-slow means the soft proxy diverges from the hard decision and training loss no longer reflects the true tachometric curve.

**Why straight-through over soft readout (option c)**: The threshold crossing is the biological event. rPT = t_commit − t_cue, and t_commit is determined by the hard crossing. If a fixed T_commit is used instead, rPT in the model no longer means what rPT means in the data: it becomes the time between cue onset and an arbitrary snapshot, not the time between cue onset and saccade commitment. The tachometric curve's shape — particularly the vortex — depends on the distribution of crossing times across trials. A fixed snapshot collapses this distribution, approximating the curve shape but severing its mechanistic interpretation. The straight-through estimator preserves the mechanism at the cost of noisier gradients.

### 2.6 Trial-to-trial variability: stochastic initial hidden state

**Why this matters**: In Zhu et al. 2024, rPT variance within the 100 ms gap condition (the condition that drives the vortex) arises from RT variability — the monkey does not respond at a perfectly fixed latency. In the CAS model, this variability is captured by drawing initial build-up rates from a joint Gaussian distribution with high inter-plan correlation (ρ = 0.95; Salinas et al. 2019, Table 1). Without this, the model produces an unrealistically narrow RT distribution and a vortex that is too sharp.

**Implementation**: At the start of each trial, sample the initial hidden state from a learned distribution:

```python
h(0) = h_mean(m) + σ_init * ε,    ε ~ N(0, I)
```

where `h_mean(m)` is the maturation-dependent mean initial state (from mechanism 1's continuous m input; zero at trial onset before the go signal) and `σ_init` is a free scalar parameter controlling trial-to-trial variability.

To produce correlated variability across the two output dimensions (analogous to the ρ = 0.95 inter-plan correlation in Salinas et al. 2019), use a shared noise component:

```python
ε_shared  ~ N(0, 1)           # shared across all units
ε_private ~ N(0, I_N)         # independent per unit
h(0) = h_mean + σ_shared * ε_shared * 1_N + σ_private * ε_private
```

where `σ_shared` and `σ_private` are free parameters. High σ_shared relative to σ_private produces the correlated build-up variability seen in the CAS model. Initialize σ_shared ≈ 0.3, σ_private ≈ 0.1.

**Expected effect**: This produces realistic RT distributions with the bimodality observed in Zhu et al. 2024 (Fig. 3D) — a mode for captured errors around rPT = 115–121 ms and a mode for correct responses around rPT = 180–198 ms. Without this, those RT distributions will be too narrow to match.

---

## Section 3: Training procedure

### 3.1 Training data and target curve parameters

Training targets are the empirical summary statistics from Zhu et al. 2024 (Section 1.2). The analytical parametric curve form (following Zhu et al. 2024 Methods and Salinas et al. 2019, Eq. 2) is:

```
v(x) = max(L(x), R(x), 0)

L(x) = B + (A_L - B) / (1 + exp((x - C_L) / D_L))   # left (decreasing) side
R(x) = B + (A_R - B) / (1 + exp((x - C_R) / D_R))   # right (increasing) side
```

where A_L = 0.5 (chance, fixed), B = vortex minimum, A_R = asymptote, C_L = vortex center, C_R = rise center, D_L and D_R = widths. This is the exact fitting function used in Zhu et al. 2024.

Target parameter table (updated from empirical data):

| Parameter | Young | Adult | Source |
|---|---|---|---|
| A (asymptote, A_R) | 0.92 | 0.97 | Zhu Fig. 3B visual |
| t_rise (ms) | 155 | 140 | Zhu Fig. 3B, confirmed 95% CI |
| σ_rise (ms) | ~25 | ~15 | Visual estimate; verify vs. Goldstein 2022 |
| t_vortex (ms) | 105 | 106 | Zhu Fig. 3B, 95% CI |
| D (depth below chance) | 0.28 | 0.27 | Zhu Fig. 3B; D = 0.5 − min |
| σ_vortex (ms) | ~25 | ~20 | Visual estimate |

Generate N = 5000 trials per rPT bin × 30 bins = 150,000 trials per maturation condition for validation curve generation. During training, use mini-batches with stratified rPT sampling.

### 3.2 Loss function

```
L = L_behavior + λ_reg * L_regularization
```

**Behavioral loss** — summary statistic MSE (more stable than curve-level MSE):

```python
L_behavior = Σ_{m ∈ {0,1}} [
    w_rise    * MSE(t_rise_model,    t_rise_target[m])    +
    w_asym    * MSE(A_model,         A_target[m])          +
    w_vortex  * MSE(t_vortex_model,  t_vortex_target[m])  +
    w_depth   * MSE(D_model,         D_target[m])
]
```

**rPT-weighted loss** (Addendum 3): Weight the mini-batch loss by the empirical rPT distribution from Zhu et al. 2024. The 100 ms gap condition, which generates most of the vortex and recovery region (rPT ~70–200 ms), should receive higher weight than the tails. Concretely:

```python
# Empirical rPT weight function — upweight the developmentally informative region
def rpt_weight(rPT_ms):
    if 70 <= rPT_ms <= 200:
        return 3.0    # vortex + recovery: highest weight
    elif 200 < rPT_ms <= 300:
        return 1.0    # asymptote: normal weight
    else:
        return 0.5    # guessing tail: low weight

# Apply per trial in mini-batch
loss_per_trial = cross_entropy(p_correct, target)
weights = torch.tensor([rpt_weight(rpt) for rpt in batch.rPT])
L_behavior = (weights * loss_per_trial).mean()
```

This concentrates training pressure on the rPT region where the developmental difference lives (Zhu et al. 2024, Fig. 3B: curves differ primarily between 100–200 ms), preventing the optimizer from finding solutions that fit the asymptote well but miss the vortex/recovery transition.

**Summary statistic extraction** (computed every 50 epochs for validation, not every step):
```python
t_rise   = rPT_bins[np.argmin(np.abs(tc_curve - 0.75))]
t_vortex = rPT_bins[np.argmin(tc_curve)]
D        = 0.5 - np.min(tc_curve)
A        = np.mean(tc_curve[rPT_bins > 200])
```

**Regularization**:
```
L_reg = ||W_rec||_F^2 + ||r(t)||^2
```
λ_reg = 1e-4 to 1e-3.

**No neural loss.** SI(t, rPT) is never a training objective.

### 3.3 Optimization

- **Framework**: PyTorch (preferred) or JAX
- **Reference implementation**: https://github.com/fmastrogiuseppe/LowRank — adapt for antisaccade inputs and maturation conditioning
- **Optimizer**: Adam, lr = 1e-3, ReduceLROnPlateau
- **Batch size**: 256 trials, stratified by rPT bin and maturation condition
- **Epochs**: 500–2000; early stopping on validation loss
- **Curriculum**: Epochs 1–100: train only on extreme rPTs (0–50 ms guessing + 250–300 ms asymptote). Epochs 100+: introduce vortex region (90–160 ms) gradually. This prevents collapse to trivial solutions before the exogenous burst dynamics are established.
- **Initialization**: Xavier for W_in, W_out; small random W_rec (scale 0.1/√N); σ_shared = 0.3, σ_private = 0.1
- **Gradient clipping**: clip_norm = 1.0

**Temperature annealing schedule**:
```python
tau(epoch) = tau_start * (tau_end / tau_start) ** (epoch / N_anneal_epochs)
# tau_start = 5.0, tau_end = 0.5, N_anneal_epochs = N_epochs // 2
```

### 3.4 Generating the tachometric curve from the model

For each rPT value (0 to 300 ms, step 10 ms):

1. Sample gap values such that the expected rPT = target rPT (i.e., gap = target_rPT approximately, since RT ≈ gap + rPT)
2. For each of N = 500 trials: run forward pass, find t_commit via hard threshold crossing, compute rPT = t_commit − t_cue
3. Bin trials by their **emergent rPT** (not the imposed gap) into 10 ms bins
4. TC_model(rPT) = fraction of trials in each bin where winner = goal direction
5. Average over cue-left and cue-right trials

Note: steps 3–4 use the emergent rPT, not the imposed gap. This is essential for correctness — it mirrors exactly how Zhu et al. 2024 computed the tachometric curve (rPT = RT − gap, binned post-hoc).

---

## Section 4: Extracting FEF-analog spatial signal

### 4.1 Defining FEF-analog units

In Zhu et al. 2024, only visuomotor (VM) neurons are analyzed — units with both significant visual responses after cue onset and significant presaccadic activity. Selection criteria (from Zhu et al. 2024 Methods): (1) mean firing rate 50–150 ms post-cue significantly above baseline, (2) mean firing rate 250 ms post-go-signal significantly above baseline, (3) consistent visual response on antisaccade task.

For the model, identify analogous units post-training:

**Step 1: Spatial tuning index**
```python
# Run 1000 trials cue_left, 1000 trials cue_right at long rPT (250 ms)
r_cue_left[i]  = mean activity of unit i in window [t_cue, t_cue+50ms]
r_cue_right[i] = mean activity of unit i in window [t_cue, t_cue+50ms]
spatial_tuning[i] = r_cue_left[i] - r_cue_right[i]
```

**Step 2: Select spatially tuned units**
Top 50% by |spatial_tuning[i]| → "FEF-analog VM units."

**Step 3: Presaccadic activity criterion**
Retain only units with mean activity in the 250 ms presaccadic window significantly above baseline (t-test, p < 0.05 across trials). This mirrors the VM neuron selection in Zhu et al. 2024.

Split into left-preferring (spatial_tuning > 0) and right-preferring (spatial_tuning < 0).

### 4.2 Computing SI(t, rPT)

For cue_right as reference:
- Preferred = right-preferring units (cue-side)
- Anti-preferred = left-preferring units (goal-side)

```python
for rPT in rPT_bins:
    trials = run_trials(cue_location='right', rPT=rPT, N=500)
    R_pref[t, rPT] = mean(r_right_preferring(t))
    R_anti[t, rPT] = mean(r_left_preferring(t))
    SI[t, rPT] = (R_pref - R_anti) / (R_pref + R_anti + ε)
```

**Alignment**: Compute SI aligned to: (1) go-signal onset, (2) cue onset, (3) saccade onset (t_commit). The presaccadic window (−50 to 0 ms before t_commit) is the primary comparison window, matching Zhu et al. 2024's SROC computation.

**Key prediction**:
- SI starts near 0 (no spatial preference at fixation)
- SI rises positive at cue onset (exogenous burst → cue-side neurons dominate)
- At long rPTs: SI transitions to negative in adult (m=1); SI remains positive in young (m=0)
- Quantitative targets: adult long-rPT SI < 0.5 equivalent (SROC < 0.5); young long-rPT SI never crosses zero (SROC remains > 0.5)

### 4.3 Additional dynamical analyses

**A. Population geometry (PCA)**
Project hidden activity r(t, rPT) onto top principal components. Key questions: Does maturation change the angle between cue and goal coding axes? Is there a transition manifold in rPT space?

```python
X = r(t, rPT)   # [N_hidden, N_timepoints × N_rPT_bins]
U, S, V = PCA(X)
# Visualize PC trajectories across rPT and m
```

**B. Goal pre-bias**
Mean hidden activity in window t < 0 (before go signal). Prediction: mature network (m=1) shows higher preparatory activity aligned with goal direction.

**C. Mode activations**
For rank-2 RNN, compute time courses of projections onto modes m_k and n_k as a function of rPT and m. This is the most mechanistically interpretable analysis, directly linking learned connectivity to the CAS model's E and G populations.

**D. Participation ratio**
```
PR = (Σ_i λ_i)² / Σ_i λ_i²
```
Does maturation change the effective dimensionality of the solution? Open question with biological precedent in both directions.

---

## Section 5: Maturation comparison analysis

### 5.1 Interpolating across maturation states

Evaluate the trained model at m ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0}. For each m:
1. Generate tachometric curve → extract t_rise, A, t_vortex, D
2. Compute SI(t, rPT) at key rPT bins (short: 100 ms, medium: 150 ms, long: 200 ms)
3. Compute PC trajectories and mode activations

**Expected monotonic predictions**:
- t_rise decreases with m (adult faster)
- A increases with m (adult higher asymptote)
- Peak SI at short rPT decreases with m (less cue capture)
- SI zero-crossing shifts left with m (faster goal flip)

Monotonic behavior across m validates that the maturation scalar is doing meaningful computational work, not fitting two disconnected point solutions.

### 5.2 Ablation analysis

To isolate mechanism contributions:

**Ablation 1**: Set m=1, use m=0 noise parameters (σ_shared, σ_private) → does t_rise change? Tests whether RT variability structure drives the rise point.
**Ablation 2**: Set m=1, freeze A_exo to m=0 value → does vortex depth change? Tests whether exogenous gain drives the vortex difference.
**Ablation 3**: Retrain with m only as initial state modulator (not ongoing input) → is the same tachometric curve achievable? Tests sufficiency of mechanism 1 vs. mechanism 2.

### 5.3 Cross-prediction validation (Direction 1 — locked)

Fit to behavioral tachometric curves only. Post-hoc, extract SI(t, rPT) from the trained model and compare to Zhu et al. 2024 Fig. 7E.

**Success metric**: Pearson correlation between model SI time course and empirical SROC values at matched rPT bins. Target r > 0.7 for the adult condition. For young condition, the key test is whether model SI remains positive at long rPTs (SROC-equivalent > 0.5), matching the empirical finding that young PFC never fully transitions to goal-dominant.

**If the prediction fails**: This is itself a finding — it implies that the behavioral and neural changes do not share a common mechanistic explanation. Proceed to characterize what other dynamical signatures differ between m=0 and m=1 (Section 4.3 analyses).

---

## Section 6: Implementation code structure

### 6.1 File organization

```
antisaccade_model/
├── task/
│   ├── trial_generator.py      # generates trials: gap, cue, rPT, stochastic h(0)
│   ├── tachometric_targets.py  # empirical parametric curves (young/adult)
│   └── task_params.py          # all constants (timings, thresholds, deadline)
├── model/
│   ├── lrrnn.py                # rank-2 LR-RNN (adapted from fmastrogiuseppe/LowRank)
│   ├── readout.py              # output layer, threshold crossing, straight-through
│   └── model_params.py         # architecture hyperparameters
├── training/
│   ├── train.py                # main training loop with rPT-weighted loss
│   ├── losses.py               # behavioral loss, rPT weighting, regularization
│   └── curriculum.py           # rPT curriculum scheduler
├── analysis/
│   ├── tachometric_analysis.py # TC generation from model; summary stat extraction
│   ├── spatial_signal.py       # SI(t, rPT) from hidden units; VM unit selection
│   ├── geometry.py             # PCA, participation ratio, mode activations
│   └── maturation_sweep.py     # evaluate at intermediate m values
├── visualization/
│   ├── plot_tc.py              # tachometric curve plots vs. Zhu Fig. 3B
│   ├── plot_si.py              # SI time courses vs. Zhu Fig. 7E
│   └── plot_geometry.py        # PC trajectories, mode activations
└── experiments/
    ├── run_behavior_fit.py     # main: fit to TC, extract SI post-hoc
    └── run_ablations.py        # ablation analyses (Section 5.2)
```

### 6.2 Key dependencies

```
python >= 3.10
torch >= 2.0
numpy >= 1.24
scipy >= 1.10
matplotlib >= 3.7
scikit-learn >= 1.2
tqdm
```

Reference implementation: https://github.com/fmastrogiuseppe/LowRank (Python 2; translate to Python 3 following Eqs. 1–5 in Mastrogiuseppe & Ostojic 2018).

### 6.3 Core training loop pseudocode

```python
model = LRRNN(N=200, rank=2, N_input=5, N_output=2)
optimizer = Adam(model.parameters(), lr=1e-3)
targets = load_tachometric_targets()   # young and adult from Zhu et al. 2024
temperature = AnnealingSchedule(start=5.0, end=0.5, n_epochs=N_epochs//2)

for epoch in range(N_epochs):
    rPT_bins = curriculum_scheduler(epoch)

    for batch in stratified_dataloader(rPT_bins, m_values=[0,1], batch_size=256):
        # Sample stochastic initial state (Section 2.6)
        h0 = sample_initial_state(batch.m, sigma_shared, sigma_private)

        # Forward pass
        z_history = model(batch.u, h0)   # [T x batch x 2]

        # Straight-through: hard crossing (no grad), soft loss at t_commit (grad)
        tau = temperature(epoch)
        losses = []
        t_commits = []
        for i, z in enumerate(z_history.unbind(1)):
            with torch.no_grad():
                winner, t_commit = find_threshold_crossing(z, theta=1.0)
            loss_i, _ = compute_loss(z, target=1, tau_temp=tau)  # target=1: goal
            # Apply rPT-based weight
            rPT_i = t_commit - batch.t_cue[i]
            losses.append(rpt_weight(rPT_i) * loss_i)
            t_commits.append(t_commit)

        L_beh = torch.stack(losses).mean()
        L_reg  = lambda_reg * (model.W_rec.norm('fro')**2 + model.activity_norm())
        loss   = L_beh + L_reg

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Validation every 50 epochs
    if epoch % 50 == 0:
        for m in [0.0, 1.0]:
            tc = generate_tachometric_curve(model, m=m, N=500)
            stats = extract_summary_statistics(tc)
            log_vs_targets(stats, targets[m])

# Post-training analysis
for m in [0.0, 1.0]:
    si = compute_spatial_signal(model, m=m)
    compare_to_zhu_fig7(si, m=m)
```

---

## Section 7: Biological plausibility checklist

| Constraint | Check | Biological motivation | Source |
|---|---|---|---|
| Firing rates in range | mean r(t) ∈ [1, 80] spk/s | Cortical neuron range | General electrophysiology |
| No persistent saturation | max tanh < 0.95 | Avoid unrealistic max firing | — |
| Exogenous burst timing | Peak effect at ~76–105 ms post-cue | SC/FEF visual response latency | Salinas et al. 2019, Table 1 |
| Endogenous ramp onset | G-signal rises from go-signal | FEF preparatory activity | Zhu et al. 2024, Fig. 5 |
| Vortex timing | Model vortex at 105–106 ms rPT | Matches Zhu et al. 2024 Fig. 3B | Zhu et al. 2024 |
| RT distribution shape | Bimodal correct/error rPT distributions | Matches Zhu et al. 2024 Fig. 3D | Zhu et al. 2024 |
| Goal pre-bias increases with m | Stronger presaccadic goal activity for m=1 | Zhu et al. 2024: adult cue/saccade response balance more equal (3.3 vs 7.9 spk/s) | Zhu et al. 2024, Methods |
| SI sign flip in adult only | m=1 SI < 0 at long rPT; m=0 SI never < 0 | Zhu et al. 2024: adult SROC < 0.5 at long rPT; young SROC stays > 0.5 | Zhu et al. 2024, Fig. 7E |
| Mutual inhibition in W_rec | Negative cross-coupling between modes | SC/FEF inhibitory interneurons | Salinas et al. 2019 |
| Lapse rate ~2% | λ ≈ 0.02 | Zhu et al. 2024 adult error tail | Salinas et al. 2019, Table 1 |

---

## Section 8: Failure modes and mitigations

| Failure mode | Symptom | Mitigation |
|---|---|---|
| No vortex | Monotonic sigmoid, no below-chance dip | Increase A_exo; check exogenous burst τ_exo |
| Vortex too early/deep | Below-chance across wide rPT range | Reduce A_exo; increase τ_exo |
| m=0 and m=1 identical | Network ignores maturation scalar | Check W_in column for m; increase its learning rate |
| SI never flips sign | Always positive regardless of rPT or m | Endogenous goal signal too weak; check task_rule input weight |
| RT distribution too narrow | Vortex too sharp; no bimodality | Increase σ_shared; check stochastic h(0) implementation |
| Gradient explosion | Loss diverges < epoch 20 | Reduce lr to 1e-4; verify clip_norm=1.0 |
| Rank-1 collapse | One mode dominates; no spatial discrimination | Increase rank to 3; add orthogonality regularization on modes |
| Smooth interpolation fails | Abrupt TC change between m=0 and m=1 | Train on continuous m ~ Uniform(0,1), not just {0,1} |
| Loss insensitive to vortex | Good asymptote fit, poor vortex fit | Increase rPT weight in 70–200 ms region (Section 3.2) |

---

## Section 9: Expected timeline

| Phase | Duration | Deliverable |
|---|---|---|
| 1. Task + target TC + trial generator | 1–2 days | trial_generator.py, tachometric_targets.py |
| 2. LR-RNN + straight-through + stochastic h(0) | 2–3 days | lrrnn.py, readout.py |
| 3. Training loop with rPT-weighted loss + curriculum | 1–2 days | train.py, losses.py |
| 4. Initial training run (m ∈ {0,1}) | 1 day | Trained checkpoint |
| 5. Behavioral validation | 1 day | TC plots, summary statistic table vs. Zhu Fig. 3B |
| 6. RT distribution check | 0.5 days | Correct/error rPT distributions vs. Zhu Fig. 3D |
| 7. SI extraction + neural prediction | 1–2 days | SI time courses vs. Zhu Fig. 7E |
| 8. PCA, geometry, mode analysis | 1–2 days | Population geometry plots |
| 9. Maturation sweep (intermediate m) | 1 day | Summary statistics vs. m |
| 10. Ablations | 1–2 days | Mechanism attribution |

**Total estimated time**: 11–18 days for complete first implementation.

---

## Section 10: Key open questions the model will answer

1. **Is behavioral fitting sufficient to constrain the neural SI dynamics?**
   Does fitting the tachometric curve alone predict the SROC trajectory from Zhu et al. 2024 Fig. 7E? If yes: the behavioral and neural changes share a mechanistic explanation. If no: what additional dynamical signature distinguishes young and adult that SI does not capture?

2. **What is the geometric signature of maturation in state space?**
   Does maturation rotate the angle between cue and goal coding axes? Change their dimensionality? Produce a more separable manifold?

3. **Which aspect of maturation drives which behavioral change?**
   Ablations (Section 5.2) isolate contributions of RT variability structure, exogenous gain, and ongoing m-conditioning to t_rise, D, and A separately.

4. **Is SI downstream of something more fundamental, or is it the primary dynamical difference?**
   The Section 4.3 analyses (mode activations, PCA geometry, goal pre-bias) may reveal a deeper dynamical change of which SI is a consequence.

5. **Do intermediate m values reproduce individual monkey developmental trajectories?**
   Testable if individual monkey data from Zhu et al. 2024 are available (contact stanford@wakehealth.edu).
