# Antisaccade Maturation Modeling: Implementation Gameplan

**Reference paper**: Zhu, J., Zhou, X.M., Constantinidis, C., Salinas, E., & Stanford, T.R. (2024).
"Parallel signatures of cognitive maturation in primate antisaccade performance and prefrontal activity."
*iScience*, 27(8), 110488. https://doi.org/10.1016/j.isci.2024.110488

**Predecessor papers required for context** (all freely available):
- Salinas, E. et al. (2019). "Voluntary and involuntary contributions to perceptually guided saccadic choices resolved with millisecond precision." *eLife* 8:e46359.
- Goldstein, B.R. et al. (2022). "Exogenous capture accounts for fundamental differences between pro- and antisaccade performance." *eLife* 11:e76964.
- Zhou, X. et al. (2016). "Behavioral response inhibition and maturation of goal representation in prefrontal cortex after puberty." *PNAS* 113(12):3353–3358.
- Liu, Y.H. et al. (2021). "Emergence of prefrontal neuron maturation properties by training recurrent neural networks in cognitive tasks." *iScience* 24(10):103178.
- Mastrogiuseppe, F. & Ostojic, S. (2018). "Linking connectivity, dynamics and computations in low-rank recurrent neural networks." *Neuron* 99(3):609–623. [Key reference for low-rank RNN architecture]

---

## Scientific goal

Build a low-rank recurrent neural network (LR-RNN) conditioned on a scalar "maturation state" that:

1. Is trained only on behavioral tachometric curves from young and adult monkeys (behavioral fit)
2. Spontaneously reproduces — without being fit to them — the neural signatures observed in FEF/dlPFC during antisaccade task performance, specifically: faster transition from cue-evoked to goal-directed spatial signal (SI) with maturation
3. Reveals what internal dynamical differences between young and adult network states produce these behavioral and neural changes
4. Is biologically plausible in its architecture, dynamics, and interpretation of computational units

The key hypothesis: fitting behavior alone is sufficient to determine internal dynamics, such that SI(t, rPT) changes emerge as a consequence of the behavioral fit rather than as an additional constraint.

---

## Section 1: Task structure and data targets

### 1.1 The compelled antisaccade task (trial structure)

Each trial proceeds as follows (values from Zhu et al. 2024 / Salinas et al. 2019):

```
[Fixation: 500–800 ms]
  → Go signal (fixation point offset): t = 0
  → Gap period: 0–350 ms (variable, uniform random)
  → Cue onset: t = Gap (bright stimulus at ±10° from fixation, left or right)
  → Response window: monkey must make a saccade within 450 ms of go signal
  → Correct response: saccade to the location OPPOSITE the cue (antisaccade)
  → Error: saccade toward the cue (prosaccade / reflexive capture)
```

**Raw processing time (rPT)** = time from cue onset to saccade commitment.
This is the key independent variable. rPT = RT (reaction time from go signal) − Gap.
Because RT is roughly fixed by motor urgency and gap is variable, manipulating gap
indirectly samples different rPTs. Trials with short gaps → short rPTs (motor plan commits
before cue is fully processed). Trials with long gaps → long rPTs (cue processed before commitment).

**Note for implementation**: In simulation, rPT is directly controlled, which is cleaner
than simulating the gap manipulation. Generate trials at evenly spaced rPTs from 0 to 300 ms.

### 1.2 Target tachometric curve shape

The empirical tachometric curve (proportion correct vs. rPT) has four distinguishable phases:

| Phase | rPT range | Behavior | Mechanism |
|---|---|---|---|
| Guessing | 0–~90 ms | ~50% correct (chance) | Motor plan commits before cue processed |
| Vortex | ~90–130 ms | Drops toward 0% | Exogenous burst captures saccade toward cue |
| Recovery | ~130–200 ms | Steep sigmoidal rise | Endogenous goal signal overtakes exogenous |
| Asymptote | >200 ms | Plateau below 100% | Noise and rule-adherence failures |

**Young monkeys**: rise point ~160–180 ms, asymptote ~75–85%, vortex deeper
**Adult monkeys**: rise point ~130–150 ms, asymptote ~85–95%, vortex slightly shallower

**Fit targets** (summary statistics, not raw curves):
- `t_rise`: rPT at which accuracy crosses 75% (rise point)
- `A`: asymptotic accuracy (plateau value)
- `t_vortex`: rPT of minimum accuracy (vortex location)
- `D`: vortex depth (minimum accuracy value)

These are extracted from empirical data using analytical sigmoid fits as in Goldstein et al. (2022).
The implementation should reproduce these four statistics for each maturation condition.

### 1.3 Target neural signature (FEF/dlPFC spatial signal)

The FEF/dlPFC spatial selectivity index SI(t, rPT) is defined as:

```
SI(t, rPT) = [R_preferred(t, rPT) − R_antipreferred(t, rPT)] /
              [R_preferred(t, rPT) + R_antipreferred(t, rPT)]
```

where R_preferred is the mean firing rate of neurons whose preferred spatial field
matches the cue location (i.e., cue-side neurons), and R_antipreferred is the mean
firing rate of neurons whose preferred field is opposite to the cue (goal-side neurons).

**Interpretation of sign**:
- SI > 0: population represents cue location (stimulus-driven state)
- SI < 0: population represents goal location (goal-driven state)
- SI crosses zero: transition from exogenous to endogenous control

**Developmental finding** (Zhu et al. 2024, Figures 3, 7):
- In young monkeys: SI at short rPTs is strongly positive (cue-locked); transition to negative SI
  occurs late and incompletely for short rPTs
- In adult monkeys: SI transitions to negative (goal-directed) more rapidly as a function of rPT;
  the spatial signal co-varies with behavioral accuracy more tightly
- The PFC spatial signal and behavioral accuracy follow consistent trajectories as functions of rPT
  in adult-stage but not young-stage monkeys

**Critically, this is a PREDICTION target, not a fit target.** The model is fit only to
the behavioral tachometric curves. SI(t, rPT) is then extracted from model hidden units
and compared to the empirical neural data post-hoc.

---

## Section 2: Model architecture

### 2.1 Overview: Low-rank RNN conditioned on maturation state

The model is a continuous-time low-rank RNN (Mastrogiuseppe & Ostojic 2018) with:
- A maturation scalar `m ∈ [0, 1]` (0 = young, 1 = adult) as an additional input
- Input encoding of task events (go signal, cue onset, cue location, task rule)
- Hidden recurrent dynamics that generate competing motor plans
- A readout layer that maps hidden activity to a saccade direction decision

The low-rank constraint (rank R = 2 or 3) is chosen for three reasons:
(a) it is biologically motivated — low-dimensional dynamics emerge from structured connectivity
(b) it makes the learned dynamics interpretable in terms of a small number of modes
(c) it prevents the network from using arbitrary high-dimensional solutions that have no
    neural correlate

**Architecture dimensions**:
```
N_input  = 5  (go, cue_left, cue_right, task_rule=antisaccade, maturation_scalar)
N_hidden = 200  (recurrent units; enough for low-rank structure to emerge)
N_rank   = 2   (connectivity rank; start here, increase to 3 if underfitting)
N_output = 2   (saccade direction: toward_cue, toward_goal)
```

### 2.2 Continuous-time RNN dynamics

The hidden state h(t) evolves as:

```
τ * dh/dt = -h(t) + W_rec * r(t) + W_in * u(t) + noise(t)
r(t) = φ(h(t))   [firing rate; φ = tanh or ReLU]
```

where:
- `τ = 10 ms` (membrane time constant; biologically motivated for cortical neurons)
- `W_rec` = recurrent weight matrix, constrained to rank R
- `W_in` = input weight matrix
- `u(t)` = input vector at time t
- `noise(t)` ~ N(0, σ²) with σ as a free parameter

**Low-rank constraint on W_rec**:
```
W_rec = Σ_{k=1}^{R} m_k * n_k^T    [outer product of "input" and "output" modes]
```
where `m_k` and `n_k` are N-dimensional vectors. This is the Mastrogiuseppe & Ostojic
parameterization. The key interpretable quantity is the overlap between these modes and
the input/output axes — this defines the low-dimensional manifold of network dynamics.

**Why low-rank is bioplausible**: Structured low-rank connectivity can arise from
Hebbian learning and reflects the organization of long-range cortico-cortical projections.
The modes m_k and n_k can be interpreted as "selectivity" and "mixed selectivity" dimensions,
analogous to the spatial and temporal coding axes seen in PFC recordings.

### 2.3 Input encoding

Time is discretized at dt = 1 ms. Simulate T = 500 ms per trial.

Input vector u(t) at each timestep:

```python
u(t) = [
    go_signal(t),          # 1.0 from t=0 onward (step function)
    cue_left(t),           # 1.0 from t=Gap onward if cue is left
    cue_right(t),          # 1.0 from t=Gap onward if cue is right
    task_rule,             # constant 1.0 (antisaccade throughout)
    maturation_scalar,     # constant m ∈ [0,1] for the entire trial
]
```

The cue inputs are step functions — the cue appears instantaneously at t = Gap and stays on.
The go signal precedes the cue by the gap duration, so the network must initiate motor preparation
before knowing the cue location.

**Biological note on cue input**: In the real brain, the cue onset produces a transient
burst in visual cortex and superior colliculus. This can be approximated by adding a
brief exponential pulse (peak at cue onset, decay τ_exo ~ 30–50 ms) to the cue input,
rather than a step function. This more faithfully reproduces the exogenous burst timing
that generates the vortex. This is the key element Salinas et al. 2019 added to the
ART model to produce the vortex. Include it:

```python
cue_exo(t) = A_exo * exp(-(t - t_cue) / τ_exo) * (t >= t_cue)   # exogenous burst
cue_sustained(t) = 1.0 * (t >= t_cue)                             # sustained cue
u_cue(t) = cue_exo(t) + cue_sustained(t)                          # total cue input
```

where A_exo and τ_exo are free parameters (biologically: A_exo ~ 2–5, τ_exo ~ 30 ms).

### 2.4 Output and decision rule

The output layer maps hidden state to a two-dimensional motor plan:

```
z(t) = W_out * r(t)
z = [z_cue, z_goal]   (unnormalized log-odds of each saccade direction)
```

Decision: the network commits to the saccade direction when either z_cue or z_goal
first crosses a fixed threshold θ. This is the race-to-threshold mechanism, consistent
with the ART model. After commitment, the output is fixed.

**rPT in the simulation**: rPT is the time of threshold crossing minus the cue onset time.
Trials where the threshold is crossed before cue onset have rPT < 0 and are guesses.

For training, instead of a hard threshold (non-differentiable), use a softmax readout
integrated over time and compute a binary cross-entropy loss:

```
p_correct(t) = softmax(z(t))[goal]
Loss_behavior = -Σ_t w(t) * log(p_correct(t))
```

where w(t) is a weighting function that peaks at the time of threshold crossing
(estimated from the current network's activity on each trial). Alternatively, use
a leaky accumulator with a soft commitment mechanism.

### 2.5 Maturation conditioning mechanism

The maturation scalar `m` enters the network in three ways:

1. **As a direct input** (described above in u(t)) — simplest, ensures m is available at every timestep
2. **As a modulator of the initial hidden state**: h(0) = f(m) where f is a learned
   linear function. This allows maturation to set a "preparatory state" before trial onset,
   analogous to the pre-stimulus goal bias seen in mature monkeys.
3. **As a modulator of the exogenous burst amplitude**: A_exo(m) = A_0 * (1 - α*m),
   so that mature networks receive a weaker exogenous burst. This is biologically motivated
   by the Zhu et al. 2024 finding that the cue-evoked spatial signal is weaker in adult monkeys.
   This should be explored as an alternative after the base model is trained.

**Start with mechanism 1 only**. After training, mechanisms 2 and 3 can be ablated
to determine which aspect of maturation conditioning drives the behavioral and neural changes.

---

## Section 3: Training procedure

### 3.1 Training data

Generate synthetic training data from the **empirical tachometric curve summary statistics**,
not from raw monkey spike trains. This is both practical (raw data may not be available)
and principled (you are fitting the behavioral phenomenon, not the specific monkey's RTs).

For each maturation condition (young: m=0, adult: m=1), define target tachometric curves
using the analytical parametric form from Goldstein et al. (2022):

```
TC(rPT) = A * Φ((rPT - t_rise) / σ_rise)      # sigmoidal rise
         - D * Φ((rPT - t_vortex) / σ_vortex)   # gaussian-subtracted vortex
```

where Φ is the cumulative normal distribution, and the parameters are:

| Parameter | Young | Adult | Notes |
|---|---|---|---|
| A (asymptote) | 0.80 | 0.92 | From Zhu et al. 2024 Fig. 3–4 |
| t_rise (ms) | 170 | 145 | Rise point at 75% correct |
| σ_rise (ms) | 25 | 20 | Width of sigmoidal rise |
| t_vortex (ms) | 110 | 105 | Vortex location |
| D (vortex depth) | 0.50 | 0.42 | Below-chance magnitude |
| σ_vortex (ms) | 15 | 15 | Width of vortex dip |

**If raw monkey data is available** (contact Salinas/Stanford/Constantinidis labs or check OSF/Zenodo
for Zhu et al. 2024 data deposit), use the actual empirical tachometric curves per subject.
The parametric approach above is a faithful approximation for initial implementation.

Generate N_trials = 5000 per rPT bin × 30 rPT bins = 150,000 trials per maturation condition.
In practice, the network is trained on batches of trials with randomized rPTs, cue locations,
and maturation scalars (sampling m from {0, 1} or a continuous uniform distribution).

### 3.2 Loss function

Total loss:

```
L = L_behavior + λ_reg * L_regularization
```

**Behavioral loss** (primary):
```
L_behavior = Σ_{m ∈ {0,1}} Σ_{rPT} MSE(TC_model(rPT, m), TC_target(rPT, m))
```

where TC_model(rPT, m) is the fraction of trials where z_goal crosses threshold first,
computed over a mini-batch of trials at each rPT value.

In practice, use **summary statistic loss** (more stable than curve-level MSE):
```
L_behavior = MSE(t_rise_model, t_rise_target)
           + MSE(A_model, A_target)
           + MSE(t_vortex_model, t_vortex_target)
           + MSE(D_model, D_target)
```
computed separately for m=0 and m=1 and summed.

**Regularization**:
```
L_reg = ||W_rec||_F^2   (Frobenius norm; prevents weight explosion)
      + ||r(t)||^2      (activity regularization; keeps firing rates in biological range)
```

λ_reg = 1e-4 to 1e-3; tune by monitoring whether hidden activity stays in [0, 1] for tanh units.

**No neural loss**. Do not include any term that penalizes deviation of SI(t, rPT)
from empirical FEF data. The SI comparison is a post-hoc prediction, not a training objective.

### 3.3 Optimization

- **Framework**: PyTorch (preferred for flexibility) or JAX (preferred for speed)
- **Optimizer**: Adam, lr = 1e-3, reduce on plateau
- **Batch size**: 256 trials per batch, stratified by rPT (ensure all rPT bins are represented)
- **Epochs**: 500–2000; early stopping based on validation loss
- **Curriculum**: Start training with only extreme rPT values (0–50 ms and 250–300 ms)
  where the task is easy (pure guessing vs. clear goal-driven). Gradually introduce
  the intermediate rPTs (the vortex region 90–160 ms) after 100 epochs. This prevents
  the network from collapsing to a trivial solution.
- **Initialization**: Xavier initialization for W_in and W_out; small random W_rec
  (scale 0.1 / sqrt(N)) to start in the stable linear regime.
- **Gradient clipping**: clip_norm = 1.0 (essential for RNNs with long sequences)

**Implementation note on differentiability**: The threshold-crossing decision rule
is non-differentiable. Use one of:
(a) **REINFORCE / policy gradient**: treat threshold crossing as a stochastic action;
    compute policy gradient loss. Noisy but works.
(b) **Straight-through estimator**: forward pass uses hard threshold, backward pass
    uses gradient of a soft proxy (sigmoid with temperature). Recommended.
(c) **Soft readout at a fixed time**: instead of threshold crossing, read out the network
    output at a fixed time T_commit (set to the mean RT from empirical data, ~250 ms),
    compute softmax, and use cross-entropy loss. Simplest and most stable; loses some
    mechanistic detail about RT distributions but is fine for tachometric curve fitting.

**Recommend option (c) for initial implementation**, then switch to (b) if you need
RT distribution predictions in addition to tachometric curves.

### 3.4 Generating the tachometric curve from the model

For each rPT value (0 to 300 ms, step 10 ms):
1. Run N=500 trials with cue onset at t = T_go + rPT (i.e., gap = rPT)
2. For each trial, record the direction of the output at time T_commit
3. TC_model(rPT) = fraction of trials where output = toward_goal (correct antisaccade)
4. Average over cue-left and cue-right trials (should be symmetric by task design)

---

## Section 4: Extracting FEF-analog spatial signal

This section details the post-training analysis pipeline for the neural prediction.

### 4.1 Defining FEF-analog units

After training, identify the subset of hidden units that have spatial tuning
analogous to FEF/dlPFC visuomotor neurons.

**Step 1: Run the "spatial tuning" analysis**

For each hidden unit i, compute its **spatial selectivity index at cue onset**:

```python
# Run 1000 trials with cue_left and 1000 trials with cue_right
# at a long rPT (e.g., 250 ms) to ensure goal-directed response
r_cue_left[i]  = mean activity of unit i in window [t_cue, t_cue+50ms] for cue_left trials
r_cue_right[i] = mean activity of unit i in window [t_cue, t_cue+50ms] for cue_right trials

spatial_tuning[i] = r_cue_left[i] - r_cue_right[i]
```

**Step 2: Select FEF-analog units**

Units with |spatial_tuning[i]| > threshold (e.g., top 50% of |spatial_tuning|)
are designated "spatially tuned" FEF analogs. These are the units that carry
a left-vs-right spatial signal, analogous to the visuomotor neurons recorded in
Zhu et al. 2024.

**Rationale**: This selection criterion mirrors how FEF/dlPFC neurons are selected
in electrophysiology experiments — researchers record from neurons that show
differential activity for stimuli in different spatial locations. Units with no
spatial tuning are equivalent to non-spatially-selective neurons that are filtered
out of the analysis.

**Left-preferring vs. right-preferring**: Split spatially tuned units into:
- Left-preferring: spatial_tuning[i] > 0 (more active for cue_left)
- Right-preferring: spatial_tuning[i] < 0 (more active for cue_right)

### 4.2 Computing SI(t, rPT)

For a given cue location (say, cue_right as the reference):
- "Preferred" = right-preferring units (fire more to the cue)
- "Anti-preferred" = left-preferring units (fire more to the goal)

```python
# For each rPT bin and each timestep t:
for rPT in rPT_bins:
    trials_cue_right = run_trials(cue_location='right', rPT=rPT, N=500)
    
    R_pref[t, rPT] = mean(r_right_preferring_units(t))   # cue-side neurons
    R_anti[t, rPT] = mean(r_left_preferring_units(t))    # goal-side neurons
    
    SI[t, rPT] = (R_pref[t, rPT] - R_anti[t, rPT]) / 
                 (R_pref[t, rPT] + R_anti[t, rPT] + ε)
```

**Alignment**: Compute SI aligned to three reference events (as in Zhu et al. 2024):
1. Go-signal onset (t = 0)
2. Cue onset (t = Gap)
3. Saccade onset (t = T_commit)

**The key prediction**: SI(t, rPT) should:
- Start near 0 (no spatial preference before go signal)
- Become positive near cue onset (exogenous cue response drives cue-side neurons)
- Transition to negative for long rPTs (goal-side neurons win as endogenous signal arrives)
- In the young (m=0) model: transition is slower, positive phase is longer
- In the adult (m=1) model: transition is faster, positive phase is shorter

### 4.3 Additional dynamical analyses (beyond SI)

After training, do not limit the analysis to SI. Examine:

**A. Population geometry (PCA)**

At each rPT and timepoint, project the full hidden population activity r(t) onto
the top principal components. Key questions:
- Does the network separate "cue" and "goal" states in different subspaces?
- Does maturation change the geometry (angle between cue and goal coding axes)?
- Is there a "transition manifold" that the network traverses in rPT space?

```python
# Stack activity across rPTs and timepoints
X = r(t, rPT)   # shape: [N_hidden, N_timepoints * N_rPT_bins]
U, S, V = PCA(X)
# Project onto top 3 PCs and visualize trajectories
```

**B. Goal-bias pre-activity**

Compute the mean hidden activity in the window before the go signal (t < 0).
Does the mature network (m=1) show higher preparatory activity in goal-direction units
relative to the young network (m=0)? This would correspond to the "pre-stimulus bias"
toward the goal direction observed in mature PFC.

**C. Temporal profile of mode activations**

For the low-rank RNN, the network dynamics are governed by projections onto the
modes m_k and n_k. Compute the time courses of these mode activations as a function
of rPT and maturation. This is the most mechanistically interpretable analysis:
it directly shows how the network's low-dimensional dynamics change with maturation.

**D. Dimensionality of the solution**

Compute participation ratio (PR) of the hidden population activity:
```
PR = (Σ_i λ_i)^2 / Σ_i λ_i^2
```
where λ_i are eigenvalues of the covariance matrix of r(t).
Does the mature network use a lower-dimensional or higher-dimensional solution?
This is an open question with biological precedent in both directions.

---

## Section 5: Maturation comparison analysis

### 5.1 Interpolating across maturation states

After training on m ∈ {0, 1}, evaluate the model at intermediate values m ∈ {0.1, 0.3, 0.5, 0.7, 0.9}.

For each intermediate m:
1. Generate tachometric curve → extract t_rise, A, t_vortex, D
2. Compute SI(t, rPT) → extract SI at key rPT bins (short: 100ms, medium: 150ms, long: 200ms)
3. Compute PC trajectories and mode activations

Plot all summary statistics as a function of m. The prediction:
- t_rise decreases monotonically with m
- A increases with m
- Peak SI at short rPT decreases with m (less cue capture)
- SI transition point shifts left with m (faster goal-direction flip)

Monotonic changes across m validate that the maturation scalar is doing meaningful
computational work, not just fitting two disconnected solutions.

### 5.2 Ablation analysis

To determine which aspect of the maturation mechanism drives each behavioral/neural change,
run ablations:

**Ablation 1**: Set m=1 but use m=0 initial hidden state → does t_rise change?
**Ablation 2**: Set m=1 but use m=0 exogenous burst amplitude → does vortex depth change?
**Ablation 3**: Retrain with m as input to only the initial state (not ongoing) → is behavior the same?

These ablations isolate the contribution of preparatory state, ongoing recurrent dynamics,
and exogenous gain modulation to the maturational changes.

### 5.3 Cross-prediction validation

The bidirectional validation:

**Direction 1 (primary, recommended)**: Fit to behavioral tachometric curves only.
Predict SI(t, rPT). Compare predicted SI to empirical SI from Zhu et al. 2024.
Success metric: Pearson correlation between predicted and empirical SI time courses
at matched rPT bins. Target r > 0.7 for the adult condition.

**Direction 2 (secondary, optional)**: Retrain a separate model instance using
SI(t, rPT) as the loss target (fit the neural dynamics, not behavior). Then generate
tachometric curves forward. Check whether behavioral summary statistics are recovered.
Compare t_rise, A from this model against the behavior-fit model.

If both directions give consistent parameter estimates (similar learned W_rec structure,
similar mode activations), this is strong evidence that the behavioral and neural changes
share a common mechanistic explanation in the model.

---

## Section 6: Implementation code structure

### 6.1 File organization

```
antisaccade_model/
├── task/
│   ├── trial_generator.py      # generates trial sequences with specified rPT, gap, cue, m
│   ├── tachometric_targets.py  # parametric tachometric curves for young/adult
│   └── task_params.py          # all task constants (timings, thresholds, etc.)
├── model/
│   ├── lrrnn.py                # low-rank RNN implementation
│   ├── readout.py              # output layer and decision rule
│   └── model_params.py         # architecture hyperparameters
├── training/
│   ├── train.py                # main training loop
│   ├── losses.py               # behavioral loss and regularization
│   └── curriculum.py           # rPT curriculum scheduler
├── analysis/
│   ├── tachometric_analysis.py # extract TC from model, fit summary statistics
│   ├── spatial_signal.py       # compute SI(t, rPT) from hidden units
│   ├── geometry.py             # PCA, participation ratio, mode activations
│   └── maturation_sweep.py     # evaluate model at intermediate m values
├── visualization/
│   ├── plot_tc.py              # tachometric curve plots
│   ├── plot_si.py              # SI time courses matched to Zhu et al. Fig. 7
│   └── plot_geometry.py        # PC trajectories, mode activation plots
└── experiments/
    ├── run_behavior_fit.py     # main experiment: fit to TC, predict SI
    ├── run_ablations.py        # ablation analyses
    └── run_neural_fit.py       # secondary: fit to SI, predict TC
```

### 6.2 Key dependencies

```
python >= 3.10
torch >= 2.0           # or jax >= 0.4
numpy >= 1.24
scipy >= 1.10          # for curve fitting and statistics
matplotlib >= 3.7      # for visualization
scikit-learn >= 1.2    # for PCA, cross-validation
tqdm                   # training progress
wandb                  # optional: experiment tracking
```

For low-rank RNN specifically, reference implementations are available at:
- https://github.com/fmastrogiuseppe/LowRank (Mastrogiuseppe & Ostojic 2018; Python 2 but translatable)
- Or implement directly following equations 1–5 in that paper

### 6.3 Pseudocode for core training loop

```python
# Initialize
model = LRRNN(N=200, rank=2, N_input=5, N_output=2)
optimizer = Adam(model.parameters(), lr=1e-3)
tc_targets = load_tachometric_targets()   # young and adult parametric curves

for epoch in range(N_epochs):
    rPT_bins = curriculum_scheduler(epoch)  # start narrow, expand to full range
    
    for batch in dataloader(rPT_bins, batch_size=256):
        # batch contains: rPT, cue_location, maturation_scalar (m=0 or m=1)
        
        # Forward pass: generate trial dynamics
        h, r, z = model(batch.u)   # u is input sequence [T x batch x N_input]
        
        # Soft decision at T_commit
        p_correct = softmax(z[T_commit])[..., GOAL_IDX]
        
        # Behavioral loss
        tc_model = compute_tachometric_curve(p_correct, batch.rPT)
        loss_beh = tc_loss(tc_model, tc_targets, batch.m)
        
        # Regularization
        loss_reg = λ_reg * (norm(model.W_rec) + mean(r**2))
        
        loss = loss_beh + loss_reg
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Validation
    if epoch % 50 == 0:
        tc_young = eval_tachometric_curve(model, m=0)
        tc_adult = eval_tachometric_curve(model, m=1)
        log_summary_statistics(tc_young, tc_adult)

# Post-training analysis
si_young = compute_spatial_signal(model, m=0)
si_adult = compute_spatial_signal(model, m=1)
compare_to_empirical(si_young, si_adult, zhu_et_al_2024_data)
```

---

## Section 7: Biological plausibility checklist

The following constraints should be verified after training:

| Constraint | Check | Biological motivation |
|---|---|---|
| Firing rates in plausible range | mean r(t) ∈ [1, 80] spk/s | Cortical neuron range |
| No persistent saturation | max tanh activation < 0.95 | Avoid unrealistic max firing |
| Exogenous burst timing | E-signal peaks at ~100 ms rPT | Consistent with SC/FEF cue response |
| Endogenous ramp onset | G-signal rises from go-signal | Consistent with FEF preparatory activity |
| Vortex timing | Model vortex at 90–130 ms rPT | Matches empirical attentional vortex |
| Goal pre-bias increases with m | h(0) in goal direction larger for m=1 | Consistent with mature PFC presetting |
| SI sign flip timing decreases with m | SI crosses zero earlier for m=1 | Core prediction from Zhu et al. 2024 |
| Mutual inhibition present | Negative cross-coupling in W_rec | SC/FEF inhibitory interneurons |

---

## Section 8: Failure modes and mitigations

| Failure mode | Symptom | Mitigation |
|---|---|---|
| No vortex in TC | Monotonic sigmoid, no below-chance dip | Check exogenous burst amplitude A_exo; increase it |
| Vortex too deep / early | SI collapses to 0% across rPT range | Reduce A_exo; increase τ_exo (slower burst) |
| m=0 and m=1 produce identical TCs | Network ignores maturation scalar | Increase capacity of m → h pathway; use mechanism 2 or 3 |
| SI does not flip sign | Always positive (always cue-locked) | Endogenous goal bias too weak; check W_in for task_rule input |
| Gradient explosion | Loss diverges after epoch 10 | Reduce lr to 1e-4; verify gradient clipping |
| Network degenerates to rank-1 | One mode dominates, no spatial discrimination | Increase rank to 3; add orthogonality regularization on modes |
| rPT interpolation not smooth | Abrupt TC change between m=0 and m=1 | Train on continuous m ~ Uniform(0,1), not just {0,1} |

---

## Section 9: Expected timeline

| Phase | Duration | Deliverable |
|---|---|---|
| 1. Task + target TC implementation | 1–2 days | trial_generator.py, tachometric_targets.py |
| 2. LR-RNN + training loop | 2–3 days | lrrnn.py, train.py, losses.py |
| 3. Initial training run (m ∈ {0,1}) | 1 day | Trained model checkpoint |
| 4. Behavioral validation | 1 day | TC plots, summary statistics table |
| 5. SI extraction + neural prediction | 1–2 days | SI time courses, comparison to Zhu et al. |
| 6. PCA, geometry, mode analysis | 1–2 days | Population geometry plots |
| 7. Maturation sweep (intermediate m) | 1 day | Summary statistic vs. m curves |
| 8. Ablations | 1–2 days | Mechanism attribution |
| 9. Optional: neural-fit direction | 2 days | Cross-prediction validation |

**Total estimated time**: 10–16 days for a complete first implementation.

---

## Section 10: Key open questions the model will answer

1. **Is the tachometric curve shape sufficient to constrain SI dynamics?**
   (Does behavioral-only fit predict the neural SI change?)

2. **What is the geometric signature of maturation in the network's state space?**
   (Does maturation change the angle between cue and goal coding axes? Their dimensionality?)

3. **Does maturation primarily change preparatory state, recurrent dynamics, or input gain?**
   (Answered by ablations in Section 5.2)

4. **Are SI dynamics the primary dynamical difference, or are they downstream of something more fundamental?**
   (Answered by the comprehensive hidden-unit analysis in Section 4.3)

5. **Can intermediate maturation states (m ∈ (0,1)) produce behavioral profiles that match
   individual monkey trajectories through development?**
   (Testable if individual monkey data from Zhu et al. 2024 are available)
