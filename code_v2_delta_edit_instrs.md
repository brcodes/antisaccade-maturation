# code_v2_delta_edit_instrs

Work through each item in order. For each: run the check, report the result, then follow the contingency. Do not move to the next item until the current one is resolved.

---

## 0. Smoke test (run first)

```bash
python -m compileall antisaccade_model
python -c "from antisaccade_model.model.lrrnn import LRRNN; m = LRRNN(); print('import OK')"
python -c "
from antisaccade_model.task.trial_generator import sample_initial_state, build_inputs
import torch
h0 = sample_initial_state(batch_size=4, n_hidden=200, task=None)
print('h0 shape:', h0.shape)
print('h0 std:', h0.std().item())
"
```

All three must pass without error before proceeding.

---

## 1. Confirm 450 ms deadline fallback exists

**Where to look**: `find_threshold_crossing` in `readout.py` or wherever threshold crossing is computed.

**What to check**: The loop over timesteps must have a fallback that fires if no output crosses θ before T_max. It should return `argmax(z(T_max))` as the winner and `T_max` as t_commit.

**Contingency — not present**: Add it. One-liner at the end of `find_threshold_crossing`:
```python
# fallback if no crossing before deadline
return int(torch.argmax(z_history[-1])), len(z_history) - 1
```
Confirm it is reached on a sanity trial: run a trial with θ set artificially high (e.g. 1e6) and verify the function returns T-1 without error.

**Contingency — present**: No change needed. Note which file and line.

---

## 2. Check m-leakage in lapse branch

**What to check**: In the lapse branch forward pass, `m` is currently passed through normally. Run:

```python
import torch
from antisaccade_model.model.lrrnn import LRRNN
from antisaccade_model.task.trial_generator import build_inputs

model = LRRNN()
model.eval()

u0 = build_inputs(..., m=0.0, lapse_mask=True)
u1 = build_inputs(..., m=1.0, lapse_mask=True)

with torch.no_grad():
    z0 = model(u0, h0=torch.zeros(1, 200))
    z1 = model(u1, h0=torch.zeros(1, 200))

diff = (z0 - z1).abs().max().item()
print("Max lapse-branch output diff across m:", diff)
```

**Contingency — diff > 0.1**: m is leaking into the lapse branch — the network can partially recover rule information on lapse trials, causing λ(m) to be underestimated. Fix: zero the maturation channel in `build_inputs` when `lapse_mask=True`:
```python
if lapse_mask:
    u[:, maturation_channel_idx] = 0.0
```
Re-run check after fix; diff should drop toward 0.

**Contingency — diff ≤ 0.1**: Acceptable. Note the value and move on.

---

## 3. Verify hard vs. soft curve separation in loss

**Where to look**: `losses.py` — how summary-stat MSE and the rPT-weighted auxiliary term are computed.

**What to check**: Confirm:
- rPT-weighted auxiliary loss is computed from `p_goal_mix` / `soft_commit` (differentiable mixed-branch output)
- Summary-stat MSE (t_rise, A, t_vortex, D) is computed from the **hard** tachometric curve via full threshold-crossing evaluation, not from `p_goal_mix`

Look for where t_rise, A, t_vortex, D are extracted and confirm they come from a `generate_tachometric_curve` call using hard threshold crossings.

**Contingency — summary stats computed from soft output**: Fix by routing summary-stat MSE through the hard curve generation path (gameplan Section 3.4). The soft curve shapes gradients per step; the hard curve measures fitness every 50 epochs. These must be two separate code paths with no shared computation.

**Contingency — correctly separated**: No change needed. Note the variable names used for each path.

---

## 4. Check post-training output correlation from stochastic h0

Run after any training run of at least 50 epochs.

```python
import torch
import numpy as np
from antisaccade_model.model.lrrnn import LRRNN
from antisaccade_model.task.trial_generator import sample_initial_state, build_inputs

model = LRRNN()
model.load_checkpoint("path/to/checkpoint", strict=False)
model.eval()

N = 1000
z_cue_0, z_goal_0 = [], []
for _ in range(N):
    h0 = sample_initial_state(batch_size=1, n_hidden=200, task=None)
    u  = build_inputs(..., m=0.5, lapse_mask=False)
    with torch.no_grad():
        z = model(u, h0=h0)
    z_cue_0.append(z[0, 0].item())
    z_goal_0.append(z[0, 1].item())

corr = np.corrcoef(z_cue_0, z_goal_0)[0, 1]
print("Empirical output correlation at t=0:", corr)
```

**Contingency — corr < 0.7**: Insufficient inter-plan correlation. Increase `sigma_init_shared` (try 0.5, then 0.7) and reduce `sigma_init_private` (try 0.05) in `TaskParams`. Re-run check after each change.

**Contingency — corr ≥ 0.7**: Acceptable. Note value. If RT distributions later look too narrow or broad, return here first.

---

## 5. Analysis stability check (run after full training)

```python
import torch
import numpy as np
from antisaccade_model.analysis.tachometric_analysis import generate_tachometric_curve

results = []
for seed in [42, 123]:
    torch.manual_seed(seed)
    tc = generate_tachometric_curve(model, m=0.0, N=500)
    t_rise = rPT_bins[np.argmin(np.abs(tc - 0.75))]
    results.append(t_rise)

print("t_rise across seeds:", results, "| diff:", abs(results[0] - results[1]), "ms")
```

**Contingency — diff > 5 ms**: N=500 insufficient given lapse sampling variance. Increase to N=1000 per rPT bin in `tachometric_analysis.py` analysis sweeps (not training).

**Contingency — diff ≤ 5 ms**: No change needed.

---

## 6. Smoke test (run last)

```python
python -c "
import torch
from antisaccade_model.training.train import make_batch
from antisaccade_model.model.lrrnn import LRRNN
from antisaccade_model.training.losses import compute_loss

model = LRRNN()
batch = make_batch(batch_size=16, m_values=[0.0, 1.0])
loss, info = compute_loss(model, batch)
loss.backward()
print('loss:', loss.item())
print('gradients flowing:', all(p.grad is not None for p in model.parameters() if p.requires_grad))
print('lapse_young:', torch.sigmoid(model.lapse_young_logit).item())
print('lapse_adult:', torch.sigmoid(model.lapse_adult_logit).item())
"
```

All assertions must pass. If any parameter has no gradient, trace it back through the loss — likely a detach() call in the wrong place or a branch that is not connected to the loss.
