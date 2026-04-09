# Exercise: Implement the WSD Learning Rate Scheduler

In this exercise you will implement the **WSD (Warmup-Stable-Decay)** learning rate schedule used by DeepSeek-V3, Kimi K2, Qwen 3, and many frontier models. You'll work on a standalone copy of MegatronLM's scheduler, fill in the missing WSD logic, then verify your implementation with tests and a plot.

---

## 1. How LR scheduling works in MegatronLM / MegatronBridge

### Two-layer architecture

The scheduling system is split across two codebases:

| Layer | Responsibility | Key file |
|-------|---------------|----------|
| **MegatronBridge** (config layer) | Converts human-friendly config (token budgets, ratios) into absolute iteration counts | `launcher.py` → `compute_derived()` |
| **MegatronLM** (runtime layer) | Steps the LR each iteration using `OptimizerParamScheduler` | `megatron/core/optimizer_param_scheduler.py` |

### Config flow

```
YAML config                           MegatronBridge                    MegatronLM
-----------                           --------------                    ----------
max_duration_in_token: 5e9            compute_derived():                OptimizerParamScheduler(
warmup_ratio: 0.1          ──>          train_iters = 1220      ──>      lr_warmup_steps=122,
decay_ratio: 0.1                        warmup_iters = 122               lr_decay_steps=1220,
lr_decay_style: WSD                     decay_iters = 122                wsd_decay_steps=122,
lr_wsd_decay_style: minus_sqrt                                           lr_wsd_decay_style="minus_sqrt"
                                                                       )
```

The bridge's `resolve_config.py` maps flat YAML keys to MegatronLM's `SchedulerConfig`:

```python
# resolve_config.py FLAT_TO_CONFIG (scheduler entries)
"warmup_iters":       ("scheduler", "lr_warmup_iters"),
"decay_iters":        ("scheduler", "lr_wsd_decay_iters"),
"lr_decay_style":     ("scheduler", "lr_decay_style"),
"lr_wsd_decay_style": ("scheduler", "lr_wsd_decay_style"),
```

### Step vs iteration

Internally, the scheduler operates on **steps** = `iters x global_batch_size`. This means the LR schedule is a function of **data seen** (samples), not optimizer steps. The multiplication happens in `get_optimizer_param_scheduler()` before constructing the scheduler. In this exercise, we simplify by treating iterations directly as steps.

### Why WSD for continual pretraining

Unlike cosine decay, WSD does not require pre-specifying the total token budget:
- The **stable phase** can run indefinitely at peak LR
- You can branch off a **decay phase** at any point to produce a converged checkpoint
- Multiple sweep experiments can share the same stable-phase checkpoint

This makes WSD ideal for CPT, where the optimal training duration isn't known upfront.

---

## 2. What you'll implement

Open `optimizer_param_scheduler.py` and find the `get_lr()` method. The WSD block has three TODO groups:

### TODO 1: Calculate the annealing start point

The WSD decay phase occupies the **last** `wsd_decay_steps` steps of training. So it begins at:

```
wsd_anneal_start = lr_decay_steps - wsd_decay_steps
```

With our parameters: `1220 - 122 = 1098`.

### TODO 2: Stable vs decay phase branching

- **Stable phase** (`num_steps <= wsd_anneal_start`): The coefficient is `1.0`, keeping LR at `max_lr`.
- **Decay phase** (`num_steps > wsd_anneal_start`): Compute how far into the decay we are:
  ```
  wsd_steps = num_steps - wsd_anneal_start
  wsd_decay_ratio = wsd_steps / wsd_decay_steps    # 0.0 at start, 1.0 at end
  ```

### TODO 3: Implement the decay styles

Each style maps `wsd_decay_ratio` (0 to 1) to a coefficient (1 to 0):

| Style | Formula | Shape |
|-------|---------|-------|
| `minus_sqrt` | `1 - sqrt(ratio)` | Fast initial drop, then tapers |
| `linear` | `1 - ratio` | Constant rate of decay |
| `cosine` | `0.5 * (cos(pi * ratio) + 1)` | Smooth S-curve |
| `exponential` | `2 * 0.5^ratio - 1` | Fast start, decelerates |

The final learning rate is: **`min_lr + coeff * (max_lr - min_lr)`**

When `coeff = 1.0`, LR = `max_lr`. When `coeff = 0.0`, LR = `min_lr`.

> **Hint**: Look at how the non-WSD `linear` and `cosine` decay styles are implemented earlier in the same method (around line 155). The WSD versions follow the same pattern, just using `wsd_decay_ratio` instead of `decay_ratio`.

---

## 3. Step-by-step instructions

1. Open `optimizer_param_scheduler.py` in your editor
2. Find the `get_lr()` method and locate `elif self.lr_decay_style == 'WSD':`
3. Replace each `...` with the correct expression:
   - **TODO 1**: One line — compute `wsd_anneal_start_`
   - **TODO 2**: Three lines — set `coeff` for stable phase, compute `wsd_steps` and `wsd_decay_ratio` for decay phase
   - **TODO 3**: Four lines — one formula per decay style
4. Save the file

---

## 4. Running the smoke test

### Generate the LR schedule plot

```bash
python W2D2/exercise/test_wsd_scheduler.py
```

This produces `W2D2/exercise/wsd_schedule.png` with two panels:
- **Left**: The minus_sqrt WSD schedule with warmup/stable/decay phases annotated
- **Right**: All four decay styles overlaid for comparison

### Run the correctness tests

```bash
pytest W2D2/exercise/test_wsd_scheduler.py -v
```

**Before implementation**: Tests will fail with `TypeError` (the `...` Ellipsis values cause comparison errors).

**After correct implementation**: All tests should pass:
- `test_warmup_start` — LR is 0.0 at step 0
- `test_warmup_end` — LR reaches max_lr at end of warmup
- `test_stable_phase` — LR stays at max_lr during stable phase
- `test_decay_start` — LR is still max_lr at the start of decay
- `test_decay_end` — LR reaches min_lr at the end of decay
- `test_decay_midpoint` — LR matches `1 - sqrt(0.5)` coefficient at midpoint
- `test_beyond_total_steps` — LR stays at min_lr after training ends
- `test_minus_sqrt_monotonic` — LR never increases during decay
- `test_start_and_end[style]` — Each decay style starts at max_lr and ends at min_lr
- `test_monotonic_decay[style]` — Each decay style is monotonically decreasing

---

## 5. Applying your changes to MegatronLM

The exercise file is a standalone copy of:

```
megatron/core/optimizer_param_scheduler.py
```

Only the `elif self.lr_decay_style == 'WSD':` block was removed. To apply your implementation to the real codebase:

1. Open the MegatronLM source at the path above
2. Find the `get_lr()` method
3. Replace the WSD block (lines 259-273 in the original) with your completed implementation
4. No changes are needed in MegatronBridge — it already configures `lr_decay_style: WSD` and `lr_wsd_decay_style: minus_sqrt` by default

The rest of the file is identical between the exercise copy and the original.

---

## Reference: WSD schedule parameters in MegatronBridge config

```yaml
# config.yaml defaults
lr: 1.0e-4              # max_lr (peak learning rate)
lr_min: 1.0e-9          # min_lr (floor after decay)
warmup_ratio: 0.1       # fraction of train_iters for warmup
decay_ratio: 0.1        # fraction of train_iters for WSD decay
lr_decay_style: WSD
lr_wsd_decay_style: minus_sqrt
```

These are translated by `launcher.py:compute_derived()`:
```python
train_iters = int(max_duration_in_token) // (global_batch_size * seq_length)
warmup_iters = int(train_iters * warmup_ratio)    # 122
decay_iters  = int(train_iters * decay_ratio)      # 122
```
