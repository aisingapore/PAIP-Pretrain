## Primary hyperparameters: learning rate and batch size

### Learning rate sweep ranges by model size

> *Why not just reuse the base model's original peak LR for continued pretraining?*

The optimal LR for CPT is almost always lower than the original pretraining peak, because the model is already near a good minimum and aggressive updates cause catastrophic forgetting. How much lower depends on the model size, domain shift, and token budget.


| Model size | Original peak LR | CPT sweep range  | Token horizon per grid point |
| ---------- | ---------------- | ---------------- | ---------------------------- |
| 7B–13B     | ~3e-4            | **1e-5 to 3e-4** | 1B tokens                    |
| 34B        | ~2e-4            | **1e-5 to 1e-4** | 1–2B tokens                  |
| 70B        | ~1.5e-4          | **1e-5 to 5e-5** | 2–3B tokens                  |


These ranges are derived from published CPT runs: Code Llama used 1e-4 across scales for 500B-token code CPT, Llemma 34B used 5e-5 for 50B-token math CPT, and Databricks swept {1e-5, 3e-6, 1e-5, 3e-5} for Llama-2-7B CPT on 14.5B tokens. DeepSeek's pretraining LR decreased from 4.2e-4 (7B) to 3.2e-4 (67B), roughly a 24% reduction for 10× more parameters.

**MegatronBridge config:** `lr` (peak learning rate), `lr_min` (minimum LR, typically 10% of peak or 1e-9).

### Token horizon for hyperparameter sweeps

> *If you sweep LR using 1B-token pilot runs but train for 20B tokens, will the best LR be the same?*

No. "Scaling Optimal LR Across Token Horizons" (Tissue et al., arXiv:2409.19913) established a power-law relationship between optimal learning rate and training duration:

$$\text{LR}^*(D) = B \cdot D^{-\beta}, \quad \beta \approx 0.3$$

where D is the number of training tokens and B is a model-dependent constant. This means the optimal LR **decreases** as training runs longer. A sweep performed on 1B tokens will select a higher LR than what is optimal for a 20B-token full run.

**Relationship to total training budget.** The token horizon per grid point during HP sweeps is typically **5–15% of the total CPT budget** T. This range balances two competing concerns:

- **Too short** (< 5% of T): The sweep cannot distinguish good from bad LRs because all runs look similar early on. You risk selecting an LR that happens to converge fastest initially but diverges or plateaus later.
- **Too long** (> 15% of T): The sweep consumes too much compute. For a 20B-token CPT, running 20 grid points at 3B tokens each would cost 60B tokens—3× the actual training budget.

**Correction when extrapolating from sweep to full run.** When the sweep token horizon H is shorter than the total budget T, the optimal LR from the sweep overestimates the optimal LR for the full run. Apply the power-law correction:

$$\text{LR}*{\text{full}} \approx \text{LR}*{\text{sweep}} \cdot \left(\frac{H}{T}\right)^{\beta}$$

**Worked example:** You sweep at H = 1B tokens and find optimal LR = 8e-5. For a T = 20B-token full run with β ≈ 0.3:

$$\text{LR}_{\text{full}} \approx 8\text{e-5} \times (1/20)^{0.3} \approx 8\text{e-5} \times 0.38 \approx 3\text{e-5}$$

A simpler rule of thumb: apply a **20–30% downward adjustment** when extrapolating from short sweeps, which is accurate when H/T is between 0.05 and 0.15.

**Why this matters more for CPT than from-scratch training.** From-scratch pretraining budgets are enormous (1T+ tokens), so even a 1B-token sweep is only 0.1% of the total and the correction is small. CPT budgets are much smaller (5–50B tokens), making the sweep-to-full ratio larger and the correction more consequential. At 7B scale with a 10B-token CPT budget, a 1B-token sweep is 10% of the budget—the correction factor shifts the LR by roughly 1.5–2×.

### Batch size: global batch size and micro batch size

> *Why can't you just keep doubling batch size to train faster?*

There is a point of diminishing returns. McCandlish et al. ("An Empirical Model of Large-Batch Training," arXiv:1812.06162, 2018) defined the **gradient noise scale** B_noise = tr(Σ)/‖G‖², which predicts the **critical batch size**—the point where doubling the batch no longer halves the required steps. Below the critical batch size, scaling is near-linear; above it, you pay in compute without proportional speedup.

**Concept:** GBS (global batch size) is the total number of samples processed per optimizer step across all GPUs. MBS (micro batch size) is what each GPU processes in one forward/backward pass. The relationship: GBS = MBS × data_parallel_size × gradient_accumulation_steps.

**Batch size warmup.** Merrill et al. (arXiv:2505.23971, 2025) revisited critical batch size dynamics and found that it starts near zero, rises rapidly, then plateaus during training, and does not depend strongly on model size. This supports the common practice of **batch size warmup**: starting at ~50% of target GBS and ramping over the first 5–10% of training. The intuition is that early in training (or early in CPT after re-warming), gradients are noisy and a smaller batch provides more frequent updates to navigate the rapidly changing loss landscape.

**Recommended range:** 4M–16M tokens per batch for CPT. Code Llama and Llemma both used 4M tokens. DeepSeek-V3 ramped from 12.6M to 63M tokens over training.

**MegatronBridge config:** `global_batch_size` (GBS, default 1024), `micro_batch_size` (MBS, default 1). GBS is specified in number of sequences, so tokens per batch = GBS × seq_length.

### LR-batch size joint scaling

> *If you double the batch size, should you double the learning rate too?*

Not for Adam-family optimizers. The **square root scaling rule** applies: when doubling batch size, increase LR by ~1.4× (√2), not 2×. This was theoretically derived by Malladi et al. (NeurIPS 2022) via stochastic differential equation analysis of Adam. The intuition is that Adam's per-parameter adaptive scaling already compensates for some of the variance reduction from larger batches, so the LR correction is less aggressive than SGD's linear scaling rule.

Li et al. (arXiv:2405.14578, NeurIPS 2024) discovered a **"surge phenomenon"** where optimal LR first rises then falls as batch size increases, with a sweet spot near the gradient noise scale. The practical implication: there is an optimal batch size beyond which even with the correct LR scaling, further increases hurt convergence quality.

For a joint sweep, use **4–5 LR values × 3–4 batch size values = 12–20 grid points**, each trained for the token horizon from the table above (1–3B tokens depending on model scale). This is sufficient to identify the Pareto-optimal LR-BS combination.

---

## Secondary hyperparameters

### Weight decay

> *If every major lab uses weight_decay=0.1, why bother sweeping it?*

Because CPT's optimization landscape differs from pretraining. The model starts from a good minimum with well-calibrated weight norms, so the regularization strength that was optimal during pretraining may not be right for the adaptation phase.

**Concept.** Weight decay adds a penalty proportional to the magnitude of weights, preventing them from growing unbounded. In the **decoupled** formulation (AdamW, Loshchilov & Hutter 2017), weight decay is applied directly to the weights rather than through the gradient:

$$w_{t+1} = (1 - \lambda) \cdot w_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

where λ is the weight decay coefficient. This is different from L2 regularization in Adam, where the penalty is added to the loss gradient. The distinction matters because Adam's adaptive scaling distorts the L2 penalty—large-gradient parameters get less regularization than intended, small-gradient parameters get more. Decoupled weight decay regularizes uniformly.

**Impact.** Weight decay serves two roles: (1) it prevents weight norm explosion during long training runs, and (2) it acts as an implicit regularizer that encourages simpler solutions. A recent finding from Han et al. (arXiv:2602.11137, 2025) suggests that **higher weight decay (0.3–1.0) during pretraining makes models more plastic**—easier to adapt to downstream tasks. This is particularly relevant for CPT, where adaptability is the entire goal. The mechanism: larger weight decay keeps the model closer to the origin in parameter space, leaving more "room" for the optimizer to move toward the new domain.

**Recommended range:** 0.1 as the default (universal across Llama 3, DeepSeek-V3, Qwen 2.5). For CPT sweeps, **{0.01, 0.05, 0.1, 0.3}** covers the relevant range.

**MegatronBridge config:** `weight_decay` (default 0.1), `decoupled_weight_decay` (always set to `true` for AdamW-style decoupling).

### Gradient clipping

> *If gradient clipping triggers on more than 10% of steps, what does that tell you about your learning rate?*

It tells you the LR is probably too high. Gradient clipping is a safety net, not a tuning knob—if it activates frequently, the optimizer is routinely overshooting.

**Concept.** Gradient clipping rescales the gradient when its norm exceeds a threshold, preventing catastrophically large updates from noisy batches or data anomalies:

$$\text{if } g > \text{maxnorm}: \quad g \leftarrow g \times \frac{\text{maxnorm}}{g}$$

This is **global norm clipping**—the norm is computed over all parameters at once, and the entire gradient vector is rescaled proportionally. This preserves the relative direction of gradients across layers, unlike per-parameter clipping which can distort the update direction.

**Impact.** Gradient clipping prevents training divergence from loss spikes caused by outlier batches (e.g., a batch of noisy or anomalous data). In CPT, this is especially important during the re-warming phase when the optimizer encounters data from a new distribution and gradient statistics are volatile. For extra stability, AdaGC (arXiv:2502.11034, 2025) offers **adaptive per-tensor clipping** using EMA of historical gradient norms, and eliminated all loss spikes in their Llama-2 7B/13B experiments—though it adds implementation complexity.

**Recommended range:** **1.0** (max grad norm) is the universal standard and rarely needs adjustment for CPT.

**MegatronBridge config:** `clip_grad` (default 1.0).

### Warmup

> *"Reuse, Don't Retrain" found that zero warmup achieved the best evaluation results on a 15B model. Should you skip warmup entirely?*

Probably not without testing it, but this result highlights that warmup is less critical than peak LR selection. The conventional wisdom is that warmup prevents early-training instability, but the evidence for CPT is mixed.

**Concept.** Warmup linearly ramps the learning rate from zero (or near-zero) to the target peak over a specified number of steps:

$$\text{LR}(t) = \text{peakLR} \times \frac{t}{T_{\text{warmup}}}, \quad \text{for } t < T_{\text{warmup}}$$

**Intuition.** Adam's second-moment estimate $\hat{v}_t$ needs time to calibrate to the true gradient variance. In the first few steps, $\hat{v}_t$ is initialized from zero and biased low (even with bias correction), so the effective step size is larger than intended. Warmup compensates by keeping the nominal LR small while $\hat{v}_t$ converges. For CPT specifically, when optimizer states are reset (fresh initialization from a public checkpoint without optimizer states), this calibration period is critical. When continuing with existing optimizer states, the case for warmup is weaker—the $\hat{v}_t$ estimates are already calibrated to the parameter landscape.

**Impact.** Warmup prevents early loss spikes and gradient norm explosions during the transition to a new learning rate. However, peak LR selection matters far more than warmup duration—a well-chosen peak with no warmup often outperforms a poor peak with extensive warmup. Include **warmup ∈ {0, 0.5%, 1%, 2%}** in your sweep to determine if warmup helps for your specific setting.

**Recommended range:** 1–2% of total CPT steps, or ~2000 steps as a fixed count. The warmup duration is discussed further in `2-scheduler.md`, which covers how warmup integrates with the WSD schedule phases.

**MegatronBridge config:** `warmup_ratio` (warmup as fraction of total training steps, default 0.1).

### AdamW epsilon

> *Why does Qwen use eps=1e-6 instead of the standard 1e-8?*

Mixed-precision safety. When training in bf16 or fp16, the denominator $\sqrt{\hat{v}_t} + \epsilon$ can underflow for parameters with very small gradients, causing division-by-near-zero and exploding updates. A larger epsilon provides a wider safety margin.

**Concept.** Epsilon is a small constant added to the denominator of the Adam update rule to prevent division by zero:

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Intuition.** When $\hat{v}_t$ is very small (parameters that receive near-zero gradients, such as rarely-activated neurons or certain embedding dimensions), the update $\hat{m}_t / \sqrt{\hat{v}_t}$ becomes disproportionately large. Epsilon floors this ratio. At larger model widths, individual parameter gradients become smaller in magnitude (each parameter "sees" a smaller fraction of the total gradient signal), making epsilon relatively more important. This is one reason Everett et al. ("Scaling Exponents Across Parameterizations and Optimizers," arXiv:2407.05872, ICML 2024) proposed **Adam-atan2**, which replaces $m/(\sqrt{v}+\epsilon)$ with $\text{atan2}(m, \sqrt{v})$—a scale-invariant formulation that eliminates the epsilon parameter entirely.

**Recommended range:** **1e-8** (default) or **1e-6** for mixed-precision safety (Qwen convention). If you observe NaN gradients or sudden loss spikes that correlate with specific layers, try increasing epsilon before reducing LR.

**MegatronBridge config:** `adam_eps` (default 1e-8).

### AdamW beta1 (first moment decay)

**Concept.** Beta1 controls the exponential moving average of the gradient (momentum). A higher value produces smoother updates but slower adaptation to changing gradient directions:

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

**Intuition.** With β₁ = 0.9, the optimizer effectively averages over the last ~10 gradient steps (the effective window is 1/(1-β₁) = 10). This provides enough momentum to smooth out per-batch noise while remaining responsive to genuine changes in the loss landscape. Lowering β₁ makes the optimizer more reactive but noisier; raising it makes updates smoother but risks the optimizer "remembering" stale gradient information when the data distribution shifts—which happens during CPT.

**Recommended range:** **0.9** (universal across Llama 2/3, DeepSeek-V3, Qwen 2.5, OLMo 2). This parameter is not worth sweeping for CPT.

**MegatronBridge config:** `adam_beta1` (default 0.9).

### AdamW beta2 (second moment decay)

> *Why do all major LLM labs use beta2=0.95 instead of Adam's default 0.999?*

Speed of adaptation. The second moment $\hat{v}_t$ controls the per-parameter step size by tracking the recent gradient magnitude. With β₂ = 0.999, this estimate averages over ~1000 steps and responds very slowly when gradient statistics change. With β₂ = 0.95, it averages over ~20 steps, enabling the optimizer to quickly recalibrate when encountering new data domains or training phases.

**Concept.** Beta2 controls the exponential moving average of the squared gradient, which Adam uses to normalize updates per-parameter:

$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

The adaptive learning rate for each parameter is inversely proportional to $\sqrt{\hat{v}_t}$: parameters with large recent gradients get smaller effective step sizes, and vice versa.

**Intuition for CPT.** During continued pretraining, the model transitions from one data distribution to another. The gradient statistics for many parameters shift—some layers that were "quiet" during pretraining become active when processing domain-specific data, and vice versa. A fast-adapting $v_t$ (β₂ = 0.95) recalibrates within ~20 steps, while the default 0.999 would take ~1000 steps to adjust. This lag can cause some parameters to receive updates that are too large or too small for the current gradient regime, leading to instability or slow convergence.

**Recommended range:** **0.95** (standard for LLM training; do NOT use the default 0.999).

**MegatronBridge config:** `adam_beta2` (default 0.95).

### Dropout

> *If dropout=0 is universal for modern LLM pretraining, when would you actually use it?*

Only when you are forced to repeat data. Dropout regularizes against overfitting, and single-epoch training on diverse data does not overfit—so dropout adds noise without benefit. The exception is multi-epoch CPT on small domain corpora, where the model sees the same examples multiple times and begins memorizing them.

**Concept.** Dropout randomly zeroes a fraction p of activations during training and rescales the rest:

$$\text{output} = \frac{\text{mask} \cdot \text{input}}{1 - p}, \quad \text{mask} \sim \text{Bernoulli}(1-p)$$

**Impact.** Liu et al. (arXiv:2505.24788, ACL 2025) confirmed that single-epoch training without dropout outperforms training with dropout on downstream tasks. The intuition: dropout forces the model to learn redundant representations, which is helpful when the same data is seen multiple times but wasteful when each example is seen only once. For multi-epoch scenarios, Muennighoff et al. ("To Repeat or Not To Repeat," arXiv:2305.13230, 2023) found that **dropout of 0.1 is highly effective** at mitigating the degradation from data repetition.

**Recommended range:** **0** (default). Only add **0.05–0.1** if your domain corpus is small enough to require multiple training epochs.

**MegatronBridge config:** Dropout is configured in the model architecture definition, not through `resolve_config.py` flat keys. Set in the model YAML or Megatron model arguments.

---

## Stability techniques

### Z-loss

> *Why would you add an extra loss term that doesn't directly optimize for next-token prediction?*

Because logit explosion can silently degrade training long before it causes a visible loss spike. Z-loss provides continuous pressure to keep logit magnitudes bounded, acting as an early warning system and preventive measure simultaneously.

**Concept.** Z-loss adds an auxiliary penalty on the log-sum-exp of the output logits (the softmax normalizer), encouraging it to stay near zero. Introduced in PaLM (Chowdhery et al. 2024), it has been adopted by OLMo and Chameleon.

**Formula:**

$$\mathcal{L}_z = \alpha \cdot \left(\log \sum_i \exp(z_i)\right)^2$$

where $z_i$ are the output logits and α is a small coefficient.

**Intuition.** When logit magnitudes grow unchecked during training, the softmax distribution becomes increasingly peaked (confident), the cross-entropy gradients become unstable, and eventually a single bad batch can trigger a catastrophic loss spike. Z-loss acts as a "leash" on the logit scale—it does not prevent the model from being confident (the relative ordering of logits is unaffected), but it penalizes absolute magnitudes. This keeps the softmax operating in its numerically stable regime where gradients flow smoothly.

**Why Z-loss is the safest stability addition for CPT.** Unlike logit soft-capping or QK-norm (which modify the model architecture), Z-loss only modifies the loss function. This means you can add it to any model at any point during training without architectural compatibility concerns. The model weights and forward pass remain unchanged—only the gradient signal includes the extra stabilizing term.

**Recommended range:** α ≈ **1e-4** (standard across PaLM, OLMo, Chameleon).

**MegatronBridge config:** Architecture-level loss setting (not in `resolve_config.py` flat keys). Configure in model definition or training arguments.

### Logit soft-capping

> *If Z-loss is a gentle nudge to keep logits small, logit soft-capping is a hard wall. When is the wall better than the nudge?*

When you want to push the learning rate higher without risking divergence. Soft-capping guarantees a bounded logit range regardless of what the model produces, enabling more aggressive optimization.

**Concept.** Logit soft-capping applies tanh to compress logits into a bounded range before softmax. Introduced in Gemma 2 (Google, 2024):

**Formula:**

$$z_{\text{capped}} = \text{cap} \cdot \tanh\left(\frac{z}{\text{cap}}\right)$$

where cap is a hyperparameter (typically 30–50) that determines the maximum logit magnitude.

**Intuition.** The tanh function naturally saturates: for small inputs, it is nearly linear (preserving the model's intended logit values), but for large inputs, it compresses toward ±cap. Unlike hard clipping, the gradient flows smoothly through tanh—the model can still "push" logits toward the boundary, it just gets diminishing returns, which is exactly the behavior you want. This is softer than a ReLU clip (which has zero gradient past the threshold) but harder than Z-loss (which only penalizes through the loss, not the forward pass).

**Impact.** Rybakov et al. (arXiv:2410.16682, 2024) showed that logit soft-capping allows **1.5× higher learning rates** without divergence compared to baseline training. Higher LR means faster adaptation during CPT—a significant practical benefit when the token budget is limited and you need to maximize learning per step.

**Recommended range:** cap value **30–50**.

**Critical constraint for CPT:** The base model must have been pretrained with logit soft-capping for you to use it during CPT. Adding it to a model that was trained without it requires significant re-adaptation—the model's learned logit scale suddenly gets compressed, and it must re-learn appropriate weight magnitudes. This wastes CPT budget.

### QK-norm

> *Attention logits can grow unboundedly during training—what prevents a single head from "hogging" all the attention?*

Nothing, by default. QK-norm addresses this by normalizing the query and key vectors before they interact, keeping the attention score magnitudes in check.

**Concept.** QK-norm applies LayerNorm to the Q and K projections before computing attention scores. Introduced by Dehghani et al. (2023):

**Formula:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{\text{LayerNorm}(Q) \cdot \text{LayerNorm}(K)^T}{\sqrt{d_k}}\right) V$$

**Intuition.** Without normalization, the Q and K vectors can grow in magnitude over the course of training as weights accumulate updates. The dot product Q·K scales with the product of their norms, so even modest growth in Q and K norms can cause attention logits to explode. When attention logits become very large, softmax saturates—nearly all attention weight goes to a single token, and the gradient with respect to other tokens vanishes. QK-norm prevents this by ensuring Q and K have unit variance regardless of the underlying weight magnitudes.

**Caveat.** QK-norm can hurt long-context performance because it dampens the model's ability to form very sharp attention patterns at long distances. Sharp attention (high-magnitude logits on a specific far-away token) is exactly what you need for tasks like long-range copying or retrieval from context. QK-norm limits how sharp these patterns can be, trading long-context capability for training stability.

**Recommended usage:** Beneficial for very long training runs where attention logit drift accumulates. For CPT on 5–50B tokens, the training duration is typically not long enough for attention logit explosion to become a problem unless the base model was already close to the instability threshold.

**Critical constraint for CPT:** Same as logit soft-capping—the base model must already have QK-norm. Adding LayerNorm layers to Q and K during CPT changes the model architecture and requires the model to re-learn its attention patterns from scratch, which is prohibitively expensive.

### When to add stability techniques during CPT

A simple decision rule:

- **Z-loss:** Safe to add at any time. Only modifies the loss function; no architectural change. Recommended as a default for all CPT runs.
- **Logit soft-capping and QK-norm:** Only use if the base model already has them. Do NOT add them during CPT—the architectural re-adaptation cost exceeds the stability benefit for typical CPT budgets.
- **If you observe loss spikes during CPT:** First check gradient clipping activation rate (>10% suggests LR is too high). Then try adding Z-loss. Only as a last resort, consider reducing LR or adding AdaGC adaptive clipping.

---

## Cross-references

### Hyperparameter transfer with μP

**μP (Maximal Update Parameterization)** enables zero-shot HP transfer from a small proxy model (~40M params) to targets at 7B–70B scale, potentially saving ~93% of tuning compute. However, μP's theoretical guarantees assume random initialization and break down for CPT where the model already has learned weights. Practitioners typically use direct LR sweeps at target scale rather than full μP reformulation for CPT. See `**W2D4-5/uP.md*`* for the complete workflow: proxy model selection, depth transfer, scale ratio constraints, and CPT-specific limitations.

### Data mixing ratios

The replay ratio of general-domain data mixed with domain-specific data during CPT depends on the severity of the distribution shift: **5% replay** for mild shift (general→domain-specific English), **10–20%** for moderate shift (math/code specialization), and up to **50%** for strong shift (new language). See `**W1D3/0-datamix.md`** for temperature sampling, UniMax strategies, CMR scaling law, two-phase data strategies, and proxy-model ratio optimization.

---

## Quick reference table for CPT sweeps at 7B–70B


| Hyperparameter           | Default                     | Sweep range                    | Notes                              |
| ------------------------ | --------------------------- | ------------------------------ | ---------------------------------- |
| **LR schedule**          | WSD                         | {WSD, cosine}                  | WSD preferred for flexible budgets |
| **Peak LR (7B)**         | 1e-4                        | {1e-5, 3e-5, 5e-5, 1e-4, 2e-4} | ~1/3× to 1× of original 3e-4       |
| **Peak LR (70B)**        | 3e-5                        | {1e-5, 2e-5, 3e-5, 5e-5}       | More conservative at scale         |
| **Min LR**               | 10% of peak                 | {0, 5%, 10%} of peak           | Avoid decaying to zero             |
| **Warmup**               | 2000 steps                  | {0, 500, 1000, 2000} steps     | Peak LR matters more               |
| **Batch size**           | 4M tokens (7B), 8–16M (70B) | {1M, 2M, 4M, 8M} tokens        | √ scaling with LR                  |
| **Optimizer**            | AdamW                       | —                              | β₁=0.9, β₂=0.95, ε=1e-8            |
| **Weight decay**         | 0.1                         | {0.01, 0.05, 0.1, 0.3}         | Decoupled (AdamW)                  |
| **Gradient clip**        | 1.0                         | Rarely sweep                   | Monitor activation frequency       |
| **Replay ratio**         | 5%                          | {2%, 5%, 10%, 20%}             | Higher for stronger domain shift   |
| **Dropout**              | 0.0                         | {0.0, 0.05, 0.1}               | Non-zero only for multi-epoch      |
| **Decay fraction (WSD)** | 15% of tokens               | {10%, 15%, 20%}                | —                                  |
| **Method**               | Full FT                     | {Full FT, LoRA r=256}          | Full FT strongly preferred         |
| **EMA decay**            | 0.9999                      | —                              | Free improvement                   |
| **Z-loss**               | 1e-4                        | {0, 1e-4}                      | Safe stability addition            |


