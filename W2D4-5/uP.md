# Hyperparameter transfer from small proxy models to large LLMs

**The μP framework enables transferring learning rate and initialization hyperparameters from models as small as 40M parameters to targets exceeding 7B — reducing tuning cost to roughly 7% of a single target-scale run.** But μP only handles the width dimension. Optimal learning rate also depends on training duration, batch size, and depth — each requiring separate correction. And for continued pretraining, μP's theoretical guarantees break down entirely because the model is no longer at random initialization. This report walks through the complete methodology: what transfers, the exact workflow, the constraints, and why CPT demands a different approach.

---

## 1. What transfers and what does not

The Tensor Programs V paper (Yang et al., 2022) provides the definitive categorization. Under μP, **optimization-related hyperparameters transfer across model width**: base learning rate, momentum coefficients (β₁, β₂), LR schedule shape, initialization variance, and per-layer multiplicative constants (α_input, α_output). These can be tuned once on a narrow proxy model and applied directly to a wide target.

**What does not transfer**: regularization hyperparameters — specifically weight decay and dropout — which depend jointly on model size and data size. Batch size does not transfer either. Training duration (token horizon) is explicitly outside μP's scope. The paper states minimum requirements for the proxy: width ≥ 256, depth ≥ 4, batch size ≥ 32, sequence length ≥ 128, and at least 5,000 training steps.

The weight decay story has grown more nuanced since the original paper. Kosson et al. (2025) showed that **μP's geometric alignment assumptions hold only briefly at the start of training**. For the remainder, it is actually weight decay — specifically independent/truly decoupled weight decay — that stabilizes update dynamics across widths and enables LR transfer. Standard PyTorch AdamW couples weight decay with learning rate (`w = (1 - η·λ)·w - η·update`), which means μP's per-layer LR scaling of 1/m_d makes effective weight decay vanishingly small for large models' hidden layers. This causes unbounded parameter norm growth and undermines transfer. The fix is straightforward: use truly independent weight decay (`w = (1 - λ)·w - η·update`), where λ is not scaled by learning rate.

Recent work on depth transfer adds another dimension. Standard μP was derived for width scaling only. Bordelon et al. (ICLR 2024) and Yang et al. (Tensor Programs VI, 2023) showed that transferring across depth requires an additional **1/√L residual branch scaling** (each block output divided by √depth before adding to the residual stream). The CompleteP parameterization (Dey et al., 2025) found that in practice **1/L scaling (not 1/√L) is superior** for practical transformers, enabling HP transfer from 2 to 128 layers and supporting aspect ratios (width-to-depth) as low as 11.8 while staying within 1% of compute-optimal.

| Hyperparameter | Transfers? | Notes |
|---|---|---|
| Learning rate (base) | ✅ Yes | Core μP transfer; scales as η_base/m_d for hidden layers with Adam |
| Init variance | ✅ Yes | Hidden weights: σ²_base/m_d |
| α_input, α_output | ✅ Yes | Tunable embedding/output multipliers; transfer directly |
| β₁, β₂ (Adam) | ✅ Yes | Standard (0.9, 0.95) works across all scales |
| LR schedule shape | ✅ Yes | Cosine or WSD shape transfers; peak magnitude needs scaling |
| Weight decay | ❌ No | Requires independent (decoupled) WD; scale-dependent adjustment needed |
| Dropout | ❌ No | Depends on model size + data size jointly |
| Batch size | ❌ No | Calibrate independently via critical batch size |
| Token horizon | ❌ No | Optimal LR decreases with longer training; separate correction needed |
| Epsilon (Adam) | ⚠️ Partial | Fixed ε can cause underflow at large widths; scale with width or use Adam-atan2 |

---

## 2. The complete hyperparameter transfer workflow

### Step 1: Select the proxy model

The proxy model must be small enough to be cheap but large enough for μP's statistical convergence. **Minimum hidden dimension is 256** — below this, the law of large numbers and central limit theorem don't converge sufficiently for the infinite-width theory to hold. In the GPT-3 experiment, Yang et al. used a proxy with d_model=256 (vs. 4096 target), keeping the same 32 layers, yielding a ~40M parameter proxy for a 6.7B target — a **168× parameter ratio** and **16× width ratio**.

Width is the primary dimension to shrink. **Depth should approximately match the target model** to avoid depth-dependent HP shifts. The EleutherAI/Cerebras practitioner's guide is explicit: choose depth roughly equivalent to the large-scale target. If you must reduce depth (for cost), use CompleteP's 1/L residual scaling to enable depth transfer. Training duration for each proxy run should follow Chinchilla-style scaling — roughly **20 tokens per parameter** — so a 40M proxy trains on ~800M tokens per run.

Batch size for the proxy is critical and often overlooked. The Cerebras guide warns that **if the proxy trains below the critical batch size but the target trains above it, LR transfer will fail**. Estimate the critical batch size via the gradient noise scale (McCandlish et al., 2018) and ensure the proxy trains at or above it.

### Step 2: Implement the parameterization

Converting from standard parameterization (SP) to μP requires five code changes for a decoder-only transformer with Adam:

1. **Hidden weight init variance**: σ²_hidden = σ²_base / m_d, where m_d = d_model / d_base
2. **Hidden weight learning rate**: η_hidden = η_base / m_d (for Adam; SGD scaling differs)
3. **Output logit scaling**: logits = α_output · x·W^T / m_d (not 1/√m_d — the square root only works at init; correlations during training require the full 1/m_d)
4. **Attention scaling**: Q^T K / d_head instead of Q^T K / √d_head (compensates for Q-K correlation that emerges during training)
5. **Embedding forward**: embed = α_input · x·W_emb (with tunable multiplier)

No corrections are needed for biases or layer-norm parameters. The embedding learning rate stays at η_base (unchanged). α_input and α_output are tunable scalars discovered during the proxy sweep.

**μP-simple** is a lighter variant (from Wortsman et al., 2023) that adopts only the 1/m_d LR scaling while keeping standard fan-in initialization — no output logit scaling, no attention scaling change. It successfully transfers optimal LR across width in many settings and is easier to implement, though full μP provides more robust transfer. Everett et al. (2024) showed that **SP with per-layer μP learning rates** ("SP-full-align") can actually outperform full μP in validation loss and LR transfer quality, suggesting the LR scaling is the most critical component.

For weight decay, use **truly independent/decoupled weight decay** — not PyTorch's default AdamW. This single change may matter more than the parameterization choice itself. In code: either implement a custom optimizer step where decay is independent of LR, or adjust by setting `weight_decay = weight_decay / group['lr']` in each parameter group.

### Step 3: Verify the implementation

Before any HP sweep, run two diagnostic tests:

**Coordinate check** (cheapest): Train 2-layer models at 4+ widths for 10 steps with large learning rate. Plot mean absolute activation per layer type. All activation magnitudes must be flat/independent of width. If curves blow up or shrink toward zero, there is a bug — most commonly a forgotten scaling factor on init, output logits, or attention.

**μTransfer test** (more expensive): Sweep learning rate across multiple widths. The optimal LR should be constant across widths. If it shifts systematically, the parameterization is incorrect. Additionally check the "wider is better" property: at any fixed HP setting, wider models should always achieve better training loss at every point during training.

### Step 4: Run the HP sweep at proxy scale

The sweep searches over **four hyperparameters jointly** via random search:

- Base learning rate η_base
- Base initialization standard deviation σ_base
- Embedding input multiplier α_input
- Output logit multiplier α_output

Yang et al. used **200 random samples** for GPT-3 and **256 samples** for BERT. Random search is preferred over grid search because it explores the space more efficiently when some dimensions matter more than others. Each sample is a full proxy training run at ~20 tokens per parameter. For a 40M proxy, that's ~800M tokens per run, which takes minutes on a single A100. The entire 200-sample sweep completes in hours on modest hardware.

**Evaluation criterion**: Use validation loss on the target data distribution. Downstream task metrics can supplement but are noisier at small scale. Select the HP configuration that achieves the lowest validation loss.

### Step 5: Validate at intermediate scale (optional but recommended)

For high-stakes training runs, validate at one intermediate scale before committing to the full target. For example, if transferring from 40M to 7B, validate at 256M or 590M. This means re-running the best 3–5 HP configurations at the intermediate scale for a full training run. If the transferred HPs remain near-optimal (within the top configs), confidence in the transfer is high. If they're clearly suboptimal, investigate — the likely causes are depth mismatch, batch size below critical threshold, or a parameterization bug.

### Step 6: Transfer to target scale and verify

Apply the discovered HPs directly to the target model using μP's scaling rules. The learning rate for hidden layers becomes η_hidden = η_base / m_d. All other transferred HPs (α_input, α_output, σ_base) apply without modification.

**Critical additional corrections**:

- **Token horizon correction**: If the target trains on significantly more tokens than the proxy, reduce the learning rate. Bjorck et al. (ICLR 2025) established that **LR*(D) = B · D^(−β)** with β ≈ 0.3. Concrete corrections: **10× more tokens → reduce LR by ~2×; 100× more tokens → reduce by ~4×; 1000× more tokens → reduce by ~8×**. This correction composes independently with μP's width correction.
- **Batch size**: Set independently based on critical batch size estimation at target scale, not transferred from proxy. B_crit scales approximately as B_crit ∝ T (proportional to training token budget). For Adam-family optimizers, the relationship between optimal LR and batch size follows the "surge phenomenon" (Li et al., NeurIPS 2024): optimal LR first rises then falls as batch size increases, eventually converging to a constant — so simple linear scaling from SGD does not apply.
- **Weight decay**: Adjust separately. Wang & Aitchison (2024) showed optimal weight decay should decrease as dataset size increases and increase as model size increases under μP LR scaling.

**Sanity check at target scale**: Run three short configurations — LR/2, LR (transferred), and LR×2 — for 5–10B tokens. Compare loss curves. The transferred LR should be near-optimal. If LR/2 or LR×2 clearly wins, expand search around the better point. If transfer has clearly failed, fall back to a 7–10 point LR grid search at target scale.

---

## 3. Scale ratio constraints and practical limits

### How far can width transfer stretch?

Yang et al. demonstrated transfer from width 256 to 4096 (16× width ratio, 168× parameter ratio from 40M to 6.7B). In stress tests, they pushed to widths up to 32,768 (128× width ratio). **In practice, μP transfer is approximate** — it becomes increasingly detuned at larger ratios. Cerebras's documented examples use a **10× width ratio** (256 → 2560), and their model zoo scales output projections by an additional `2 × num_blocks` factor to aid depth transfer. While no published source pins down a hard maximum, **practical reliability holds well within a 10–30× width ratio** (roughly 100–1000× in parameter count).

Beyond reliable range, transferred HPs are not catastrophically wrong — they're just increasingly sub-optimal. The loss landscape's optimal LR basin shifts, resulting in slower training or mild instability. The fix is straightforward: use an intermediate validation scale to catch this before committing to the full target run.

### Depth transfer remains the harder problem

μP was designed for width. Depth transfer requires additional machinery. Without depth-specific corrections, changing depth between proxy and target introduces HP shifts that μP cannot handle. Three approaches exist:

**Bordelon et al. (ICLR 2024)** proposed 1/√L residual branch scaling combined with μP, demonstrating transfer across both width and depth for ResNets and Vision Transformers. **Yang et al. (Tensor Programs VI)** provided theoretical foundations for a "Depth-μP" that divides each residual block's contribution by √L, identifying this as maximizing both feature learning and feature diversity. **CompleteP (Dey et al., 2025)** found that 1/L scaling (α=1) outperforms 1/√L (α=0.5) for practical transformers. CompleteP demonstrated transfer from **2 to 128 layers** — exceeding the depth of LLaMA-70B (80 layers) and LLaMA-405B (126 layers). Under standard μP without depth correction, only architectures with width-to-depth ratio ≥ 38.7 stay within 1% of compute-optimal; CompleteP reduces this to 11.8.

**Practical recommendation**: Match proxy depth to target depth whenever possible. If depth must change, use CompleteP's 1/L residual scaling. Never assume standard μP handles depth transfer.

### Token horizon is the most underappreciated constraint

This is where many practitioners get burned. **μP does not transfer across training duration** — the paper's own validation confirms this (Bjorck et al. tested μP models and found optimal LR still decreases with longer token horizon). The scaling law is remarkably clean: **LR*(D) = B · D^(−β)** with β ≈ 0.3, achieving R² of 0.96–0.99 across model sizes from 50M to 2.7B. The relationship is independent of model size, meaning the width correction (μP) and token-horizon correction compose cleanly.

The LLaMA-1 case study is illustrative. Bjorck et al. fit the scaling law for LLaMA-7B: B = 8.29×10⁻⁴, β = 0.3. The predicted optimal LR for 1T tokens was **1.15×10⁻⁴**. LLaMA-1 actually used **3×10⁻⁴** — too high by a factor of 2.5×. This suggests LLaMA-1 would have achieved better loss with a lower learning rate.

If training a proxy on 1B tokens but the target on 100B tokens (100× longer), the correction is: LR_target = LR_proxy × (1B/100B)^0.3 ≈ LR_proxy × 0.25. That is, **reduce LR by 4× for a 100× increase in token budget**. When combined with μP's width correction, the full transfer formula becomes:

**η_target = (η_base / m_d) × (D_proxy / D_target)^0.3**

Under Chinchilla-optimal training where both proxy and target train at 20 tokens per parameter, D scales with N, so both corrections apply simultaneously. For a 40M proxy (800M tokens) transferring to a 7B target (140B tokens): width correction × token correction = (1/m_d) × (800M/140B)^0.3 ≈ (1/m_d) × 0.19.

---

## 4. Optimizer-specific considerations

### Adam, AdamW, and the weight decay coupling problem

μP was designed for SGD and Adam. For AdamW — the standard LLM optimizer — the coupling between weight decay and learning rate creates a critical implementation issue described above. Use truly decoupled weight decay. Beyond this, standard Adam hyperparameters transfer well: **β₁ = 0.9 and β₂ = 0.95 are used consistently across all LLM scales** from hundreds of millions to hundreds of billions of parameters. These values can be kept constant.

**Epsilon** is more subtle. At large widths, gradient magnitudes can shrink enough that a fixed ε = 1e-8 dominates Adam's denominator, effectively converting Adam to SGD-like behavior. Everett et al. (ICML 2024) identified this as a key issue. Two solutions: scale epsilon with width, or use **Adam-atan2**, which replaces the standard `m/(√v + ε)` with `atan2(m, √v)`. Adam-atan2 is entirely scale-invariant and eliminates the epsilon hyperparameter altogether. It is worth using for principled scaling work at very large widths, though standard Adam with ε = 1e-8 works fine up to ~7B scale.

### Muon optimizer

Muon (MomentUm Orthogonalized by Newton-Schulz) is compatible with μP. Shah et al. (Essential AI, 2025) provided the first empirical demonstration of μP calibration for Muon, validated up to 4B parameters. A separate 2025 paper derived μP scaling rules for Muon, SOAP, and Shampoo, showing that combined with independent weight decay, these matrix-preconditioned optimizers achieve **consistent ~1.4× speedups over well-tuned AdamW** from 190M to 1.4B parameters. Without μP, all three optimizers deteriorate rapidly as scale increases.

Muon applies only to 2D (matrix) parameters; embeddings, output layers, normalization, and 1D parameters still use AdamW. The optimizer has an inherent scale-invariance property from orthonormalization, providing "built-in" spectral μP behavior. Only two HPs require tuning: peak learning rate and weight decay.

### Learning rate schedule transfer

**Schedule shape transfers across scales** — cosine decay, WSD (warmup-stable-decay), or linear decay can be kept constant between proxy and target. The peak LR magnitude needs μP scaling + token-horizon correction, but the shape is preserved. WSD has become increasingly popular because it decouples peak LR from total training length; the phase fractions (warmup/stable/decay) are robust across scales. Warmup duration expressed as a fraction of training steps (typically 1–2%) transfers well. For cooldown, recent work identifies **sqrt or "lowered-linear 0.7"** schedules as optimal, with decay-to-zero yielding better final loss than decaying to a fraction of peak LR.

---

## 5. Practical recommendations and common pitfalls

### When NOT to use HP transfer

HP transfer via μP makes clear economic sense for **from-scratch pretraining at ≥1B parameters**. The 200-sample sweep at 40M costs less than 1% of a single 7B training run. But there are scenarios where direct tuning at target scale is better:

- **Continued pretraining** (μP's theory doesn't cover non-random init; see Section 6)
- **Architecture mismatch** between proxy and target (different attention type, nonlinearity, position embeddings, or vocab size all invalidate transfer)
- **Batch size below critical threshold** at proxy scale
- **Dramatic token horizon differences** without applying the correction factor
- **Models not originally trained with μP** — you cannot retroactively apply μP transfer to an SP-trained model family

### The ten most common implementation bugs

1. **Using coupled weight decay** (PyTorch's default AdamW) instead of truly decoupled — this alone can sink μP
2. **Using 1/√m_d for output logits** instead of 1/m_d — works at init but breaks during training
3. **Forgetting to scale d_ffn** in the delta model for coordinate checks (must vary all width-dependent dims)
4. **Using 1/√d_head for attention** instead of 1/d_head — effects emerge slowly, so short coord checks may miss this
5. **Gradient clipping breaking μP** — with correct μP, clipping should be unnecessary; one practitioner reported "with clipping, my muP wasn't working at all"
6. **Using DataParallel** instead of DistributedDataParallel (DataParallel strips custom attributes that μP adds to parameters)
7. **Setting LR absolutely** in schedulers instead of relatively — `pg['lr'] = 1e-3 * 2` breaks μP; use `pg['lr'] *= 2`
8. **Proxy batch size below critical batch size** — transfer to larger model at/above CBS will be sub-optimal
9. **Not running coordinate check** before sweeping — hours of sweep compute wasted on a buggy implementation
10. **Ignoring token horizon correction** — sweeping on 1B tokens and training target on 100B tokens with the same LR leaves it 4× too high

### Open-source implementations

**Microsoft mup** (github.com/microsoft/mup) is the official PyTorch implementation. It provides drop-in replacements: `MuReadout` for output layers, `MuAdam`/`MuSGD` for optimizers, `set_base_shapes()` for width multiplier setup, and `get_coord_data()` for coordinate checks. Key limitation: base and target models must have the same depth. Does not support DataParallel. **EleutherAI/nanoGPT-mup** offers minimal working implementations with branches for standard μP, SuPar (sparse μP), and CompleteP (depth transfer). **Cerebras model-zoo** integrates μP for GPT-2, GPT-3, LLaMA, Falcon, and other architectures with configuration-driven setup (specify `mup_base_hidden_size` and `mup_base_filter_size`). **u-μP** (ICLR 2025) combines μP with Unit Scaling, eliminating the need for base shapes entirely and enabling out-of-the-box FP8 training — validated at 7B scale.

---

## 6. Why continued pretraining breaks the transfer paradigm

**μP's theoretical foundation assumes random initialization. CPT starts from a trained checkpoint — structured weights, correlated activations, and a loss landscape that differs qualitatively from random init.** No published work validates μP HP transfer for continued pretraining, and there are strong theoretical reasons to expect it will not work.

The core issue is that μP's scaling rules ensure activation magnitudes are controlled across widths *at initialization*. During from-scratch training, the theory predicts how updates evolve from this controlled starting point. But a pretrained 7B model's internal representations are already at a trained equilibrium — the weight distributions are structured and scale-dependent in ways that reflect the specific model size, training data, and training duration. Shrinking this model to 256 width doesn't produce a meaningful "proxy" of the pretrained state.

Additionally, CPT involves a distribution shift from pretraining data to new domain data. The optimal learning rate depends strongly on the magnitude of this shift and the "loss potential" remaining in the model. These factors are model-size-dependent in ways μP cannot capture: larger models forget less (per Poro 2 findings on LLaMA-3.1 8B vs 70B), so the forgetting-adaptation tradeoff inherently differs across scales.

### What practitioners actually do for CPT at 7B–70B

Rather than μP-style transfer, experienced teams use **direct LR sweeps at target scale with short runs**:

**Databricks (2024)** swept learning rates on LLaMA-2-7B for 1B tokens, finding "a substantial difference in improving MMLU and preventing forgetting across learning rates." They validated that findings transferred heuristically to 13B and 70B — using lower LR for larger models, not a principled scaling rule. **NVIDIA's "Reuse, Don't Retrain" (Parmar et al., 2024)** kept all HPs from pretraining (β₁=0.9, β₂=0.95, weight decay=0.1) and varied only the LR schedule, finding that matching the pretraining schedule shape (cosine if pretrained with cosine) was critical — WSD caused "significant regression in evaluation accuracy." **Poro 2 (AMD/ROCm, 2025)** ran shorter preliminary experiments on 8B before applying findings to 70B, finding larger models more robust to HP choices.

### Recommended CPT HP selection protocol

Since μP transfer is not viable for CPT, use this direct approach:

1. **Keep most HPs from original pretraining**: β₁ = 0.9, β₂ = 0.95, weight decay = 0.1, gradient clipping = 1.0, same schedule shape
2. **Sweep only the learning rate**: Test 5–10 values on a log scale (e.g., 1e-5, 2e-5, 5e-5, 1e-4, 2e-4) at target scale
3. **Run short sweeps**: 1–2B tokens per LR value is sufficient to discriminate
4. **Monitor both adaptation and forgetting**: Track domain-specific loss AND general benchmarks (MMLU, etc.)
5. **Use LR at 1/10 to 1/2 of original pretraining peak** as the search range
6. **For cross-size transfer within a model family** (e.g., tuning on 8B then applying to 70B): use a lower LR for the larger model and always validate with a 3-point check (LR/2, LR, LR×2) at target scale

The total cost of a 10-point LR sweep at 7B scale for 1B tokens each is ~10B tokens — roughly 10–20% of a typical 50–100B token CPT run. This is affordable and produces higher-confidence results than attempting to transfer from a proxy.

Warmup behavior in CPT differs from from-scratch training. Parmar et al. found that **no warmup achieves the best evaluation results** when starting from a well-trained checkpoint, though Gupta et al. (2023) showed re-warming helps when the distribution shift is large. For models pretrained with WSD, use the pre-decay checkpoint (not the fully annealed one) — it retains more "loss potential" for adaptation.

---

## Conclusion

Hyperparameter transfer via μP is a mature, well-validated technique for from-scratch pretraining that can reduce tuning costs by an order of magnitude. The core recipe — shrink width to 256, sweep four HPs over 200 random samples, transfer with η/m_d scaling — works reliably within a ~10–30× width ratio. But three corrections are essential and often overlooked: use truly decoupled weight decay (not PyTorch's default), reduce LR by (D_proxy/D_target)^0.3 for longer training horizons, and calibrate batch size independently via critical batch size estimation.

For continued pretraining, μP transfer is not applicable. The trained checkpoint violates μP's random-init assumption, and no published work validates cross-scale HP transfer for CPT. The pragmatic alternative — a 5–10 point LR sweep at target scale for 1–2B tokens, keeping all other HPs from pretraining — is affordable and produces reliable results. The most important insight for CPT is that learning rate is essentially the only HP that needs tuning; everything else can be inherited from the original pretraining configuration.

The frontier is moving toward unified transfer across all scaling dimensions. The Complete^(d)P framework (Mlodozeniec et al., 2025) transfers across width, depth, batch size, and training duration simultaneously, with per-module optimization at 50M scale transferring to budgets over 14,000× larger. Whether this eventually extends to non-random initialization — enabling principled HP transfer for CPT — remains an open research question.