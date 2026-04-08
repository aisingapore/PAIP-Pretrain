
## AdamW: the industry standard configuration

### The update rule

> *Why does every major lab use nearly identical Adam settings, but the original Adam paper's defaults (β₂=0.999) are wrong for LLMs?*

Because LLM training operates in a regime—large batch, long horizon, shifting data distributions—that the original Adam paper (Kingma & Ba, 2014) did not anticipate. The community converged on a modified recipe through years of large-scale experimentation.

**AdamW** (Loshchilov & Hutter, 2017) computes per-parameter adaptive learning rates using exponential moving averages of the gradient (first moment) and squared gradient (second moment), with **decoupled weight decay**:

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_{t+1} = (1 - \lambda) \cdot \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

where $g_t$ is the gradient, $\eta$ is the learning rate, $\lambda$ is the weight decay coefficient, and the hat notation denotes bias correction (necessary because $m_0 = v_0 = 0$).

**Why decoupled weight decay matters.** In the original Adam with L2 regularization, the weight penalty is added to the gradient *before* the adaptive scaling: $g_t' = g_t + \lambda\theta_t$, then the adaptive update is applied to $g_t'$. This means parameters with large gradients get *less* effective weight decay (Adam's denominator downscales the update), while parameters with small gradients get *more*—the opposite of what you want. AdamW fixes this by applying weight decay directly to the weights (the $(1-\lambda)\cdot\theta_t$ term), bypassing the adaptive scaling entirely. Every LLM training recipe uses the decoupled formulation.

### Configuration parameters

**β₁ = 0.9, β₂ = 0.95, weight_decay = 0.1, gradient_clipping = 1.0** is the universal recipe used by Llama 2/3, DeepSeek-V3, Qwen 2.5, and OLMo 2.


| Parameter | Default | Range | Intuition |
| --------- | ------- | ----- | --------- |
| β₁ | 0.9 | — | EMA of gradient mean (momentum). Effective window ≈ 1/(1-β₁) = 10 steps. Universal across all labs; not worth tuning for CPT. |
| β₂ | **0.95** | — | EMA of squared gradient (adaptive LR scaling). Effective window ≈ 1/(1-β₂) = 20 steps. The original Adam default (0.999, ~1000-step window) is far too slow—gradient variance estimates lag behind distribution shifts during CPT, causing unstable per-parameter step sizes. |
| ε | 1e-8 | 1e-8 to 1e-6 | Denominator stabilizer. Prevents division by near-zero when $\hat{v}_t$ is tiny (parameters with very small gradients). Qwen uses **1e-6** for mixed-precision (bf16) safety—at reduced precision, 1e-8 can underflow in the denominator. |
| weight_decay (λ) | 0.1 | 0.01–0.3 | Decoupled L2 penalty. Prevents weight norm explosion over long runs. Higher values (0.3–1.0) may increase plasticity for downstream adaptation (Han et al., 2025). |
| clip_grad | 1.0 | — | Global gradient norm clipping. If >10% of steps trigger clipping, the LR is too high. Rarely needs adjustment. |


**MegatronBridge config:** `optimizer: adam` (default), `adam_beta1`, `adam_beta2`, `adam_eps`, `weight_decay`, `clip_grad`, `decoupled_weight_decay`.

For detailed per-parameter tuning guidance, sweep ranges, and Socratic intuitions, see `3-hparam.md`.

### Optimizer state reset for CPT

For CPT, the AdamW hyperparameters above do not need to change from pretraining. The main decision is what to do with optimizer states ($m_t$ and $v_t$):

- **Starting from a public checkpoint** (most common): optimizer states are not released. **Reset** to fresh initialization ($m_0 = v_0 = 0$). LR re-warming compensates for the missing state information—$\hat{v}_t$ calibrates during warmup.
- **Continuing your own model** (you have the states): two options. (1) Continue with existing states plus gentle re-warming—works well when the domain shift is small. (2) Reset states plus full re-warming—more robust for large distribution shifts, because stale $v_t$ estimates from the old data distribution can cause unstable step sizes on the new distribution.

The recommendation: **reset states and re-warm** unless you have strong reason to believe the gradient statistics are similar across domains. This is the safer default.

---

## Muon: orthogonalized momentum

### Core algorithm

> *Adam treats each weight as an independent scalar and adapts its step size individually. What if the gradient's matrix structure contains useful information that Adam throws away?*

Muon (MomentUm Orthogonalized by Newton-Schulz), introduced by Keller Jordan (kellerjordan.github.io/posts/muon/), answers this question. It treats weight matrices as linear operators and orthogonalizes the gradient update so that all spectral directions receive equal update magnitude. Moonshot AI's scaling paper ("Muon is Scalable for LLM Training," arXiv:2502.16982, 2025) demonstrated ~2× computational efficiency over AdamW in scaling law experiments and trained Moonlight, a 3B/16B MoE model on 5.7T tokens.

**Algorithm for a weight matrix W of shape m × n:**

1. **Compute gradient**: $G_t = \nabla_W \mathcal{L}$

2. **Accumulate momentum** (SGD-style, not Adam-style):

$$M_t = \beta \cdot M_{t-1} + (1 - \beta) \cdot G_t, \quad \beta = 0.95$$

3. **Orthogonalize via Newton-Schulz iteration.** Initialize:

$$X_0 = \frac{M_t}{\|M_t\|_F}$$

Then iterate for $k = 1, \ldots, K$ (default $K = 5$):

$$X_k = a \cdot X_{k-1} + b \cdot X_{k-1} X_{k-1}^\top X_{k-1} + c \cdot (X_{k-1} X_{k-1}^\top)^2 X_{k-1}$$

The coefficients $(a, b, c)$ depend on the `coefficient_type` (default "quintic") and are chosen so that the iteration converges to the matrix sign function. After $K$ steps, $X_K$ has approximately equal singular values—the spectral structure of the gradient has been "flattened."

4. **Scale**: $U = X_K \cdot \sqrt{\frac{\max(m, n)}{\min(m, n)}} \cdot \text{extra\_scale\_factor}$

The spectral scaling factor $\sqrt{\max(m,n)/\min(m,n)}$ ensures that layers of different shapes receive updates of comparable magnitude.

5. **Update**: $W_{t+1} = W_t - \eta \cdot U$

**Intuition.** A weight matrix's gradient can be decomposed via SVD into spectral directions. In standard Adam, a few dominant singular values can capture most of the gradient energy, causing the optimizer to repeatedly update the same spectral directions while neglecting others. Muon's orthogonalization equalizes all singular values, ensuring every direction in weight space gets equal attention. This is analogous to preconditioning with the inverse square root of the gradient covariance—but computed cheaply via Newton-Schulz rather than explicit eigendecomposition.

### Why Muon only works on ≥2D weight matrices

> *A weight matrix has spectral structure (singular values, left/right singular vectors). A bias vector does not. Why does this distinction matter for orthogonalization?*

The Newton-Schulz iteration requires the $X X^\top$ product, which is only meaningful for matrices with both row and column dimensions. For **1D parameters** (biases, LayerNorm scale and shift), there is no matrix structure—they are just vectors with a single "direction," so spectral equalization is undefined.

**Embeddings** are 2D (shape: vocab_size × hidden_dim) but are excluded for a different reason: their gradients are **sparse**. In each batch, only the rows corresponding to tokens actually present receive nonzero gradients. Orthogonalization would spread the gradient signal across all rows—including rows for tokens not in the batch—destroying the natural sparsity structure and wasting the update. The output projection (lm_head) is similarly excluded because it shares this discrete-access pattern (and is often tied to the embedding table).

**Fallback mechanism.** Megatron uses a `ChainedOptimizer` that runs two optimizers sequentially each step:
- **Muon** updates all 2D linear weights (attention projections, MLP layers)
- **AdamW** updates everything else (embeddings, output projection, biases, LayerNorm parameters)

This means a Muon training run always involves *two* optimizers with *two* sets of hyperparameters.

### Muon optimizer state and memory

> *Muon uses SGD-style momentum instead of Adam's two-state (m, v) design. What does this buy you?*

Muon stores only one buffer per parameter:

| Optimizer | State per parameter | Memory (per param element) |
| --------- | ------------------- | -------------------------- |
| AdamW | `exp_avg` (m) + `exp_avg_sq` (v) | 2× param size (8 bytes in fp32) |
| Muon | `momentum_buffer` (M) | 1× param size (4 bytes in fp32) |

For the 2D-weight portion of the model (which is the large majority—~80-90% of parameters in a typical transformer), Muon saves ~50% of optimizer memory compared to AdamW. However, the AdamW fallback for non-linear parameters still needs the full two-state storage. Net memory savings depend on the fraction of parameters that are 2D linear weights.

### Configuration parameters

| Parameter | Default | Impact |
| --------- | ------- | ------ |
| `momentum_beta` | 0.95 | SGD momentum decay (not the same as Adam β₁). Higher = smoother updates but slower adaptation to distribution shifts. |
| `use_nesterov` | True (MegatronBridge) | Nesterov look-ahead momentum. Evaluates the gradient at the "future" position, giving a slight convergence improvement. |
| `num_ns_steps` | 5 | Number of Newton-Schulz iterations. 5 is sufficient for convergence; the approximation quality plateaus after ~5 steps. Must be ≥1. |
| `scale_mode` | "spectral" | How the post-orthogonalization scale factor is computed. "spectral" uses $\sqrt{\max(m,n)/\min(m,n)}$, balancing update magnitude across layers of different shapes. |
| `extra_scale_factor` | 1.0 | Additional multiplier applied after spectral scaling. Use to fine-tune the overall Muon update magnitude relative to the AdamW fallback. |
| `coefficient_type` | "quintic" | Polynomial coefficients for the NS iteration. "quintic" uses 5th-order coefficients for faster convergence. |
| `fp32_matmul_prec` | "medium" | Precision for the matrix multiplications inside NS iteration. "medium" balances numerical accuracy and throughput. |
| `split_qkv` | True | Split grouped QKV attention weights and orthogonalize Q, K, V components separately. Without this, a single orthogonalization across the concatenated QKV matrix would mix query/key/value subspaces, destroying the attention structure. |
| `tp_mode` | "blockwise" | How TP-sharded weights are handled during orthogonalization. "blockwise": each TP shard is orthogonalized independently (no cross-rank communication). "duplicated": full-matrix NS after gathering. "distributed": NS iteration communicates across TP ranks. |


### Not validated for CPT

**Recommendation: Use AdamW for CPT at 7B–70B scale.** Three factors argue against Muon for CPT:

- **No CPT-specific validation exists.** All published Muon results are for from-scratch pretraining. The Moonlight paper found that when the SFT optimizer differs from the pretraining optimizer, Muon SFT shows no significant advantage—a concerning signal for switching optimizers mid-training, which is exactly what CPT requires if the base model was pretrained with AdamW.
- **Diminishing returns at scale.** "Fantastic Pretraining Optimizers and Where to Find Them" (Wen et al., Stanford, arXiv:2509.02046, 2025) showed that Muon's speedup over **well-tuned** AdamW drops from ~1.4× at 0.1B to ~1.1× at 1.2B and may vanish at 7B+. Much of Muon's reported advantage comes from comparisons against poorly-tuned AdamW baselines.
- **Implementation complexity.** The dual-optimizer pattern (Muon + AdamW fallback), QKV splitting, TP mode selection, and Emerging-Optimizers dependency add significant operational burden compared to a single AdamW optimizer.

Consider Muon only if you are doing very large-scale CPT (hundreds of billions of tokens) that resembles continued full pretraining, or if you are training from scratch with Muon and want to maintain optimizer consistency.

---

## Other optimizers worth knowing about

**SOAP** (arXiv:2409.11321, 2024) combines Shampoo's preconditioning with Adam and shows 35–40% wall-clock improvements over AdamW in large-batch settings. It outperforms Muon in overtraining scenarios (≥8× Chinchilla data ratio) but has higher memory overhead and implementation complexity. **Schedule-Free Adam** (Defazio et al., Meta, 2024) eliminates the LR schedule entirely through interpolation and iterate averaging—attractive for CPT because it removes the need to specify a stopping time, but lacks large-scale CPT validation. **Lion** (arXiv:2302.06675, 2023) offers 50% less optimizer memory than AdamW but requires 3–10× smaller LR and larger weight decay, with no clear advantage over well-tuned AdamW for text LLMs.

---

## Using Muon in the MegatronBridge stack

> *This section is a practical walkthrough for teams that want to experiment with Muon despite the CPT caveats above. It covers the dependency chain, how parameters are split between optimizers, how state is distributed across parallelism dimensions, and the YAML configuration.*

### Dependency: Emerging-Optimizers

Muon's implementation in Megatron-LM depends on NVIDIA's **Emerging-Optimizers** library, which provides the `OrthogonalizedOptimizer` base class and the `newton_schulz_tp()` utility for tensor-parallel-aware Newton-Schulz iteration.

**Install:**

```bash
pip install git+https://github.com/NVIDIA-NeMO/Emerging-Optimizers.git@v0.1.0
```

In MegatronBridge, set the `eo_dir` field in the cluster config section of `config.yaml` to point to your Emerging-Optimizers checkout. The launcher adds this path to `PYTHONPATH` at runtime.

**Graceful fallback:** Megatron wraps the import in a try/except. If the package is missing, `HAVE_EMERGING_OPTIMIZERS` is set to `False` and the code raises an assertion at optimizer creation time—not at import time. This means you can import the muon module without the dependency installed; it only fails when you actually try to create a Muon optimizer.

### Parameter splitting: what gets Muon vs AdamW

The function `get_megatron_muon_optimizer()` in `megatron/core/optimizer/muon.py` implements a **freeze-swap-freeze** pattern to create two optimizers that together cover all parameters:

**Step 1 — Classify parameters.** Iterate all named parameters in the model:

- **Linear params** (→ Muon): 2D weight tensors that are NOT embedding or output parameters. These are the attention projections (Q, K, V, O) and MLP layers (gate, up, down).
- **Nonlinear params** (→ AdamW): everything else—embedding tables, output projection (lm_head), all 1D parameters (biases, LayerNorm scales/shifts).
- **Special flags**: `param.expert_tp = True` for MoE expert weights (uses a separate TP group), `param.is_qkv = True` for attention QKV weights (triggers split orthogonalization).

**Step 2 — Create Muon optimizer.** Freeze all nonlinear params (`requires_grad = False`), then create `TensorParallelMuon` with only the linear parameter groups.

**Step 3 — Create AdamW optimizer.** Freeze all linear params, unfreeze nonlinear params, then call the standard `get_megatron_optimizer()`. Because linear params are frozen, AdamW only creates state for the nonlinear params.

**Step 4 — Unfreeze everything.** Restore `requires_grad = True` on all parameters.

**Step 5 — Chain.** Return `ChainedOptimizer([muon_optimizer, adam_optimizer])`. Each training step calls `.step()` on both optimizers sequentially.

**QKV special handling.** When `split_qkv=True`, the QKV weight matrix (which concatenates query, key, and value projections) is split along the head-group dimension into three components. Each component is orthogonalized separately, then concatenated back. This preserves the distinct roles of Q, K, and V—a single orthogonalization across the full QKV matrix would mix these subspaces and destroy the attention structure.

### Optimizer state distribution across parallelism dimensions

| Parallelism | Muon behavior | Communication |
| ----------- | ------------- | ------------- |
| **TP** (tensor) | Each TP rank holds its shard of the weight matrix. With `tp_mode="blockwise"` (default), each shard is orthogonalized independently—**no cross-rank communication** for the NS iteration. With `tp_mode="distributed"`, NS iteration communicates across TP ranks to orthogonalize the full logical matrix. | Blockwise: zero extra comms. Distributed: collectives within TP group during each NS step. |
| **DP** (data) | Standard DDP gradient all-reduce. **Muon does NOT support Megatron's `use_distributed_optimizer`** (which shards optimizer state across DP ranks). Each DP rank holds the full optimizer state for its parameters. | Gradient all-reduce per backward pass (same cost as AdamW with standard DDP). |
| **DP (layer-wise)** | `LayerWiseDistributedOptimizer` is the alternative for memory savings. It shards parameters by layer across DP ranks using **ping-pong balancing** (sorts layers by size, assigns alternately to balance memory). Each rank only runs the optimizer step for its assigned layers. | One `all_gather` per step across the `dp_cp` group to broadcast updated parameters. Volume: ~2× total model parameter size. |
| **CP** (context) | Treated as part of DP for optimizer purposes. Communication happens over the combined `dp_cp` group (all DP + CP ranks together). | Same as DP. |
| **EP** (expert) | Expert parameters are handled by a **separate `TensorParallelMuon` instance**. Uses the `expt_tp` group (expert tensor parallelism) for NS iteration instead of the regular `tp` group. Gradient sync happens over the `expt_dp` group. | Separate gradient sync for expert params; NS iteration within expert TP group. |
| **PP** (pipeline) | Each PP stage has its own optimizer instance operating independently on that stage's parameters. No cross-stage optimizer communication. | None specific to optimizer. |

**Key constraint:** `use_distributed_optimizer = False` is **mandatory** for Muon. This is enforced in both `resolve_config.py` and `muon.py` (raises an exception otherwise). The standard Megatron distributed optimizer is tightly coupled to DDP's gradient buffer initialization and is incompatible with the `ChainedOptimizer` pattern. If you need to save optimizer memory, use `LayerWiseDistributedOptimizer` instead.

### Communication overhead comparison

| Operation | AdamW (standard DDP) | AdamW (dist_optimizer) | Muon (default) | Muon (layer-wise dist) |
| --------- | -------------------- | ---------------------- | -------------- | ---------------------- |
| Gradient sync | all-reduce | reduce-scatter | all-reduce | all-reduce |
| Param sync | — | all-gather | — | all-gather |
| NS iteration comms | — | — | none (blockwise) | none (blockwise) |
| Extra volume per step | 0 | ~4× params | 0 | ~2× params |

With `tp_mode="blockwise"` (default), Muon adds **zero communication overhead** beyond standard gradient all-reduce—the Newton-Schulz iteration is entirely local to each rank. The `LayerWiseDistributedOptimizer` adds one `all_gather` per step to broadcast updated parameters, with total volume roughly 2× the model parameter size.

With `tp_mode="distributed"`, additional collectives occur within the TP group during each of the 5 NS iterations—this can be significant for large TP sizes (≥4) and is one reason "blockwise" is the default.

### MegatronBridge YAML configuration

**Minimal config to enable Muon:**

```yaml
# In experiment YAML
optimizer: muon           # triggers apply_optimizer_config() in resolve_config.py
weight_decay: 0.01        # applied to both Muon and AdamW fallback
```

This sets `cfg.optimizer.optimizer = "muon"`, forces `use_distributed_optimizer = False`, and enables Nesterov momentum.

**Full Muon-specific keys** (set via `OptimizerConfig`; defaults shown—usually no need to override):

```yaml
muon_momentum: 0.95           # SGD momentum for Muon's internal optimizer
muon_use_nesterov: true       # auto-set by resolve_config.py
muon_num_ns_steps: 5          # Newton-Schulz iterations
muon_scale_mode: spectral     # scaling factor computation
muon_extra_scale_factor: 1.0  # additional scale multiplier
muon_fp32_matmul_prec: medium # precision for NS matmuls
muon_split_qkv: true          # orthogonalize Q, K, V separately
muon_tp_mode: blockwise       # TP handling mode
```

**Constraints enforced by code:**
- `use_distributed_optimizer` must be `false` (raises exception)
- `fp16` must be `false` (only `bf16` is supported with Muon)
- `num_ns_steps ≥ 1`

**Cluster config requirement:** Set `eo_dir` in the cluster section of `config.yaml` to the Emerging-Optimizers checkout path:

```yaml
cluster:
  smc:
    eo_dir: /path/to/Emerging-Optimizers
```

The launcher script adds this to `PYTHONPATH` so Megatron can import `emerging_optimizers` at runtime.

---
