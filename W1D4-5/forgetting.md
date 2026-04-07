
## Mitigations on Catastrophic Forgetting

### Full finetuning strongly outperforms LoRA for CPT

Biderman et al. ("LoRA Learns Less and Forgets Less," arXiv:2405.09673, TMLR 2024) provide the definitive comparison on Llama-2-7B and 13B. For code CPT on 20B tokens, the best LoRA (rank 256) achieved HumanEval=0.224 versus **0.263 for full finetuning**—and LoRA at 20B tokens roughly matched full FT at just 4B tokens, a **5× data efficiency gap**. The gap grows with more training data and is not closed even at high ranks. Full finetuning learns weight perturbations with **10–100× higher rank** than typical LoRA configurations.

Shuttleworth et al. (arXiv:2410.21228, 2024) found that LoRA creates "intruder dimensions"—high-ranking singular vectors dissimilar to pre-trained weights that cause forgetting and hurt continual learning. The optimal rank for minimal forgetting follows a U-shaped curve with **r=64** at the minimum.

**Decision framework**: Use full parameter CPT whenever compute allows. Use LoRA only for instruction finetuning (where it is competitive with full FT), memory-constrained scenarios, or when you need multiple domain-specific adapters served from the same base model. If forced to use LoRA for CPT, target all linear layers (not just Q/V), use rank 128–256, set α=2r, and use LR ~10× higher than full FT LR. QLoRA (arXiv:2305.14314) is a last resort for resource-constrained settings but compounds LoRA limitations with quantization noise.

### Replay is the most effective anti-forgetting technique

Simple data replay dominates all alternatives for CPT at LLM scale. Shi et al.'s comprehensive survey (arXiv:2404.16789, 2024) found that across 41 domain-adaptive pretraining papers, replay was the dominant approach, with LoRA/parameter expansion being the only alternatives used in practice. **EWC (Elastic Weight Consolidation)** is theoretically appealing but computationally expensive at LLM scale and has been superseded by simple replay. SAM (Sharpness-Aware Minimization) complements replay by flattening the loss landscape (Li et al., EMNLP 2024) but adds non-trivial compute overhead.

### Checkpoint averaging delivers free improvements

**LAWA (Lookahead Weight Averaging)** from Sanyal et al. (arXiv:2306.03241, 2023) averages k checkpoints sampled with spacing ν along a single training trajectory and **outperforms both EMA and SWA** for LLM pre-training. Evaluated on the Pythia suite (1B–12B), LAWA consistently improves performance and mitigates loss spikes. Models with higher learning rates benefit more. For large models, use **larger spacing between averaged checkpoints** since nearby checkpoints have low diversity.

**Model Soups** (Wortsman et al., arXiv:2203.05482, ICML 2022) average weights of multiple fine-tuned models with different hyperparameters. "Greedy soups"—iteratively adding models only if they improve validation performance—yield the best results. The technique works because fine-tuned models from the same pre-trained initialization lie in a single low-error basin.

**Practical recipe for CPT**: maintain **EMA with decay 0.9999** during training, save checkpoints every 5K steps after warmup, and at the end of training try averaging the last 5–10 checkpoints with spacing. Expect ~1–2% relative improvement on downstream tasks at zero inference cost.

### Evaluation: monitor both dimensions simultaneously

Track domain-specific perplexity and benchmarks (the primary optimization target) alongside general benchmarks (MMLU, HellaSwag, ARC) to detect forgetting. Plot the **learning-forgetting tradeoff curve** (Biderman et al. 2024): domain performance on the x-axis, general performance on the y-axis, with each checkpoint as a point. Stop when domain performance plateaus or general benchmarks drop by more than 2–5% relative. Evaluate every 1–5B training tokens, with full evaluation suites every 10–20B tokens.

An important nuance: many observed performance drops during CPT are **task alignment loss, not true knowledge loss** (the "Spurious Forgetting" phenomenon). The model retains knowledge but loses instruction-following format. A brief re-alignment SFT phase after CPT typically recovers these capabilities. Always start CPT from the **base model**, not an instruction-tuned variant—CPT from a chat model risks losing alignment that must be fully rebuilt.

---

## Quick reference table for CPT sweeps at 7B–70B

| Hyperparameter | Default | Sweep range | Notes |
|---|---|---|---|
| **LR schedule** | WSD | {WSD, cosine} | WSD preferred for flexible budgets |
| **Peak LR (7B)** | 1e-4 | {1e-5, 3e-5, 5e-5, 1e-4, 2e-4} | ~1/3× to 1× of original 3e-4 |
| **Peak LR (70B)** | 3e-5 | {1e-5, 2e-5, 3e-5, 5e-5} | More conservative at scale |
| **Min LR** | 10% of peak | {0, 5%, 10%} of peak | Avoid decaying to zero |
| **Warmup** | 2000 steps | {0, 500, 1000, 2000} steps | Peak LR matters more |
| **Batch size** | 4M tokens (7B), 8–16M (70B) | {1M, 2M, 4M, 8M} tokens | √ scaling with LR |
| **Optimizer** | AdamW | — | β₁=0.9, β₂=0.95, ε=1e-8 |
| **Weight decay** | 0.1 | {0.01, 0.05, 0.1, 0.3} | Decoupled (AdamW) |
| **Gradient clip** | 1.0 | Rarely sweep | Monitor activation frequency |
| **Replay ratio** | 5% | {2%, 5%, 10%, 20%} | Higher for stronger domain shift |
| **Dropout** | 0.0 | {0.0, 0.05, 0.1} | Non-zero only for multi-epoch |
| **Decay fraction (WSD)** | 15% of tokens | {10%, 15%, 20%} | — |
| **Method** | Full FT | {Full FT, LoRA r=256} | Full FT strongly preferred |
| **EMA decay** | 0.9999 | — | Free improvement |
| **Z-loss** | 1e-4 | {0, 1e-4} | Safe stability addition |

## Conclusion

Three decisions dominate CPT outcomes at 7B–70B scale, and everything else is secondary. **First**, re-warm the learning rate—without it, the model cannot adapt. For 5–20B token budgets, target 1/3× to 1/10× of the original pretraining peak. **Second**, replay 5% of general pretraining data to anchor general capabilities while learning the new domain. **Third**, use full parameter finetuning, not LoRA—the data efficiency gap is approximately 5× and grows with training duration. WSD scheduling eliminates the need to commit to exact token budgets upfront, making sweep design substantially more efficient. AdamW with β₂=0.95 and weight decay 0.1 remains the only optimizer validated for CPT at scale. μP can reduce sweep costs by enabling HP transfer from small proxies, though its applicability to CPT (versus from-scratch training) remains theoretically unvalidated. The most underappreciated technique is checkpoint averaging via LAWA—it delivers 1–2% improvements at zero additional inference cost and should be default practice for any CPT run.