## Learning rate schedule: WSD wins for CPT flexibility

### The core problem and solution

When a pretrained model finishes training with cosine decay, its learning rate sits near zero. Resuming CPT at that near-zero rate means the model barely adapts to new data. **Re-warming the LR is not optional—it is the single most critical step** for effective CPT (Gupta et al., arXiv:2308.04014). Skipping warmup causes loss spikes, gradient norm explosions, and potentially catastrophic forgetting that wastes the entire run.

Ibrahim et al. ("Simple and Scalable Strategies to Continually Pre-train Large Language Models," arXiv:2403.08763, 2024) validated on **405M and 10B parameter models** that re-warming to the original peak LR, re-decaying with cosine, and replaying 5% of original data matches full retraining. Gupta et al. (arXiv:2308.04014, 2023) confirmed on Pythia-410M that re-warming temporarily spikes loss on both upstream and downstream data, but ultimately yields better downstream performance than models trained from scratch.

### Schedule comparison

**WSD (Warmup-Stable-Decay)** is the strongest schedule choice for CPT sweeps. Introduced by the MiniCPM paper (Hu et al., arXiv:2404.06395, 2024), WSD divides training into three phases: linear warmup (~1–2% of steps), constant LR at peak (60–80% of training), and rapid decay to ~10% of peak (10–20% of training). Its key advantage for CPT is that **it does not require pre-specifying the total token budget**—you can hold the stable phase indefinitely, branch off a decay phase at any point, and share the stable-phase checkpoint across sweep experiments. WSD has been adopted by DeepSeek-V3, Kimi K2, Qwen 3, and many frontier models. Wen et al. (arXiv:2410.05192, 2024) provide theoretical justification via a "river valley" loss landscape model and propose WSD-S, a variant that recycles decay phases for multiple checkpoints from a single run.

**Cosine decay** remains a strong baseline. It requires knowing the total token budget upfront and decays to a minimum LR (typically 10% of peak, per Llama 3 and OLMo 2 conventions). Hägele et al. (arXiv:2405.18392, 2024) showed that constant-plus-cooldown (WSD-style) matches or beats cosine with proper tuning, and that re-warming from a fully-decayed cosine checkpoint causes recovery spikes. **Linear decay** performs comparably for shorter runs—OLMo 2 uses linear-to-zero during its mid-training CPT stage on 50–300B tokens (arXiv:2501.00656). **Constant LR (no decay)** shows an interesting property: a recent paper on Warmup-Stable-Only found that skipping decay consistently outperforms decay-based schedulers in post-SFT performance for 1B and 8B models, though pre-training loss is higher without decay.

### What peak LR to use

The optimal peak LR for CPT depends heavily on the token budget:


| CPT budget   | Recommended peak LR (relative to original) | Source                                                              |
| ------------ | ------------------------------------------ | ------------------------------------------------------------------- |
| >100B tokens | **1× (same as original)**                  | Ibrahim et al. 2024, Gupta et al. 2023                              |
| 5–50B tokens | **1/3× to 1/10× of original**              | TiC-LM (arXiv:2504.02107), "Reuse Don't Retrain" (arXiv:2407.07263) |
| <5B tokens   | **1/10× to 1/30× of original**             | TiC-LM 2025                                                         |


For the **5–20B token range** this guide targets: use approximately **1/3× to 1/10× of the original pretraining peak LR**. At 7B (where original peak is typically ~3e-4), this means a CPT peak of **3e-5 to 1e-4**. At 70B (where original peak is ~1.5e-4), aim for **1e-5 to 5e-5**. Code Llama (arXiv:2308.12950) used 1e-4 across 7B–70B for 500B tokens of code CPT; Llemma 34B (arXiv:2310.10631) used 5e-5 for 50B math tokens.

**Warmup duration**: 2000 steps is a safe default (used by Llama 3, OLMo 2). For CPT specifically, **1–2% of total steps** is typical. Gupta et al. found that the exact warmup length is surprisingly non-critical—what matters far more is reaching an appropriate peak LR.

**Decay minimum LR**: Set to **10% of peak** rather than zero. NVIDIA's "Reuse, Don't Retrain" (arXiv:2407.07263) showed that decaying all the way to zero hurts performance on late-stage data. For WSD, allocate **10–20% of total CPT tokens** to the decay phase.

### Checkpoint selection

A critical and often overlooked tip: if possible, resume CPT from an **intermediate checkpoint** before the final LR decay phase, not from the fully-converged endpoint. DeepSeek-Coder-V2 explicitly used an intermediate checkpoint of DeepSeek-V2 for this reason—the optimizer states are in a more "learning-ready" configuration and require less aggressive re-warming.

### Common pitfalls

- **Skipping re-warming entirely**: The model trains at near-zero LR and cannot adapt.
- **Re-warming to full original peak with a small CPT budget**: Causes severe forgetting that can't be recovered in 5–20B tokens.
- **Using cosine pegged to the wrong total step count**: Leads to premature or insufficient decay.
- **Decaying LR to zero**: The model cannot learn from late-stage high-quality data.

---

