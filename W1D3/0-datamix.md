# Week 1, Day 3: Data-Mixing, Selection

# Data mixing and selection for multilingual continued pretraining

**The single most important decision in multilingual CPT is not which model to train—it's how you compose the training data.** Recent work from Poro 2, Sailor 2, and Swallow demonstrates that principled data mixing can produce a 7B model that outperforms 70B baselines on target languages while preserving English and reasoning capabilities. The key formula: allocate **60–75% to new languages** (balanced via UniMax or RegMix), **20–30% to English/capability replay**, and **5–10% to code and math**. Train in two phases, with quality escalation in the final 10–20%. This report synthesizes concrete numbers, formulas, and lessons from every major multilingual CPT project published between 2023 and 2025.

---

## 1. How temperature sampling and UniMax tame the 50× data imbalance

The foundational challenge of multilingual CPT with 1B–50B tokens per language is sampling strategy. Naive proportional sampling lets high-resource languages dominate; uniform sampling forces extreme repetition on low-resource languages. Two families of solutions have emerged.

### Temperature-based sampling

The mT5 formula remains the most widely used starting point:

$$p_l = \frac{n_l^\alpha}{\sum_i n_i^\alpha}$$

where $n_l$ is the token count for language $l$ and $\alpha$ is the temperature exponent. At $\alpha = 1$, sampling is proportional (high-resource dominates); at $\alpha = 0$, sampling is uniform (massive repetition of scarce data). **The empirically validated sweet spot is α = 0.3**, used by both mT5 and XLM-R. With 50× variance between languages, α = 0.3 compresses the sampling ratio from 50:1 down to roughly **5.6:1**—a substantial flattening that still respects data availability.

| Model | α value | Effect |
|-------|---------|--------|
| mBERT | 0.7 | Mild flattening, favors high-resource |
| XLM / M4 | 0.5 / 0.2 | Moderate to strong flattening |
| **mT5 / XLM-R** | **0.3** | **Best empirical compromise** |

### UniMax: the superior alternative

UniMax (Chung et al., ICLR 2023) exposes a fundamental flaw in temperature sampling: at τ = 3.33 (α = 0.3) with a 1T token budget, the lowest-resource languages in mC4 are repeated **over 100 times**. UniMax fixes this with an explicit repetition cap. The algorithm pre-allocates tokens to underrepresented languages based on a maximum epoch count $N$, then distributes remaining budget uniformly across languages with sufficient data. **With N = 3–5, UniMax consistently outperforms temperature sampling** across all model scales (Large, XL, XXL) on TyDi QA and WMT benchmarks. Benefits persist and even increase with model scale.

For the 10–20 language scenario with 1B–50B tokens per language, UniMax with $N = 3$ means a 1B-token language contributes at most 3B tokens, while a 50B-token language gets its fair uniform share. This naturally handles the variance without excessive repetition. **UniMax should be the default choice over temperature sampling for CPT.**

### The repetition ceiling: 4 epochs maximum

"Scaling Data-Constrained Language Models" (Muennighoff et al., NeurIPS 2023) studied 400 training runs and established that **up to 4 epochs of repeated data yields negligible loss compared to unique data** at matched compute. Beyond 4 epochs, diminishing returns accelerate sharply. At **~16 epochs, repeated tokens provide half the value** of unique tokens. Beyond that, returns flatten, and at 44+ epochs, loss can actually increase mid-training. The practical implication is unambiguous: **50B unique tokens seen once is substantially better than 10B tokens seen 5 times.** For data-scarce languages, cap repetition at 4 epochs and invest remaining compute budget elsewhere.

---

## 2. Optimizing mixing ratios with proxy models

Beyond heuristic formulas, three principled methods can find near-optimal mixing ratios using small proxy models at 2–10% of final training compute.

### RegMix: the most efficient approach

RegMix (Liu et al., ICLR 2025 Spotlight) treats mixture optimization as a regression problem. The recipe: generate ~512 random data mixtures via Dirichlet sampling, train **1M-parameter proxy models** on each for ~1B tokens, fit Ridge regression with mixture proportions as features and validation loss as labels, then simulate millions of candidate mixtures to find the optimum. **This costs ~2% of final training FLOPs and matches or exceeds DoReMi.** Sailor 2 adopted RegMix with 1,000 proxy runs and found it upsampled Khmer, Malay, Burmese, Thai, and Tagalog while downsampling Vietnamese and Indonesian—a non-obvious result that pure heuristics would miss.

### DoReMi: minimax optimization

DoReMi (Xie et al., NeurIPS 2023) trains a small reference model, then trains a proxy model using Group Distributionally Robust Optimization that upweights domains where the proxy shows high excess loss relative to the reference. The extracted weights transfer to 30× larger models. DoReMi achieves **+6.5% average few-shot accuracy** over baseline weights on The Pile and reaches baseline accuracy **2.6× faster**. However, it costs ~10% extra compute and shows occasional instability, making RegMix generally preferable for multilingual CPT.

### Scaling laws that actually transfer

He et al. (ACL Findings 2025) validated a power-law relationship linking performance with dataset size, model size, and sampling ratios across 23 languages, finding that **optimal sampling ratios from 85M-parameter models generalize to models orders of magnitude larger** (1.2B+). ATLAS (Longpre et al., ICLR 2026) — the largest public multilingual scaling study at 774 training runs across 400+ languages — introduced cross-lingual transfer matrices and found a counterintuitive result: **English is among the least effective "donor" languages for cross-lingual transfer; German and Russian are more effective sources** for many language families. This suggests including related high-resource languages in the training mix, even if they aren't explicit targets.

---

## 3. Replay ratios: the forgetting–learning tradeoff with real numbers

Catastrophic forgetting is the central risk of multilingual CPT. The empirical evidence from multiple projects converges on clear guidelines.

### The Ibrahim et al. framework

"Simple and Scalable Strategies to Continually Pre-train Large Language Models" (Ibrahim et al., 2024) establishes the canonical recipe: **LR re-warming + LR re-decaying + data replay**. This combination matches training from scratch. Their experiments at 405M and 10B parameters tested replay from 0.5% to 50%:

- **Weak distribution shift** (English→English, similar domains): **5% replay** is sufficient. Even 1% provides substantial protection.
- **Strong distribution shift** (English→German): **25% replay** is needed. Without replay, upstream loss increased by 1.39; with 25% replay, the increase dropped to 0.16.

For multilingual CPT on 10–20 languages—a moderately strong shift from any modern base model—the evidence points to **20–30% replay as the sweet spot.**

### What successful projects actually use

| Project | Base Model | CPT Budget | Replay % | New Lang % | Code/Math % |
|---------|-----------|------------|----------|------------|-------------|
| **Poro 2** | Llama 3.1 8B/70B | 165B | 25% English | 70% Finnish | 4% code + 1% parallel |
| **Sailor 2** | Qwen 2.5 7B→8B | 500B | 20% (En+Zh+Math) | ~70% SEA langs | Math in replay; no code |
| **Swallow** | Llama 2/3 7B–70B | 100B+ | English mixed in | Majority Japanese | Code included |
| **Nemotron-Hindi** | Nemotron 4B | 400B | 50% English | 50% Hindi | — |
| **SEA-LION v3** | Gemma 2 9B | 200B | Included in mix | SEA languages | Code included |

**Poro 2 provides the most thoroughly documented evidence.** Their baseline mix of 70% Finnish / 25% English / 4% code / 1% parallel text yielded ~6 percentage points average Finnish improvement but ~10 percentage points English decline—especially in math. Their playbook states explicitly: *"For strong distribution shifts, replay of up to 50% original domain data may be needed."* **The critical finding: excluding math data caused massive math degradation for the 8B model but much less for the 70B**, confirming that larger models are inherently more resilient to forgetting.

### Capability-specific replay is non-negotiable

The evidence is overwhelming that replay must be domain-targeted, not just generic English text. Poro 2 demonstrated that replacing 4% code data with 4% math data substantially maintained English math performance and boosted Finnish math. Sailor 2 included Open-Web-Math-Pro in replay but deliberately excluded code to protect multilingual performance. The recommended replay budget allocation for 10–20 language CPT:

- **English high-quality text** (FineWeb-Edu, RedPajama): 12–18% of total budget
- **Math data** (FineMath, Open-Web-Math, Proof-Pile): 3–5%
- **Code data** (The Stack filtered subset): 3–5%
- **Parallel/translation data**: 1–3%

**Total replay: 20–30%.** Adjust upward toward 30% for 7B models, downward toward 20% for 70B models.

---

## 4. Data quality filtering without destroying scarce data

Quality filtering for low-resource languages requires a fundamentally different approach than English-centric pipelines. The most important lesson from recent work: **aggressive filtering can be catastrophic for data-scarce languages.**

### The FineWeb-2 warning

FineWeb-2's ablation on 9 languages revealed a critical finding: for Swahili (~1B filtered tokens), **applying quality filters performed worse than deduplication-only** (~3B tokens). Over-filtering destroyed too much scarce data. This finding should inform every filtering decision for languages with fewer than ~5B tokens. The practical rule: **for languages under 5B tokens, apply only deduplication and basic heuristic cleaning; skip model-based quality filtering.** For languages with 10B+ tokens, apply progressively more aggressive filtering.

### Recommended filtering pipeline by resource level

**High-resource languages (>10B tokens):** Use MLP classifiers on XLM-RoBERTa embeddings (FineWeb2-HQ approach) to retain the **top 10–15%**. Alternatively, FastText quality classifiers trained on Wikipedia-like text as positive examples and random CommonCrawl as negative, applying DCLM's approach. FastText is competitive with transformer-based classifiers and far cheaper computationally.

**Mid-resource languages (3–10B tokens):** Apply FastText quality classifiers at a **top 20–30%** threshold. Use per-language KenLM perplexity scoring with Wikipedia-trained models and **80th percentile thresholds**. Apply CulturaX-style IQR-based per-language threshold adaptation rather than fixed global cutoffs.

**Low-resource languages (<3B tokens):** Apply **only MinHash deduplication** (Jaccard τ = 0.8, 5-gram windows) and basic heuristic cleaning (remove boilerplate, obvious spam, encoding errors). Do not apply perplexity filtering or classifier-based selection. Every token matters.

### CommonCrawl quality issues specific to low-resource languages

MADLAD-400's audit of 498 languages found that **79 languages (16%) were majority noise** and had to be removed entirely. Low-resource language data from CommonCrawl is plagued by five specific issues: religious text dominance (Bible, jw.org accounting for nearly all data in some languages), language misidentification acting as a "garbage collector" for noise, encoding problems (virama characters in Brahmic scripts, Zawgyi vs. Unicode for Myanmar), extreme duplication (up to 70% at paragraph level), and pornographic/harmful content that standard filters miss. Solutions include GlotLID for language identification (2000+ labels, script-aware), FTFY for encoding fixes, per-language confidence thresholds calibrated via downstream evaluation, and human spot-checking of 20+ documents per language before committing to a corpus.

### Domain upweighting within each language

When a language has multiple data sources, the empirical evidence favors substantial upweighting of curated sources. Sailor 2 trained FastText classifiers per language using machine-translated high-quality English documents as positive examples, then used these classifiers to identify the **top 10–20%** of web data. Wikipedia should be upsampled **2–5×** relative to its natural proportion. The synthetic data literature suggests a ratio of roughly **1/3 high-quality curated + 2/3 filtered web** as optimal—pure curated data underperforms web data alone, but the mixture beats both.

---

## 5. Phase-based training and curriculum design

Nearly every successful multilingual CPT project published in 2024–2025 uses multi-phase training with quality escalation. The evidence for this approach is now strong enough to consider it standard practice.

### The two-phase consensus

**Phase 1 (80–90% of token budget):** Broad data distribution with balanced language mixing. Use UniMax or RegMix-optimized weights. Standard learning rate with cosine or trapezoidal schedule. This phase builds broad language competency.

**Phase 2 / Annealing (10–20% of token budget):** Shift to high-quality data subset. Reduce learning rate to **1/10 of Phase 1 peak** (Sailor 2) or anneal linearly to zero (Llama 3, EuroLLM). Upsample high-quality sources (Wikipedia, textbooks, educational content, curated parallel data). Consider Polyak averaging of late checkpoints.

Salamandra provides the most explicit multi-phase example: epochs 1–3 used standard web data (2.4T tokens/epoch), epochs 4–5 replaced English OSCAR with FineWeb-Edu (higher quality), and the final 315B tokens used only the highest-quality data. Llama 3's annealing improved the **8B model by +24% on GSM8K** and +6.4% on MATH, though the 405B model saw negligible gains—confirming that **annealing benefits smaller models more**, exactly the models most practitioners train.

### Low-resource language staging

Sailor 2's two-stage approach deserves special attention for the multi-language scenario. **Stage 1 (450B tokens, high LR)** trained on high-resource SEA languages only—Vietnamese 102B, Indonesian 94B, Thai 92B, English 51B, Chinese 50B, with smaller allocations to Burmese, Malay, Thai, Tagalog, Khmer. **Stage 2 (60B tokens, 1/10 LR)** introduced the lowest-resource languages (Cebuano, Lao, Javanese, Waray, Sundanese, Ilocano) plus 2.5B English instruction data. This staged introduction prevents low-resource languages from being drowned out during the high-LR phase and gives the model stable representations to transfer from before encountering extremely scarce data.

### LR schedule interacts with data ordering

A critical 2025 finding: standard aggressive cosine LR decay to near-zero **suppresses gradient signal from high-quality data** when it appears late in training. Two solutions: use moderate LR decay (final LR only ~10× below peak, not 100×), or apply Curriculum Model Averaging (CMA), which averages late-stage checkpoints. The combination yields **+1.6% average benchmark improvement** purely from data reordering. For multilingual CPT, this means: if using a cosine schedule decaying to very small values, front-load your highest-quality data or use a trapezoidal schedule that maintains a stable LR plateau before decaying.

---

## 6. Concrete recommendations for the 10–20 language scenario

### Recommended mixing formula

Start with UniMax ($N = 3$) for across-language balancing, allocating your target-language budget (~65% of total) across the 10–20 languages. If compute permits, refine with RegMix: train 512 proxy models at 1M parameters each on ~1B tokens per run (~2% of total compute overhead), fit Ridge regression, and select the optimal mixture. The RegMix step is particularly valuable when token counts vary by 50×, as it captures non-obvious cross-lingual transfer effects that heuristic formulas miss entirely.

**Concrete per-language allocation example** for 15 languages with 300B total target-language tokens and UniMax $N = 3$:

| Language data | Tokens available | UniMax allocation | Epochs seen |
|--------------|-----------------|-------------------|-------------|
| Language A | 50B | ~30B | 0.6 |
| Language B | 40B | ~28B | 0.7 |
| Languages C–F | 15–25B each | ~20B each | 0.8–1.3 |
| Languages G–K | 5–10B each | ~15B each | 1.5–3.0 |
| Languages L–O | 1–3B each | ~3–9B each | 3.0 (capped) |

### Total token budget

Based on successful projects, target **200–500B total tokens** for 7B–70B models. This represents roughly 1–5% of original pretraining budgets for modern base models. Swallow found performance monotonically improved up to 100B tokens for single-language CPT; Sailor 2 used 500B for 15 languages. A reasonable starting point for 10–20 languages is **300–400B tokens total**. Larger models can use proportionally less: Poro 2 used the same 165B budget for both the 8B and 70B models.

### Overall data composition

| Component | % of total | Purpose | Source examples |
|-----------|-----------|---------|----------------|
| New target languages | 60–70% | Language acquisition | FineWeb-2, HPLT, CulturaX, local corpora |
| English replay | 15–20% | Forgetting prevention | FineWeb-Edu, SlimPajama |
| Math replay | 3–5% | Preserve reasoning | FineMath 4+, Open-Web-Math-Pro |
| Code replay | 3–5% | Preserve coding | The Stack v2 (filtered) |
| Parallel data | 1–3% | Cross-lingual alignment | OPUS, NLLB-mined |

### Phase structure

**Phase 1 (85% of budget, ~255–340B tokens):** Full mixture with all languages present. Higher-resource languages at full volume; low-resource languages at up to 3× epoch cap. Learning rate: cosine decay from peak (e.g., 3e-4 for 8B, 1.5e-4 for 70B). Consider Sailor 2's approach of introducing the lowest-resource languages (<2B tokens available) only in Phase 2.

**Phase 2 (15% of budget, ~45–60B tokens):** Learning rate reduced to 1/10 of Phase 1 peak. Shift to top 10–20% quality data per language (selected by FastText or XLM-RoBERTa classifiers). Introduce lowest-resource languages here if following staged approach. Include 2–5B tokens of instruction-following data in English and target languages. Apply Polyak averaging over final checkpoints.

### Failure modes and how to detect them

**Math/code capability collapse** is the most common and insidious failure. Track GSM8K, MATH, and HumanEval scores at every checkpoint. If math performance drops more than 5 percentage points within any 20B-token window, increase math replay proportion immediately. Poro 2 showed this can happen rapidly and irreversibly for 8B models.

**Output language confusion** manifests as the model generating English when prompted in a target language, or mixing languages unpredictably. Monitor using language identification on model outputs during evaluation. This typically signals insufficient target language data relative to English replay.

**Vocabulary expansion disruption** (if applicable) can cause 10%+ QA degradation within 20B tokens if embedding initialization is poor. Sailor found this with Mistral, motivating their switch to Qwen which already had broad vocabulary coverage. Choose base models with good existing coverage of your target scripts.

**Over-filtering data destruction** particularly affects languages with <5B tokens. Monitor per-language validation loss. If any language shows stagnating or rising loss despite continued training, the effective data may have been over-filtered. Relax quality thresholds for that language.

**The strongest single predictor of CPT success**, per Poro 2's extensive experiments, is the **base model's existing English capability**. Stronger English models adapt better to new languages. This means investing in the highest-quality base model available—and protecting its English capabilities through adequate replay—pays compound dividends.

---

## Conclusion

The field has converged on a clear recipe for multilingual CPT as of early 2026. **UniMax with N = 3–5 is the recommended default** for cross-language balancing, superior to temperature sampling because it explicitly prevents the >100× repetition that degrades low-resource performance. **RegMix can refine this at ~2% compute overhead** and catches non-obvious cross-lingual transfer patterns. The replay ratio should be **20–30% total**, subdivided into English text, math, code, and parallel data—excluding any capability category causes measurable degradation, especially for 7B-scale models. **Data quality filtering must be calibrated per resource level**: aggressive for high-resource, minimal for languages under 5B tokens, following FineWeb-2's finding that over-filtering destroys scarce data. **Two-phase training with quality escalation** in the final 10–20% is now standard practice, with Phase 2 learning rates at 1/10 of Phase 1 and high-quality data upsampled. Perhaps the most actionable insight: train the smallest feasible proxy models first (even 1M parameters) to optimize your mixture before committing to the full run—this single practice, adopted by both Sailor 2 and informed by RegMix, can mean the difference between a successful multilingual model and hundreds of wasted GPU-hours.