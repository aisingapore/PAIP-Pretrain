# Applying Data-Mix Rules to a Large Multilingual Corpus

The previous two documents covered the *what* and the *where*: `0-datamix.md` surveyed the mixing strategies (temperature sampling, UniMax, replay ratios) and `1-dataloader_walkthrough.md` showed how Megatron's `BlendedDataset` consumes per-dataset weights at training time. This document covers the *how*: given a real corpus of 459 datasets across 14 languages, how do you programmatically compute the weight for every single dataset?

The answer is a 5-step rule cascade implemented in `calc_proportion_herorun.py`. It takes two small config dicts and one large YAML catalog as input, and outputs a flat list of `[weight, path, weight, path, ...]` that plugs directly into BlendedDataset.

---

## 1. Why manual weights don't scale

A typical multilingual pretraining corpus has:

- **14 languages** spanning high-resource (English, 1.1T tokens), mid-resource (Thai, ~96B tokens), and low-resource (Khmer, ~11B tokens)
- **2 quality tiers** per language: CC (web-crawled, larger) and nonCC (curated sources like Wikipedia, smaller)
- **Multiple datasets per tier**: English nonCC alone has 36 sources ranging from 28B tokens (Cosmopedia) to 624K tokens (GlobalVoice) -- a 44,000x range
- **Quality-tagged variants**: some datasets are split into head/middle/tail buckets by edu-score, tripling the entry count

Assigning 459 weights by hand is error-prone and non-reproducible. Worse, the weights must satisfy **five competing objectives simultaneously**:

1. Mitigate catastrophic forgetting by allocating sufficient replay data
2. Balance mid/low-resource languages so they aren't drowned out by English
3. Balance different sources/domains within each language
4. Upsample higher-quality data
5. Avoid over-repeating small datasets

A rule cascade solves this: each step narrows the allocation from coarse (language level) to fine (individual dataset level), and each step is responsible for one or two objectives.

---

## 2. Input: the hierarchical data catalog

All datasets are organized in a YAML file with a strict 4-level hierarchy:

```
Language
  Quality (CC / nonCC)
    Domain (currently always "mixed")
      Dataset[]
```

Here is a trimmed excerpt showing the real structure:

```yaml
CODE:
  CC:
    mixed:
    - name: CODE_OLMo2-StarCoder
      megatron_token_count: 98980804567

EN:
  CC:
    mixed:
    - name: EN_Fineweb-Edu
      megatron_token_count: 190650069973
    - name: EN_DCLM-OLMo2-HQ
      megatron_token_count: 765854035287
  nonCC:
    mixed:
    - name: EN_Cosmopedia
      megatron_token_count: 27575408324
    - name: EN_Wikipedia
      megatron_token_count: 6592965498
    # ... 34 more nonCC sources

KM:
  CC:
    mixed:
    - name: KM_SLPv1_GGI-synthetic
      megatron_token_count: 2389764065
    - name: KM_SLPv1_GGI-head-25        # <-- quality-tagged triplet
      megatron_token_count: 369699213
    - name: KM_SLPv1_GGI-middle-50
      megatron_token_count: 786789066
    - name: KM_SLPv1_GGI-tail-25
      megatron_token_count: 287443056
    # ... more CC sources
  nonCC:
    mixed:
    - name: KM_Wikipedia_GGI-synthetic
      megatron_token_count: 19387227
    - name: KM_Opendevelopment           # <-- no quality tag
      megatron_token_count: 105935053
    # ... more nonCC sources
```

Each dataset entry has a `name` and a `megatron_token_count` (the exact number of tokens in the merged `.bin/.idx` file). The only field the proportion calculator uses is the token count.

### Quality-tagged datasets

Some datasets have been filtered by an LLM-based education quality scorer (e.g., Kimi-K2-Instruct) into three buckets:

| Suffix | Meaning | Typical share |
|--------|---------|---------------|
| `-head-25` | Top 25% by edu-score (highest quality) | Smallest token count |
| `-middle-50` | Middle 50% | Largest token count |
| `-tail-25` | Bottom 25% (lowest quality) | Medium token count |

These triplets always share a common prefix (e.g., `KM_SLPv1_GGI`). The proportion calculator treats them as a single logical dataset during budget allocation, then re-expands and reweights them to upsample the high-quality head bucket.

---

## 3. Input: the two control dicts

The entire mixing strategy is controlled by just two dictionaries. Everything else is derived from the data catalog.

### LANG_RATIO -- top-level token budget per language

```python
LANG_RATIO = {
    "CODE": 0.1,
    "EN":   0.4,
    "ZH":   0.09,
    "VI":   0.085,
    "ID":   0.085,
    "TH":   0.085,
    "TL":   0.025,
    "TA":   0.045,
    "MS":   0.0475,
    "KM":   0.015,
    "LO":   0.005,
    "MY":   0.0175,
    "JV":   0,
    "SU":   0,
}
```

The values need not sum to 1.0 -- they are auto-normalized in Step 1. The design reflects three tiers:

| Tier | Languages | Allocation range | Rationale |
|------|-----------|-----------------|-----------|
| **High-resource** | EN | 0.4 (31.5% after normalization) | Largest allocation: serves as both new training data AND replay data for catastrophic forgetting mitigation. The 0-datamix.md recommendation of 20-30% English replay is embedded here. |
| **Mid-resource** | ZH, VI, ID, TH | 0.085-0.09 each (6.7-7.1%) | These languages have 50B-100B+ tokens available. Roughly equal allocation balances them against each other. |
| **Low-resource** | TL, TA, MS, KM, LO, MY | 0.005-0.0475 (0.4-3.7%) | Proportioned to actual data availability and target-language importance. Smaller allocations prevent excessive repetition of scarce data. |
| **Deprioritized** | JV, SU | 0 | Explicitly excluded -- insufficient quality data. Setting to 0 is better than including data that would be repeated 50+ times. |

**Connection to the replay objective**: CODE=0.1 ensures code data is always present in the mix (preventing code capability regression). EN=0.4 is deliberately large -- it bundles English text replay, math data, and new English training data into one allocation. This is the "20-30% replay" from `0-datamix.md` Section 3, implemented as a single language-level knob.

### CC_RATIO -- quality tier split per language

```python
CC_RATIO = {
    "EN": 0.8,
    "ZH": 0.9,
    "VI": 0.6,
    "ID": 0.8,
    "TH": 0.5,
}
```

This dict is only defined for 5 languages where the practitioner wants to **override** the natural CC/nonCC balance. For example, English has ~84% CC tokens naturally (957B CC vs 188B nonCC), but `CC_RATIO["EN"] = 0.8` forces the split to 80/20 -- slightly downweighting CC to give nonCC curated sources more representation.

**Languages not in CC_RATIO** (all low-resource languages, plus CODE) fall back to their **natural token ratio**: if KM has 10.9B CC tokens and 223M nonCC tokens, the natural CC ratio is 10.9/(10.9+0.22) = 0.98, so ~98% of KM's budget goes to CC.

---

## 4. The 5-step rule cascade

The core of the script. Each step narrows the granularity of the allocation.

```
LANG_RATIO + CC_RATIO + data_config.yaml
        │
        ▼
┌─────────────────────────────────────┐
│ Step 0: Source completeness check   │  Fail-fast if any .idx missing
├─────────────────────────────────────┤
│ Step 1: Normalize language ratios   │  14 languages → weights sum to 1.0
├─────────────────────────────────────┤
│ Step 2: Expand to lang-quality      │  14 → 28 entries (lang_CC + lang_nonCC)
├─────────────────────────────────────┤
│ Step 3: Distribute to datasets      │  28 → 459 per-dataset weights
├─────────────────────────────────────┤
│ Step 4: Bucket weight adjustment    │  Reweight head/middle/tail triplets
├─────────────────────────────────────┤
│ Step 5: Validate and output         │  Filter zeros, assert sum ≈ 1.0
└─────────────────────────────────────┘
        │
        ▼
[weight, path, weight, path, ...]  → BlendedDataset
```

### Step 0: Source completeness check

Before any math, the script walks the entire YAML catalog and verifies that every dataset has its `.idx` file on disk:

```python
index_file_path = os.path.join(data_root_dir, source, f'{source}.idx')
if not os.path.exists(index_file_path):
    missing_sources.append(source)
```

If any are missing, it raises a `ValueError` with the full list. This fail-fast check prevents silent errors where a missing dataset would effectively redistribute its budget to other datasets.

### Step 1: Normalize language ratios

**Formula:**

$$\text{lang\_proportions}[k] = \frac{\text{LANG\_RATIO}[k]}{\sum_i \text{LANG\_RATIO}[i]}$$

**Worked example:**

```
sum(LANG_RATIO) = 0.1 + 0.4 + 0.09 + 0.085 + 0.085 + 0.085
               + 0.025 + 0.045 + 0.0475 + 0.015 + 0.005
               + 0.0175 + 0 + 0
               = 1.27

EN  = 0.4   / 1.27 = 0.31496  (31.5%)
TH  = 0.085 / 1.27 = 0.06693  (6.7%)
KM  = 0.015 / 1.27 = 0.01181  (1.2%)
JV  = 0     / 1.27 = 0        (excluded)
```

**Objective served**: *language balancing*. The explicit floor allocations for low-resource languages guarantee they receive a minimum share of the training budget, regardless of their token count. Without this, proportional sampling would give KM (11B tokens) only 0.8% of the budget vs EN's (1.1T tokens) 85%.

### Step 2: Expand to language-quality pairs

Each language proportion is split into CC and nonCC sub-allocations.

**For languages with an explicit CC_RATIO:**

$$\text{lang\_CC} = \text{lang\_proportions}[\text{lang}] \times \text{CC\_RATIO}[\text{lang}]$$
$$\text{lang\_nonCC} = \text{lang\_proportions}[\text{lang}] \times (1 - \text{CC\_RATIO}[\text{lang}])$$

**Worked example (EN):**
```
EN_CC    = 0.31496 x 0.8 = 0.25197
EN_nonCC = 0.31496 x 0.2 = 0.06299
```

**For languages without an explicit CC_RATIO** (falls back to natural ratio):

```python
natural_cc_ratio = cc_token_count / (cc_token_count + noncc_token_count)
```

**Worked example (KM):**
```
KM CC tokens:    10,867,072,201
KM nonCC tokens:    222,649,768
natural_cc_ratio = 10,867,072,201 / 11,089,722,005 = 0.9799

KM_CC    = 0.01181 x 0.9799 = 0.01157
KM_nonCC = 0.01181 x 0.0201 = 0.00024
```

**Objective served**: *quality upsampling*. By setting `CC_RATIO["TH"] = 0.5` when TH naturally has ~99.7% CC data, you deliberately over-represent the small but curated nonCC sources (Wikipedia, news, etc.). Without this override, TH_nonCC would get only 0.3% of TH's budget.

### Step 3: Distribute to individual datasets

This is where 28 language-quality allocations become 459 per-dataset weights. The splitting strategy differs between CC and nonCC:

**CC datasets: split equally**

```python
source_proportions[source] = proportion / source_count
```

Every CC dataset in a language gets the same weight, regardless of token count.

*Worked example (EN_CC, 2 datasets):*
```
EN_Fineweb-Edu:     0.25197 / 2 = 0.12599
EN_DCLM-OLMo2-HQ:  0.25197 / 2 = 0.12599
```

Even though DCLM has 4x more tokens than Fineweb-Edu (766B vs 191B), they get equal sampling weight. This means Fineweb-Edu will be repeated ~4x more often -- a deliberate upsampling of the (presumably higher-quality) Fineweb-Edu dataset.

**Rationale**: CC data sources are pre-filtered web crawls that have already passed quality thresholds. Equal splitting prevents a single massive crawl from dominating the CC allocation and ensures diversity of web sources.

**nonCC datasets: split by token count**

```python
relative_prop = source["megatron_token_count"] / lang_quality_total_tokens
source_proportions[source] = lang_quality_proportion * relative_prop
```

Each nonCC dataset gets weight proportional to its size.

*Worked example (KM_nonCC, simplified to 2 key sources):*
```
KM_nonCC total tokens: 222,649,768
KM_nonCC proportion:   0.00024

KM_Opendevelopment (105.9M tokens):
  relative = 105,935,053 / 222,649,768 = 0.4758
  weight   = 0.00024 x 0.4758 = 0.0001142

KM_Wikipedia_GGI-synthetic (19.4M tokens):
  relative = 19,387,227 / 222,649,768 = 0.0871
  weight   = 0.00024 x 0.0871 = 0.0000209
```

**Rationale**: nonCC sources are heterogeneous (Wikipedia, news, books, government data). Token-proportional splitting means a small dataset like KM_GlobalVoice (81K tokens) gets a tiny weight and won't be over-repeated, while larger sources like KM_Opendevelopment (106M tokens) get proportionally more.

**Objective served**: *source balancing* (equal CC split ensures web-source diversity) and *avoiding over-sampling small datasets* (token-proportional nonCC split keeps small datasets under 1 epoch).

**Tagged dataset collapsing**: Before distributing, head/middle/tail triplets are collapsed into a single entry whose token count is the sum of all three. For example:

```
KM_SLPv1_GGI-head-25:   369,699,213
KM_SLPv1_GGI-middle-50: 786,789,066
KM_SLPv1_GGI-tail-25:   287,443,056
                         ─────────────
Collapsed entry:       1,443,931,335  (as "KM_SLPv1_GGI")
```

This collapsed entry participates in the CC equal-split alongside other CC sources. It gets re-expanded and reweighted in Step 4.

### Step 4: Bucket weight adjustment

For each tagged dataset (head/middle/tail triplet), the script redistributes weight from the low-quality tail to the high-quality head. The middle bucket is left unchanged.

**Delta calculation:**

```python
if S_T > S_H:
    delta = min((S_T / S_H) - 1, 1)    # capped at 1.0
else:
    delta = 0.5 * S_T / S_H
```

The delta measures how much "excess" tail data exists relative to head data. When tail is larger than head (the common case after quality filtering), delta represents the degree of quality imbalance.

**Reweighting:**

```python
head_new   = head_ratio   + head_ratio * delta     # upsample high-quality
middle_new = middle_ratio                           # unchanged
tail_new   = tail_ratio   - head_ratio * delta      # downsample low-quality
```

This is a **budget-neutral reallocation**: the total `head_new + middle_new + tail_new == original_total` is preserved exactly. Weight moves from tail to head; the total pie doesn't change.

**Worked example (TH_SLP_1_GGI):**

```
S_H (head-25):   2,057,048,857 tokens
S_M (middle-50): 4,585,093,464 tokens
S_T (tail-25):   1,775,548,351 tokens
Total:           8,417,690,672 tokens

S_T < S_H, so:
  delta = 0.5 x S_T / S_H = 0.5 x 1,775,548,351 / 2,057,048,857 = 0.4316

Suppose source_ratio_for_prefix = 0.00335 (from Step 3 equal split):
  SH_ratio = (2,057M / 8,418M) x 0.00335 = 0.000819
  SM_ratio = (4,585M / 8,418M) x 0.00335 = 0.001825
  ST_ratio = (1,776M / 8,418M) x 0.00335 = 0.000707

  SH_new = 0.000819 + 0.000819 x 0.4316 = 0.001172  (+43% boost)
  SM_new = 0.001825                       = 0.001825  (unchanged)
  ST_new = 0.000707 - 0.000819 x 0.4316  = 0.000354  (-50% reduction)

  Check: 0.001172 + 0.001825 + 0.000354 = 0.003351 ≈ 0.00335 ✓
```

**Worked example (TH_Fineweb2_GGI) -- more extreme imbalance:**

```
S_H (head-25):    6,056,890,592
S_M (middle-50): 20,988,635,635
S_T (tail-25):    9,749,002,919

S_T > S_H, so:
  delta = min((9,749M / 6,057M) - 1, 1) = min(0.6096, 1) = 0.6096

  → Head gets +61% boost, tail absorbs the reduction
```

**Objective served**: *quality upsampling within a single source*. This is the fine-grained quality knob -- the CC_RATIO in Step 2 controls quality at the language level, while bucket adjustment controls quality at the individual-dataset level.

### Step 5: Validate and output

The final step filters out zero-weight datasets and validates the invariant:

```python
non_zero_proportions = {k: v for k, v in adjusted.items() if v > 0}
total = sum(non_zero_proportions.values())
assert math.isclose(total, 1.0, rel_tol=1e-8)
```

The output format is a flat interleaved list:

```python
blend_list = [weight1, "path/to/source1/source1",
              weight2, "path/to/source2/source2", ...]
```

This is the format that Megatron-Bridge's BlendedDataset expects. The connection to `1-dataloader_walkthrough.md` is direct: these weights become the `dataset_weights` that BlendedDataset uses to build its `dataset_index` array, determining which source dataset each training sample is drawn from.

---

## 5. How the five objectives are met

| Objective | Mechanism | Step | How it works |
|-----------|-----------|------|-------------|
| **Catastrophic forgetting mitigation** | EN=0.4, CODE=0.1 in LANG_RATIO | Step 1 | English allocation bundles replay text + math + code data. CODE gets its own 10% allocation. Together they provide the 20-30% replay recommended in `0-datamix.md` Section 3. |
| **Mid/low-resource language balancing** | Explicit floor ratios in LANG_RATIO; 3-tier design (HR/MR/LR) | Step 1 | Every target language has a guaranteed minimum budget. KM gets 1.2% instead of the 0.8% it would get from proportional sampling. LR languages are over-represented relative to their data size. |
| **Source/domain balancing within a language** | CC: equal split. nonCC: token-proportional. | Step 3 | Equal CC split prevents a single 766B-token web crawl from overshadowing a 191B-token curated dataset. Token-proportional nonCC split ensures diverse small sources participate without excessive repetition. |
| **Quality upsampling** | CC_RATIO overweights CC tier; bucket adjustment shifts weight from tail to head | Steps 2, 4 | CC_RATIO can force more budget to the CC tier than its natural share. Bucket adjustment then further upsamples the top-25% edu-score data within individual datasets. Two levels of quality knobs. |
| **Avoid over-sampling small datasets** | Token-proportional nonCC split; JV/SU set to 0 | Steps 1, 3 | A 81K-token dataset gets a proportionally tiny weight (< 0.000001), meaning it will be seen well under 1 epoch. Languages with insufficient data are excluded entirely via zero allocation. |

---

## 6. What this script does NOT handle

### Epoch cap enforcement

The `0-datamix.md` Section 1 recommends a 4-epoch repetition maximum. This script does not enforce it -- it only computes sampling *weights*, not actual sample counts. Epoch enforcement happens downstream: Megatron's `GPTDataset` (Layer 2 in `1-dataloader_walkthrough.md`) naturally cycles through available documents. If a dataset's weight implies more samples than it has documents, GPTDataset wraps around and re-samples from the beginning.

The nonCC token-proportional split in Step 3 provides soft protection: a small dataset gets a small weight, making it unlikely to exceed a few epochs. But there is no hard cap in this script.

### Phase-based training

Production training often uses two phases: Phase 1 with the standard mix for ~90% of tokens, then Phase 2 with elevated quality weights for the final ~10% (see `0-datamix.md` Section 6). This script computes a **single static mix**. To implement two-phase training, you run it twice with different LANG_RATIO and CC_RATIO dicts for each phase.

### Dynamic loss-based reweighting

Methods like DoReMi and RegMix (`0-datamix.md` Section 2) learn mixing weights by training small proxy models and optimizing based on validation loss. This script uses **fixed heuristic rules** instead. The heuristic approach is faster (no proxy training needed) and more interpretable (every weight is traceable to a rule), but cannot discover non-obvious optima like "upsample Khmer and downsample Vietnamese" that RegMix found for Sailor 2.

---

## Reference

**Script**: `megatron_bridge/scripts/datamix/calc_proportion_herorun.py`

```bash
python calc_proportion_herorun.py \
    --data_config ../../configs/data_config.yaml \
    --data_root_dir /mnt/weka/aisg/data/megatron/gemma3 \
    --debug
```

**Key function**: `calc_proportion()` (line 717) orchestrates all 5 steps and returns the `blend_list`.
