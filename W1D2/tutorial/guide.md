# Tutorial: Merge Verification

## Overview

When preprocessing data for LLM pretraining, raw text is first tokenized into per-shard binary files, then merged into a single dataset. A silent corruption during the merge — dropped documents, truncated token sequences — could contaminate an entire training run. **Merge verification** catches these problems early by checking invariants that must hold between the shards and the merged output.

In this exercise you will implement two verification checks in `verify_merge.py`.

## Background

Megatron stores preprocessed data as an **IndexedDataset** — a pair of files:

| File  | Contents |
|-------|----------|
| `.idx` | An index of byte offsets and per-document token counts |
| `.bin` | A flat array of token IDs |

You can load one with:

```python
from megatron.core.datasets.indexed_dataset import IndexedDataset

ds = IndexedDataset("/path/to/prefix")   # loads prefix.bin + prefix.idx
ds.sequence_lengths          # numpy array — token count of each document
ds[i]                        # numpy array — token IDs of document i
len(ds)                      # number of documents
```

The helper `get_stats(prefix)` (already provided) returns `(num_docs, num_tokens)` by reading `sequence_lengths`.

## Exercises

Open `verify_merge.py` and complete the two TODO sections.

### Exercise 1 — Count Tally

**What**: Verify that the total documents and tokens across all shards equal the merged dataset's totals.

**Why**: If the merge dropped or duplicated a shard, the counts will disagree.

**Approach**:
1. Loop over each shard prefix. Call `get_stats()` to get its doc/token counts. Accumulate the sums.
2. Call `get_stats()` on the merged prefix.
3. Compare the two totals. Print PASS or FAIL.

Also build a `prefix_doc_counts` list of `(prefix, num_docs)` tuples — you will need it in Exercise 2.

### Exercise 2 — Random Index Spot-Check

**What**: For randomly chosen document indices, verify that the token-sequence length in the merged dataset matches the length in the originating shard.

**Why**: Even if the aggregate counts match, individual documents could be misaligned (e.g. if shards were concatenated in the wrong order). A spot-check catches ordering bugs.

**Approach**:
1. Build a **cumulative document-count table** from `prefix_doc_counts`. This lets you map any global document index to the correct shard and its local index within that shard.

   Example: if shard A has 100 docs and shard B has 200 docs:
   - Global indices 0–99 → shard A, local index = global index
   - Global indices 100–299 → shard B, local index = global index − 100

2. For each of `num_samples` iterations, pick a random global index, look up the shard, and compare `len(merged_ds[global_idx])` against `len(part_ds[local_idx])`.

## Running

After you have completed the pipeline at least once (so that `converted/` and `merged/` directories exist), run:

```bash
python W1D2/tutorial/verify_merge.py \
    --output-dir /path/to/output \
    --dataset-name your_dataset_name
```

## Checking Your Work

Compare your output against the reference implementation:

```bash
python W1D2/verify_data.py \
    --mode merge \
    --output-dir /path/to/output \
    --dataset-name your_dataset_name
```

Both should report the same PASS/FAIL results and matching document/token counts. The reference answer is in `W1D2/verify_data.py`, function `verify_merge()` (line 145).
