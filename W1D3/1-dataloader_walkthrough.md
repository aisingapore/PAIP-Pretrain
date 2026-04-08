# Megatron Dataloader: A Deep Walkthrough

This document traces the full lifecycle of a training sample in Megatron-LM — from raw `.bin`/`.idx` files on disk through the blended dataset hierarchy to the GPU. For each topic, the relevant MegatronBridge configuration keys are shown.

---

## Architecture Overview

The dataloader is organized into three nested layers, each solving a different problem:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  torch.utils.data.DataLoader                                            │
│  (num_workers, pin_memory, persistent_workers)                          │
│                              ▲                                          │
│              batch_sampler   │  yields list[int] indices                │
│  ┌────────────────────────────────────────────┐                         │
│  │  MegatronPretrainingSampler                │                         │
│  │  Iterates consumed_samples → total_samples │                         │
│  │  Each DP rank takes its slice of the batch │                         │
│  └────────────────────────────────────────────┘                         │
│                              ▲                                          │
│             dataset          │  __getitem__(idx)                        │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  LAYER 3 — BlendedDataset                                        │   │
│  │  dataset_index[idx]        → which source dataset                │   │
│  │  dataset_sample_index[idx] → which sample within that dataset    │   │
│  │                                                                  │   │
│  │  ┌─────────────────────┐  ┌─────────────────────┐                │   │
│  │  │ GPTDataset (wiki)   │  │ GPTDataset (code)   │  ...           │   │
│  │  │  LAYER 2            │  │  LAYER 2            │                │   │
│  │  │  document_index     │  │  document_index     │                │   │
│  │  │  sample_index       │  │  sample_index       │                │   │
│  │  │  shuffle_index      │  │  shuffle_index      │                │   │
│  │  │        ▼            │  │        ▼            │                │   │
│  │  │  IndexedDataset     │  │  IndexedDataset     │                │   │
│  │  │  LAYER 1            │  │  LAYER 1            │                │   │
│  │  │  wiki.bin / .idx    │  │  code.bin / .idx    │                │   │
│  │  └─────────────────────┘  └─────────────────────┘                │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Reading direction**: when the training loop calls `next(data_iterator)`, it triggers the sampler → BlendedDataset → GPTDataset → IndexedDataset chain, retrieving tokens from disk (or S3) via OS-managed memory mapping.

---

## Layer 1: IndexedDataset — Storage & Random Access

**Source**: `megatron/core/datasets/indexed_dataset.py`

### The .bin / .idx File Pair

Every preprocessed dataset lives as two files:

```
my_dataset.bin   ← raw token IDs packed in binary (int32 or uint16)
my_dataset.idx   ← index: sequence lengths, byte offsets, document boundaries
```

The `.idx` file has a fixed binary layout:

```
[Magic: 9 bytes "MMIDIDX\x00\x00"] [Version: uint64=1] [DType code: uint8]
[Sequence count: uint64] [Document count: uint64]
[sequence_lengths: int32 × N]    ← length (tokens) of each sequence
[sequence_pointers: int64 × N]   ← byte offset into .bin for each sequence
[document_indices: int64 × D]    ← sequence indices marking doc boundaries
```

### Memory-Mapped Access: How It Avoids Loading Everything into RAM

```python
# _IndexReader.__init__ (indexed_dataset.py:246)
self.bin_buffer_mmap = numpy.memmap(idx_path, mode="r", order="C")
self.bin_buffer = memoryview(self.bin_buffer_mmap)
# Arrays are zero-copy views — no data is copied to RAM
self.sequence_lengths = numpy.frombuffer(self.bin_buffer, dtype=numpy.int32, ...)
self.sequence_pointers = numpy.frombuffer(self.bin_buffer, dtype=numpy.int64, ...)
```

```python
# _MMapBinReader.read (indexed_dataset.py:419)
self._bin_buffer_mmap = numpy.memmap(bin_path, mode="r", order="C")
# Zero-copy read of exactly `count` tokens at byte `offset`:
return numpy.frombuffer(self._bin_buffer, dtype=dtype, count=count, offset=offset)
```

**Why this works without loading the whole dataset**: `numpy.memmap` asks the OS to map the file into the process's virtual address space. Physical RAM pages are only loaded on demand, when specific byte ranges are actually accessed. For a 1 TB dataset you only pay RAM for the pages you've touched recently — the OS evicts old pages automatically. This is the same mechanism used by databases and operating system page caches.

### `IndexedDataset.get()` — The Core Access Method

```python
# indexed_dataset.py:843
def get(self, idx, offset=0, length=None):
    sequence_pointer, sequence_length, _ = self.index[idx]
    if length is None:
        length = sequence_length - offset
    sequence_pointer += offset * DType.size(self.index.dtype)
    return self.bin_reader.read(dtype=self.index.dtype,
                                count=length, offset=sequence_pointer)
```

This allows fetching an arbitrary **sub-slice** of any document — important for samples that span document boundaries.

### Pickling for DataLoader Workers

`__getstate__` / `__setstate__` serialize only the file path and config, not the mmap. Each DataLoader worker process independently re-opens the mmap on startup. This means `N` workers = `N` independent mmaps of the same file — all reading from OS page cache without duplicating data in RAM.

> **MegatronBridge Config**
> | Key | Default | Controls |
> |-----|---------|----------|
> | `dataset.mmap_bin_files` | `True` | Use memory-mapped .bin files. Set `False` for non-mmap file reads (with retry logic). |

---

## Layer 2: GPTDataset — Sequencing & Shuffling

**Source**: `megatron/core/datasets/gpt_dataset.py`

`GPTDataset` wraps one `IndexedDataset` (one `.bin`/`.idx` pair) and builds three index arrays that answer: *in what order, and at what positions, do we read token sequences from this dataset?*

### The Three-Index System

```
document_index   : int32[num_epochs × num_docs]     — shuffled doc IDs to walk
sample_index     : int32[num_samples+1, 2]           — (doc_idx_index, offset) boundaries
shuffle_index    : uint32[num_samples]               — random permutation of sample positions
```

These are built once at dataset construction time and cached as `.npy` files. Every subsequent run loads them via `mmap_mode="r"` — again, no RAM copy.

### Index 1: Document Index — Epoch Tiling & Per-Epoch Shuffling

```python
# gpt_dataset.py:643
def _build_document_index(documents, num_epochs, numpy_random_state, separate_final_epoch):
    if not separate_final_epoch or num_epochs == 1:
        document_index = numpy.mgrid[0:num_epochs, 0:len(documents)][1]
        document_index[:] = documents         # tile: [docs, docs, docs, ...]
        document_index = document_index.reshape(-1)
        document_index = document_index.astype(numpy.int32)
        numpy_random_state.shuffle(document_index)  # shuffle the entire tiled array
        return document_index
    # If separate_final_epoch: shuffle all-but-last epochs together,
    # then shuffle the final epoch independently and concatenate.
    doc_idx_first = _build_document_index(documents, num_epochs-1, numpy_random_state, False)
    doc_idx_last  = _build_document_index(documents, 1, numpy_random_state, False)
    return numpy.concatenate((doc_idx_first, doc_idx_last))
```

The result is a flat array of document IDs repeated `num_epochs` times, shuffled so documents from different epochs are interleaved. The `separate_final_epoch` logic (threshold = 80%) isolates the final partial epoch from the main shuffle to avoid overrepresenting it.

### How Epoch Cycling Works (Repeating a Small Dataset)

When the requested number of training samples exceeds what one pass over the dataset provides, `_get_num_epochs` keeps adding passes until there are enough tokens:

```python
# gpt_dataset.py:620
def _get_num_epochs(self, num_tokens_per_epoch):
    num_epochs = 1
    num_tokens = num_tokens_per_epoch
    if self.num_samples is None:
        return num_epochs
    num_tokens_requested = (
        self.num_samples * self.config.sequence_length
        + self.config.add_extra_token_to_sequence
    )
    while num_tokens < num_tokens_requested:
        num_epochs += 1
        num_tokens += num_tokens_per_epoch
    return num_epochs
```

Each epoch cycle gets its own independent shuffle. A dataset small enough to be traversed 5× will have 5 independently shuffled passes concatenated together. The sampler sees this as a seamless stream of samples — it never "knows" epoch boundaries.

### Index 2: Sample Index — How Sequence Length Is Applied

The sample index is built by C++ function `build_sample_idx` (`helpers.cpp:145`). It treats the document_index as a recipe for walking the corpus and carves out fixed-length token sequences:

```cpp
// helpers.cpp — simplified
while (sample_idx_index <= num_samples) {
    int32_t remaining = seq_length + add_extra_token;  // e.g., 2049 for seq_len=2048
    while (remaining != 0) {
        document_length = sizes[document_idx[doc_idx_index]] - doc_offset;
        remaining -= document_length;
        if (remaining <= 0) {
            // This document had enough tokens; record end position
            doc_offset += (remaining + document_length - add_extra_token);
            remaining = 0;
        } else {
            // Document exhausted before sequence filled — advance to next doc
            ++doc_idx_index;
            doc_offset = 0;
        }
    }
    // Record start of next sample
    sample_idx[sample_idx_index] = {doc_idx_index, doc_offset};
    ++sample_idx_index;
}
```

The sample_index is a 2-D array of shape `[num_samples + 1, 2]`. Entry `i` stores `(doc_idx_index, offset)` — the *start* of sample `i`. Entry `i+1` stores the *end*. The final token retrieval in `_query_document_sample_shuffle_indices` then calls `IndexedDataset.get()` for each document span and concatenates the pieces.

**How the same data supports different sequence lengths**: the sample_index is rebuilt for each `sequence_length` value. The underlying `.bin` file never changes — only how we slice the flat token stream changes. Change `seq_length` from 2048 to 4096 and you get half as many, twice-as-long samples from the identical documents.

**Cross-document samples**: when a sequence needs more tokens than remain in a document, it spans into the next document. The EOD (end-of-document) token is included, providing the model with document boundary signals. The `reset_position_ids` / `reset_attention_mask` flags control whether attention is masked at these boundaries.

### Index 3: Shuffle Index — Sample Order Randomization

```python
# gpt_dataset.py:677
def _build_shuffle_index(num_samples, total_size, numpy_random_state):
    shuffle_idx = numpy.arange(start=0, stop=num_samples, dtype=numpy.uint32)
    numpy_random_state.shuffle(shuffle_idx)
    if num_samples == total_size:
        return shuffle_idx
    # If separate_final_epoch: append a separately-shuffled range for the last epoch
    shuffle_idx_last = numpy.arange(start=num_samples, stop=total_size, dtype=numpy.uint32)
    numpy_random_state.shuffle(shuffle_idx_last)
    return numpy.concatenate((shuffle_idx, shuffle_idx_last))
```

At access time:
```python
# __getitem__ step 1: apply shuffle
idx = self.shuffle_index[idx]
# step 2: look up sample boundaries
doc_index_beg, offset_beg = self.sample_index[idx]
doc_index_end, offset_end = self.sample_index[idx + 1]
```

**Important**: the `MegatronPretrainingSampler` iterates sample indices sequentially (0, 1, 2, ...). The shuffling lives inside the dataset, not the sampler. This means the data ordering is entirely determined at build time and is reproducible from the seed alone.

### Caching the Index Files

All three indices are saved to disk as `.npy` files under `path_to_cache`. The filename is keyed by an MD5 hash of a unique description string that encodes: dataset path, split, `random_seed`, `sequence_length`, tokenizer, and split ratios. Any change to any of these inputs produces a different hash, triggering a rebuild.

On subsequent runs, they're loaded lazily:
```python
document_index = numpy.load(path_to_document_index, mmap_mode="r")
sample_index   = numpy.load(path_to_sample_index,   mmap_mode="r")
shuffle_index  = numpy.load(path_to_shuffle_index,  mmap_mode="r")
```

`defer_npy_index_mmap=True` delays even opening these files until the first `__getitem__` call, useful for large multi-node jobs where rank 0 builds the indices while other ranks wait.

> **MegatronBridge Config**
> | Key | Default | Controls |
> |-----|---------|----------|
> | `dataset.seq_length` / `dataset.sequence_length` | required | Sequence length; determines how the flat token stream is sliced |
> | `dataset.split` | required | Train/valid/test ratio string e.g. `"9999,8,2"` |
> | `dataset.random_seed` | `1234` | Seed for document shuffle, sample index, shuffle index |
> | `dataset.path_to_cache` | next to `.bin` file | Directory for cached `.npy` index files |
> | `dataset.defer_npy_index_mmap` | `False` | Delay mmap of index files until first access |
> | `dataset.reset_position_ids` | required | Reset position IDs at document boundaries |
> | `dataset.reset_attention_mask` | required | Mask attention across document boundaries |
> | `dataset.eod_mask_loss` | required | Mask loss on EOD tokens |
> | `dataset.add_extra_token_to_sequence` | `True` | Draw `seq_len+1` tokens so `tokens` and `labels` both have `seq_len` length |

---

## Layer 3: BlendedDataset — Dataset Mixing

**Source**: `megatron/core/datasets/blended_dataset.py`

`BlendedDataset` takes a list of `GPTDataset` instances and their weights and produces a single dataset that, when iterated sequentially, interleaves samples according to the target ratios.

### The Two Blend Arrays

```python
# blended_dataset.py — key data structures
self.dataset_index        # int16[total_samples] — which dataset to pull from
self.dataset_sample_index # int64[total_samples] — which sample within that dataset
```

For sample position `i`:
```python
# __getitem__ (blended_dataset.py:97)
dataset_id        = self.dataset_index[idx]
dataset_sample_id = self.dataset_sample_index[idx]
return {"dataset_id": dataset_id, **self.datasets[dataset_id][dataset_sample_id]}
```

### The Greedy Error-Minimization Blending Algorithm

The blend arrays are built by C++ `build_blending_indices` (`helpers.cpp:77`). The algorithm is a greedy scheduler that maintains running counts of samples drawn from each dataset and always picks the dataset that is most "underrepresented" relative to its target weight:

```cpp
// helpers.cpp:77 — build_blending_indices
int64_t current_samples[num_datasets] = {0};

for (int64_t sample_idx = 0; sample_idx < size; ++sample_idx) {
    auto sample_idx_double = std::max((double)sample_idx, 1.0);

    // Find which dataset has the largest gap between target and actual
    int64_t max_error_index = 0;
    double max_error = weights[0] * sample_idx_double - current_samples[0];
    for (int64_t d = 1; d < num_datasets; ++d) {
        double error = weights[d] * sample_idx_double - current_samples[d];
        if (error > max_error) { max_error = error; max_error_index = d; }
    }

    // Assign this slot to the most-underrepresented dataset
    dataset_index[sample_idx]        = max_error_index;
    dataset_sample_index[sample_idx] = current_samples[max_error_index];
    current_samples[max_error_index] += 1;
}
```

**Key property**: at every position `i` in the sequence, the running proportions of samples drawn from each dataset are as close as possible to the target weights. The mixing is globally correct across the entire training run, not just within individual batches.

**Practical consequence**: within one Global Batch you might see a slight imbalance (e.g., 7 wiki samples and 3 code samples instead of exactly 8/2 for a 0.8/0.2 split). But across many batches, the ratio converges tightly to the specification. There is no guarantee of exact per-batch proportions.

### The `mid_level_dataset_surplus` Buffer

Before blending, `BlendedMegatronDatasetBuilder` computes how many samples each `GPTDataset` must provide:

```python
# blended_megatron_dataset_builder.py:553
sizes_per_dataset = [
    [int(math.ceil(math.ceil(target_size * weight) * (1 + surplus)))
     for target_size in target_size_per_split]
    for weight in normalized_weights
]
```

The `surplus` (default 0.005 = 0.5%) ensures each mid-level dataset is built slightly larger than the blend requires. This guards against off-by-one edge cases in the C++ algorithm. If the blend asks for more samples than a dataset has, you'll get an `IndexError` at `BlendedDataset` init time — increase `mid_level_dataset_surplus` to fix it.

### Uniform Sampling Within Each Source Dataset

Within each source dataset, the `shuffle_index` in `GPTDataset` is a **random permutation of all sample positions** (built from `random_seed` at index construction time). When `BlendedDataset` calls `self.datasets[dataset_id][dataset_sample_id]` with `dataset_sample_id = 0, 1, 2, ...`, it is accessing samples in their shuffled order, which covers the full dataset uniformly. Samples are not randomly drawn per-batch; the order is pre-determined at build time and then iterated sequentially.

> **Do you need to load all source datasets into RAM?** No. See Layer 1: the underlying token data is memory-mapped. Only the index arrays (a few MB per dataset) need to be in RAM. The OS page cache handles the rest.

> **MegatronBridge Config**
> | Key | Default | Controls |
> |-----|---------|----------|
> | `dataset.blend` | `None` | `([path1, path2, ...], [weight1, weight2, ...])` — same blend for all splits |
> | `dataset.blend_per_split` | `None` | `[(train_paths, train_weights), (val_paths, val_weights), ...]` — per-split blends |
> | `dataset.mid_level_dataset_surplus` | `0.005` | Buffer fraction to over-provision each source dataset |
> | `dataset.path_to_cache` | auto | Cache dir for blend index `.npy` files |

**Specifying blend paths in MegatronBridge** (from `data/loaders.py`):

```python
# Option A: inline in config
dataset = GPTDatasetConfig(
    blend=(
        ["/data/wiki_text", "/data/code_text"],
        [0.8, 0.2]
    ),
    ...
)

# Option B: interleaved list (weight, path, weight, path, ...)
# passed via data_paths = [0.8, "/data/wiki", 0.2, "/data/code"]

# Option C: text file (one "weight path" per line)
# passed via data_args_path = "/configs/data_paths.txt"

# Option D: JSON with per-split dicts
# { "train": [[w1, p1], [w2, p2]], "valid": [...], "test": [...] }
# passed via per_split_data_args_path = "/configs/per_split.json"
```

---

## DataLoader & Sampling

**Source**: `megatron/training/datasets/data_samplers.py`

### MegatronPretrainingSampler (default, `dataloader_type="single"`)

```python
# data_samplers.py:150
def __iter__(self):
    batch = []
    for idx in range(self.consumed_samples, self.total_samples):
        batch.append(idx)
        if len(batch) == self.micro_batch_size * data_parallel_size:
            start_idx = self.data_parallel_rank * self.micro_batch_size
            end_idx   = start_idx + self.micro_batch_size
            yield batch[start_idx:end_idx]
            batch = []
```

The sampler iterates **sequentially** from `consumed_samples` to `total_samples`. At each step, a batch of size `MBS × DP_size` is assembled; each DP rank takes a contiguous slice `[rank*MBS : (rank+1)*MBS]`.

Because the shuffling already happened inside `GPTDataset.shuffle_index` and `BlendedDataset.dataset_index`, sequential iteration over the `BlendedDataset` already produces well-shuffled, correctly-blended samples. No randomness is needed here.

### MegatronPretrainingRandomSampler (`dataloader_type="cyclic"`)

Used for fine-tuning-style loops where the dataset is small and you want true random sampling per epoch. Uses `torch.Generator` seeded by the epoch number (derived from `consumed_samples`):

```python
# data_samplers.py — simplified
generator = torch.Generator()
generator.manual_seed(self.epoch)
idx_range = torch.randperm(active_total_samples, generator=generator)
```

With `data_sharding=True`, each DP rank shuffles within its own bucket (strided assignment), preventing inter-rank communication for data routing.

### Relationship to Global Batch Size

```
Global Batch Size (GBS) = MBS × DP_size × gradient_accumulation_steps
```

Each iteration of the training loop consumes `GBS` samples. The sampler advances by exactly `GBS` positions in the pre-computed blend sequence. The blending algorithm ensures the global blend ratios hold across the entire run.

> **MegatronBridge Config**
> | Key | Default | Controls |
> |-----|---------|----------|
> | `dataset.dataloader_type` | `"single"` | `"single"` (sequential, pretraining) or `"cyclic"` (random per epoch, SFT) |
> | `dataset.data_sharding` | `True` | For cyclic mode: partition dataset across DP ranks |
> | `dataset.num_workers` | `2` | PyTorch DataLoader worker processes |
> | `dataset.pin_memory` | `True` | Pin memory for faster GPU transfer |
> | `dataset.persistent_workers` | `True` | Keep workers alive between iterations |
> | `train.global_batch_size` | required | Total samples per optimizer step |
> | `train.micro_batch_size` | required | Samples per GPU per forward pass |

---

## Determinism & Random Seeds

Seed control flows through three separate systems:

### 1. Dataset Index Construction (NumPy)

```python
# gpt_dataset.py:499
numpy_random_state = numpy.random.RandomState(self.config.random_seed)
# Used sequentially for:
document_index = _build_document_index(..., numpy_random_state, ...)  # doc shuffle
shuffle_index  = _build_shuffle_index(..., numpy_random_state)        # sample shuffle
```

`numpy.random.RandomState` is a deterministic PRNG — given the same seed, the same sequence of shuffle calls always produces the same permutations. Because both the document and shuffle indices are built in the same `__init__` call using the same `RandomState` object, the relative order of the two shuffle calls is fixed.

### 2. Cache Key (MD5 Hash)

The `.npy` cache files are named using an MD5 hash of a JSON string that encodes:
- dataset path
- `random_seed`
- `sequence_length`
- `split` / `split_matrix`
- tokenizer name

Changing any of these values invalidates the cache and forces a rebuild. This means you can safely cache indices for multiple configurations in the same directory.

### 3. Sampler Seeds (PyTorch)

For `dataloader_type="cyclic"`, `torch.Generator().manual_seed(epoch)` provides deterministic per-epoch shuffles. The epoch number is computed from `consumed_samples // dataset_size`.

For `RandomSeedDataset` (used to inject per-sample randomness for augmentation):
```python
# data_samplers.py:243
def __getitem__(self, idx):
    seed = idx + self.curr_seed
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    return self.dataset[idx]
```

### Seed Flow Summary

```
dataset.random_seed ──► numpy.RandomState ──► document_index shuffle
                    │                    └──► shuffle_index shuffle
                    └──► MD5 cache key (invalidation)

rng.seed ──────────────► torch global seed (model init, dropout, etc.)

epoch_number ──────────► torch.Generator ──► cyclic sampler shuffle
```

> **MegatronBridge Config**
> | Key | Default | Controls |
> |-----|---------|----------|
> | `dataset.random_seed` | `1234` | Seed for dataset index construction (document order, sample shuffle) |
> | `rng.seed` | `1234` | Global PyTorch/CUDA seed (model weights, dropout, augmentation) |
> | `checkpoint.save_rng` | `True` | Save full RNG state (torch CPU/CUDA, numpy, python) to checkpoint |
> | `checkpoint.load_rng` | `True` | Restore RNG state from checkpoint on resume |

---

## Checkpoint Resumption & Dataloader State

### What "Dataloader State" Means in Megatron

Unlike some frameworks that save a dataloader iterator snapshot, Megatron's built-in text dataloader does not have a separate state object to save. Its position is fully captured by a single integer: **`consumed_train_samples`**.

This works because:
1. The `BlendedDataset` / `GPTDataset` index arrays are **deterministically rebuilt** from (config + seed) — they don't change run to run.
2. The `MegatronPretrainingSampler` iterates linearly from `consumed_samples` to `total_samples`.
3. Therefore, knowing `consumed_samples` is sufficient to resume from exactly the right position.

### What Is Saved to Checkpoint

From MegatronBridge's `docs/training/checkpointing.md`, checkpoints have this structure:

```
checkpoint_dir/
├── latest_train_state.pt                      # points to latest iter
├── iter_N/
│   ├── __0_0.distcp, __0_1.distcp, ...       # model + optimizer shards (PyTorch DCP)
│   ├── .metadata                              # PyTorch DCP metadata
│   ├── common.pt                              # RNG states, misc non-sharded state
│   ├── metadata.json                          # MCore dist ckpt metadata
│   ├── run_config.yaml                        # Full ConfigContainer (incl. dataset config)
│   ├── train_state.pt                         # iteration, consumed_train/valid_samples
│   ├── tokenizer/                             # tokenizer files
│   └── dataloader_state/
│       ├── train_dataloader_dprank000.pt      # DP rank 0 dataloader state
│       ├── train_dataloader_dprank001.pt      # DP rank 1 dataloader state
│       └── ...
```

**`train_state.pt`** contains:
- `iteration` — current training step
- `consumed_train_samples` — total training samples consumed so far
- `consumed_valid_samples` — total validation samples consumed so far
- LR scheduler state

**`common.pt`** contains:
- `rng_state` — torch CPU RNG state
- `rng_state_cuda` — torch CUDA RNG state per device
- `rng_state_np` — numpy RNG state
- `rng_state_python` — Python `random` module state

**`run_config.yaml`** contains the full `ConfigContainer`, which includes `dataset.random_seed`, `dataset.blend`, `dataset.sequence_length` — everything needed to deterministically rebuild the index arrays.

**`dataloader_state/`** — for the standard text dataloader, these files contain only `consumed_samples`. For the Energon multimodal dataloader, they contain a richer iterator state.

### What Is NOT Saved (Deterministically Reconstructed)

The three index arrays (`document_index`, `sample_index`, `shuffle_index`) and the two blend arrays (`dataset_index`, `dataset_sample_index`) are **not saved to checkpoint**. They are rebuilt from the config and seed, which is much cheaper than storing potentially hundreds of GB of index data.

### Resumption Flow

```
1. Load checkpoint: read consumed_train_samples from train_state.pt
2. Restore RNG state from common.pt
3. Rebuild datasets: BlendedMegatronDatasetBuilder reads run_config.yaml
   → loads cached .npy index files (or rebuilds if cache missing)
4. Build sampler: MegatronPretrainingSampler(consumed_samples=N, ...)
5. Sampler starts iterating from position N
   → same blend sequence, same per-sample order, as if never interrupted
```

### The consumed_samples Backward Compatibility Path

If a checkpoint was saved before `consumed_train_samples` tracking was added:
```python
# training.py:3443
if iteration > 0 and consumed_train_samples == 0:
    consumed_train_samples = iteration * global_batch_size
```

This approximation is exact as long as GBS hasn't changed between runs.

> **MegatronBridge Config**
> | Key | Default | Controls |
> |-----|---------|----------|
> | `checkpoint.save` | `None` | Output directory for checkpoints |
> | `checkpoint.load` | `None` | Directory to resume from (loads latest by default) |
> | `checkpoint.ckpt_step` | `None` | Load a specific iteration instead of latest |
> | `checkpoint.save_interval` | `None` | Iterations between checkpoint saves |
> | `checkpoint.save_rng` | `True` | Save RNG state to `common.pt` |
> | `checkpoint.load_rng` | `True` | Restore RNG state on resume |
> | `checkpoint.save_optim` | `True` | Save optimizer state |
> | `checkpoint.async_save` | `False` | Save checkpoint in background while training continues |
> | `checkpoint.finetune` | `False` | Reset `iteration` to 0 when loading (for fine-tuning from pretrained ckpt) |

---

## Remote / S3 Data Streaming

**Source**: `megatron/core/datasets/indexed_dataset.py`, `megatron/core/datasets/object_storage_utils.py`

### Architecture

For remote datasets, the `.idx` file is downloaded to a local cache path at startup (done once on rank 0, then barrier so other ranks read from cache). The `.bin` file is **never fully downloaded** — it is streamed on demand via byte-range HTTP requests.

```
Local FS:  wiki.idx  (cached at path_to_idx_cache)
S3:        s3://bucket/wiki.bin  (streamed in 256 MiB chunks)
```

### S3 Backend (`_S3BinReader`)

```python
# indexed_dataset.py:532
def read(self, dtype, count, offset):
    size = count * DType.size(dtype)
    if self._cache and offset >= cache_start and offset+size <= cache_end:
        return numpy.frombuffer(self._extract_from_cache(offset, size), dtype=dtype)

    # Cache miss: fetch a 256 MiB chunk containing this offset
    bytes_start = (offset // self._cache_nbytes) * self._cache_nbytes
    bytes_end   = max(bytes_start + self._cache_nbytes, offset + size)
    self._cache = s3_client.get_object(
        Bucket=bucket, Key=key,
        Range=f"bytes={bytes_start}-{bytes_end-1}"
    )["Body"].read()
    return numpy.frombuffer(self._extract_from_cache(offset, size), dtype=dtype)
```

The 256 MiB chunk cache means that for a seq_len=2048 model with int16 tokens (4096 bytes/sample), each S3 GET serves ~65,000 samples before needing a refill. Sequential access patterns are cache-friendly.

### Multi-Storage Client (MSC) Backend

`_MultiStorageClientBinReader` uses the `multistorageclient` package, which abstracts over S3, GCS, Azure Blob, and local filesystem via a unified API. Enabled when the `MultiStorageClientFeature` is active (set by Megatron-Bridge infrastructure).

### Activation

Remote storage is activated by passing `s3://` or `msc://` prefixed paths in `blend`. The system detects the prefix via `is_object_storage_path()`:

```python
# GPTDataset.build_low_level_dataset (gpt_dataset.py:166)
if is_object_storage_path(dataset_path):
    object_storage_config = ObjectStorageConfig(
        path_to_idx_cache=self.config.object_storage_cache_path,
        bin_chunk_nbytes=256 * 1024 * 1024,  # 256 MiB
    )
    # mmap is disabled for remote paths
```

> **MegatronBridge Config**
> | Key | Default | Controls |
> |-----|---------|----------|
> | `dataset.object_storage_cache_path` | `None` | Local path to cache downloaded `.idx` files for S3/MSC datasets |
> | `dataset.blend` | `None` | Use `s3://bucket/prefix` paths to activate remote reading |

---

## Dataloader State Dict: Full Summary

### State Maintained During Training (in Memory)

| What | Where | Contents |
|------|-------|----------|
| Consumed sample count | `args.consumed_train_samples` | Integer, incremented by `global_batch_size` each step |
| Dataset index arrays | `GPTDataset.{document,sample,shuffle}_index` | numpy arrays, loaded via mmap |
| Blend index arrays | `BlendedDataset.{dataset,dataset_sample}_index` | numpy arrays, loaded via mmap |
| Sampler position | `MegatronPretrainingSampler.consumed_samples` | Integer cursor |
| RNG states | torch / numpy / python internal state | Per-process |

### State Saved to Checkpoint

| File | What is Saved | Notes |
|------|---------------|-------|
| `train_state.pt` | `consumed_train_samples`, `consumed_valid_samples`, `iteration`, LR scheduler state | The primary dataloader resumption signal |
| `common.pt` | torch CPU/CUDA RNG, numpy RNG, python RNG | Enables exact stochastic reproducibility after resume |
| `run_config.yaml` | Full `ConfigContainer` including `dataset.*` | Needed to rebuild index arrays identically |
| `dataloader_state/train_dataloader_dprank*.pt` | `consumed_samples` (text DL) or full iterator state (Energon) | One file per DP rank |
| `.npy` cache files | `document_index`, `sample_index`, `shuffle_index`, blend arrays | Stored in `path_to_cache`, **not inside** the checkpoint directory |

### State NOT Saved (Deterministically Reconstructed)

- The index arrays themselves (rebuilt from `random_seed` + config on resume)
- The internal Python DataLoader iteration position (reconstructed from `consumed_samples`)

---

## Consolidated MegatronBridge Config Reference

```python
from megatron.bridge.training.config import (
    ConfigContainer, GPTDatasetConfig, CheckpointConfig, TrainingConfig, RNGConfig
)

dataset = GPTDatasetConfig(
    # Data sources & blending
    blend=(
        ["/data/wiki_text", "/data/code_text", "/data/books"],
        [0.7, 0.2, 0.1]
    ),
    # blend_per_split = [  # alternative: per-split blends
    #     (["/data/wiki_train"], [1.0]),   # train
    #     (["/data/wiki_val"],   [1.0]),   # valid
    #     None,                            # test (use None to skip)
    # ],

    # Sequencing
    seq_length=2048,          # sequence length — determines sample index slicing
    split="9999,8,2",         # train/valid/test document split ratios

    # Randomness
    random_seed=1234,         # seed for document/sample/shuffle index construction

    # Caching
    path_to_cache="/data/cache/indices",  # dir for .npy index files
    defer_npy_index_mmap=False,           # True = lazy mmap (large multi-node jobs)

    # Document boundary behavior
    reset_position_ids=False,    # reset position IDs at EOD token
    reset_attention_mask=False,  # mask attention across doc boundaries
    eod_mask_loss=False,         # mask loss on EOD tokens

    # DataLoader
    dataloader_type="single",    # "single" = sequential (pretrain), "cyclic" = random (SFT)
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    data_sharding=True,          # for cyclic mode: per-DP-rank bucket shuffling

    # Remote data (S3 / MSC)
    # blend=(["s3://my-bucket/wiki_text"], [1.0]),  # use s3:// prefix to activate
    object_storage_cache_path="/tmp/idx_cache",  # local cache for .idx files
)

train = TrainingConfig(
    global_batch_size=1024,  # total samples per optimizer step
    micro_batch_size=4,      # samples per GPU per forward pass
    train_iters=100000,
)

checkpoint = CheckpointConfig(
    save="/checkpoints/my_run",
    load="/checkpoints/my_run",  # resume from latest
    save_interval=1000,
    save_rng=True,   # save torch/numpy/python RNG state
    load_rng=True,   # restore RNG state on resume
    save_optim=True,
    async_save=False,
)

rng = RNGConfig(seed=1234)  # global PyTorch seed (model init, dropout)
```

---

## End-to-End Data Flow Summary

```
Preprocessing (once):
  raw .jsonl → tokenize → preprocess_data.py → wiki.bin + wiki.idx

Training startup:
  GPTDatasetConfig + random_seed
      │
      ├─► Check cache: {MD5_hash}-document_index.npy exists?
      │     ├── Yes: numpy.load(..., mmap_mode="r")
      │     └── No (rank 0 only): build_document_index, build_sample_idx (C++),
      │                           build_shuffle_index → save to cache
      │
      └─► BlendedDataset: build_blending_indices (C++) → dataset_index.npy

Training loop:
  for step in range(start_iter, train_iters):
      # MegatronPretrainingSampler yields [idx_0, idx_1, ..., idx_{MBS-1}]
      # (starting from consumed_samples, each DP rank gets its slice)
      for sample_idx in batch:
          dataset_id        = BlendedDataset.dataset_index[sample_idx]
          dataset_sample_id = BlendedDataset.dataset_sample_index[sample_idx]
          shuffled_idx      = GPTDataset[dataset_id].shuffle_index[dataset_sample_id]
          doc_range         = GPTDataset[dataset_id].sample_index[shuffled_idx : shuffled_idx+2]
          tokens            = IndexedDataset.get(doc_id, offset, length)  # mmap read
      args.consumed_train_samples += global_batch_size

Checkpoint save (every save_interval steps):
  train_state.pt  ← {iteration, consumed_train_samples, consumed_valid_samples}
  common.pt       ← {torch_rng, cuda_rng, numpy_rng, python_rng}
  run_config.yaml ← full ConfigContainer

Resume from checkpoint:
  1. Load consumed_train_samples from train_state.pt
  2. Load RNG states from common.pt
  3. Rebuild dataset: same config + same seed → same index arrays (from cache or rebuilt)
  4. MegatronPretrainingSampler(consumed_samples=N) → resumes exactly where training stopped
```
