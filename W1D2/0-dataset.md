# Megatron's Dataset Architecture: From Raw Text to Training Samples

Before you run the preprocessing pipeline (covered in [1-data_convert.md](1-data_convert.md)),
it helps to understand the data structures it creates and how training reads from them.

```
Raw JSONL docs ──► preprocess_data.py ──► .bin + .idx ──► GPTDataset ──► training loop
   (text)          (tokenize + EOD)       (binary)        (slice into     (consume
                                                           sequences)      batches)
```

The architecture has two layers:

| Layer | Class | Responsibility |
|-------|-------|----------------|
| 1 | `IndexedDataset` | Storage & random access — the `.bin/.idx` file pair |
| 2 | `GPTDataset` | Sequencing & shuffling — carve fixed-length samples from raw documents |

---

## Layer 1: IndexedDataset — The `.bin/.idx` File Pair

**Source**: `megatron/core/datasets/indexed_dataset.py`

### 1.1 Document Packing with EOD Tokens

During preprocessing, each raw document is tokenized and an **end-of-document (EOD)** token
is appended. Documents are then written back-to-back into a single flat binary file (`.bin`).
The `.idx` file records each document's byte offset and token count, serving as a table of
contents.

```
.bin file (flat binary, all documents concatenated):
┌──────────────────┬─────┬──────────────────┬─────┬──────────────────┬─────┐
│  Doc 0 tokens    │ EOD │  Doc 1 tokens    │ EOD │  Doc 2 tokens    │ EOD │
│  (350 tokens)    │     │  (1200 tokens)   │     │  (87 tokens)     │     │
└──────────────────┴─────┴──────────────────┴─────┴──────────────────┴─────┘
  ^                        ^                        ^
  offset=0                 offset=1404              offset=6408
  (byte offsets recorded in .idx)
```

Key properties of this design:

- **Variable-length documents** are stored naturally — no padding, no wasted space.
- **EOD tokens** mark document boundaries in the token stream. They play a critical role
  later during training (see [Section 4](#4-cross-document-attention-prevention)).
- The `.bin` file is **write-once** — training never modifies it. Different sequence lengths,
  different shuffles, and different epochs all read from the same binary data.


### 1.2 The `.idx` File: A Table of Contents

The `.idx` file has a fixed binary layout that stores three arrays after a short header:

```
.idx file layout:
┌───────────┬──────────┬──────────┬───────────────┬──────────────────┐
│ Header    │ Version  │ DType    │ Seq Count (N) │ Doc Count (D)    │
│ 9 bytes   │ uint64   │ uint8    │ uint64        │ uint64           │
│ MMIDIDX   │ 1        │ e.g. 4   │               │                  │
├───────────┴──────────┴──────────┴───────────────┴──────────────────┤
│ sequence_lengths:  int32 × N   (token count per sequence)          │
│ sequence_pointers: int64 × N   (byte offset into .bin)             │
│ document_indices:  int64 × D   (sequence indices at doc boundaries)│
└────────────────────────────────────────────────────────────────────┘
```

These arrays are written by `_IndexWriter.write()` (indexed_dataset.py:175):

```python
def write(self, sequence_lengths, sequence_modes, document_indices):
    sequence_pointers = self._sequence_pointers(sequence_lengths)

    sequence_count = len(sequence_lengths)
    self.idx_writer.write(struct.pack("<Q", sequence_count))

    document_count = len(document_indices)
    self.idx_writer.write(struct.pack("<Q", document_count))

    # the number of tokens per sequence
    self.idx_writer.write(numpy.array(sequence_lengths, dtype=numpy.int32).tobytes(order="C"))

    # the byte offsets for all sequences
    self.idx_writer.write(numpy.array(sequence_pointers, dtype=numpy.int64).tobytes(order="C"))

    # the sequence indices marking the end of each document
    self.idx_writer.write(numpy.array(document_indices, dtype=numpy.int64).tobytes(order="C"))
```

With this layout, any of the three arrays can be accessed independently via a byte-offset
calculation — no need to parse the entire file.


### 1.3 Efficient Random Access via mmap

A pretraining corpus can be hundreds of gigabytes or more. Loading it all into RAM would be
impractical. Instead, Megatron uses **memory-mapped I/O** (`numpy.memmap`) to map both
`.idx` and `.bin` files into the process's virtual address space. The operating system then
loads physical RAM pages **on demand** — only when specific byte ranges are actually read.

**Reading the index** — `_IndexReader.__init__()` (indexed_dataset.py:280):

```python
self.bin_buffer_mmap = numpy.memmap(idx_path, mode="r", order="C")
self.bin_buffer = memoryview(self.bin_buffer_mmap)

# Zero-copy views — no data is copied to RAM
self.sequence_lengths = numpy.frombuffer(
    self.bin_buffer, dtype=numpy.int32, count=self.sequence_count, offset=offset
)
self.sequence_pointers = numpy.frombuffer(
    self.bin_buffer, dtype=numpy.int64, count=self.sequence_count,
    offset=offset + self.sequence_lengths.nbytes,
)
self.document_indices = numpy.frombuffer(
    self.bin_buffer, dtype=numpy.int64, count=self.document_count,
    offset=offset + self.sequence_lengths.nbytes + self.sequence_pointers.nbytes,
)
```

**Reading the binary data** — `_MMapBinReader.read()` (indexed_dataset.py:405):

```python
def read(self, dtype, count, offset):
    return numpy.frombuffer(self._bin_buffer, dtype=dtype, count=count, offset=offset)
```

This single line is the core of random access: given a byte `offset` from the `.idx` file
and a token `count`, it returns a numpy array **without copying any data**. The OS
transparently fetches only the pages backing that byte range from disk.

**The `get()` method** — `IndexedDataset.get()` (indexed_dataset.py:843):

```python
def get(self, idx, offset=0, length=None):
    sequence_pointer, sequence_length, sequence_mode = self.index[idx]
    if length is None:
        length = sequence_length - offset
    sequence_pointer += offset * DType.size(self.index.dtype)
    return self.bin_reader.read(dtype=self.index.dtype, count=length, offset=sequence_pointer)
```

To retrieve document #50,000 out of a million documents:

1. Look up `sequence_pointers[50000]` from the `.idx` mmap → get the byte offset (O(1))
2. Call `bin_reader.read(dtype, count, offset)` → the OS page-faults in just those bytes
3. Return a numpy array backed by the page cache

There is **no scanning** through documents 0–49,999. This is O(1) random access regardless
of dataset size. For a 1 TB dataset, only the pages you've actually touched consume RAM —
the OS evicts old pages automatically under memory pressure.


### 1.4 Pickling for DataLoader Workers

PyTorch DataLoader spawns multiple worker processes. Each needs access to the dataset, but
sharing an mmap across a `fork()` is fragile. Megatron solves this by having `__getstate__`
serialize only the file path and config — not the mmap. Each worker's `__setstate__` then
independently re-opens the mmap.

The result: **N workers = N independent mmaps**, all reading from the same underlying OS page
cache with zero data duplication in RAM.

---

## Layer 2: GPTDataset — Turning Documents into Training Samples

**Source**: `megatron/core/datasets/gpt_dataset.py`

`GPTDataset` wraps one `IndexedDataset` and builds three index arrays that answer: *in what
order, and at what positions, do we read token sequences from this dataset?*


### 2.1 The Three-Index System

```
document_index : int32[num_epochs × num_docs]   — shuffled doc IDs to walk through
sample_index   : int32[num_samples+1, 2]         — (doc_idx_index, offset) boundaries
shuffle_index  : uint32[num_samples]              — random permutation of sample IDs
```

When the training loop requests sample `idx`, the access path is:

```
__getitem__(idx)
      │
      ▼
shuffle_index[idx] ──► shuffled_idx
      │
      ▼
sample_index[shuffled_idx]   ──► (doc_idx_start, offset_start)
sample_index[shuffled_idx+1] ──► (doc_idx_end,   offset_end)
      │
      ▼
document_index[doc_idx_start .. doc_idx_end] ──► actual document IDs
      │
      ▼
IndexedDataset.get(doc_id, offset, length)   ──► token arrays from .bin
      │
      ▼
numpy.concatenate(parts) ──► one training sample
```

Here is the retrieval logic (gpt_dataset.py:301):

```python
def _query_document_sample_shuffle_indices(self, idx):
    # Step 1: shuffle mapping
    idx = self.shuffle_index[idx]

    # Step 2: get the beginning and end documents and offsets
    doc_index_beg, doc_index_beg_offset = self.sample_index[idx]
    doc_index_end, doc_index_end_offset = self.sample_index[idx + 1]

    # Step 3: retrieve token arrays from .bin
    sample_parts = []
    if doc_index_beg == doc_index_end:
        # Sample spans a single document
        sample_parts.append(
            self.dataset.get(
                self.document_index[doc_index_beg],
                offset=int(doc_index_beg_offset),
                length=doc_index_end_offset - doc_index_beg_offset + ...,
            )
        )
    else:
        # Sample spans multiple documents
        for i in range(doc_index_beg, doc_index_end + 1):
            offset = 0 if i > doc_index_beg else doc_index_beg_offset
            length = None if i < doc_index_end else doc_index_end_offset + ...
            sample_parts.append(
                self.dataset.get(self.document_index[i], offset=int(offset), length=length)
            )

    return numpy.concatenate(sample_parts, dtype=numpy.int64), ...
```

**Important**: the `MegatronPretrainingSampler` iterates indices sequentially (0, 1, 2, …).
All shuffling lives inside the dataset via `shuffle_index`, making the data ordering entirely
deterministic from the seed alone — critical for reproducible training and checkpoint
resumption.


### 2.2 Building the Indices

**Document index** — `_build_document_index()` (gpt_dataset.py:643):
tiles the document IDs `num_epochs` times, then shuffles the entire tiled array. This
interleaves documents from different epochs. If `separate_final_epoch` is set (threshold:
80%), the last epoch is shuffled independently to avoid over-representing partial-epoch data.

**Sample index** — built by a C++ function `build_sample_idx` (helpers.cpp):
walks the document_index and carves out fixed-length token sequences:

```
while samples remain:
    remaining = seq_length + 1          # +1 for the label-shifted token
    while remaining > 0:
        doc_tokens = sizes[document_idx[doc_idx]] - doc_offset
        remaining -= doc_tokens
        if remaining <= 0:
            doc_offset += (remaining + doc_tokens - 1)   # stay in current doc
        else:
            doc_idx++; doc_offset = 0                     # advance to next doc
    record sample_index[i] = (doc_idx, doc_offset)
```

When a document runs out of tokens mid-sequence, the builder advances to the next document.
The EOD token from preprocessing naturally appears at the boundary.

**Shuffle index** — `_build_shuffle_index()` (gpt_dataset.py:677):
a random permutation of sample indices, optionally with the final epoch shuffled separately.


### 2.3 Flexible Sequence Length — Same Data, Different Slicing

The `.bin` file is permanent storage. The three index arrays are **ephemeral** and rebuilt for
each `sequence_length` value. Change `seq_length` from 2048 to 4096 and you get half as many,
twice-as-long samples — from the identical binary data.

```
.bin:  [===Doc0===|EOD|==Doc1==|EOD|=====Doc2=====|EOD|==Doc3==|EOD]

seq_length=4:  [samp0][samp1][samp2][samp3][samp4][samp5]
seq_length=8:  [  sample 0  ][  sample 1  ][  sample 2  ]

Same binary data, different slicing. Only the sample_index changes.
```

The index files are cached as `.npy` files, keyed by an MD5 hash that encodes: dataset path,
split, `random_seed`, **`sequence_length`**, tokenizer, and split ratios. Any change to any of
these inputs produces a different hash and triggers a rebuild.

This design means you can experiment with sequence length schedules (e.g. short sequences
early in training, longer ones later) without reprocessing your corpus — just rebuild the
lightweight index files.

---

## 3. Cross-Document Attention Prevention

When `build_sample_idx` fills a sequence and the current document runs out of tokens, it
advances to the next document. The resulting sample contains tokens from multiple documents,
with EOD tokens at the boundaries:

```
[...end of Doc A tokens...][EOD][start of Doc B tokens...]
```

Without intervention, the model's self-attention would allow Doc B tokens to attend to Doc A
tokens — leaking information across unrelated documents. Megatron prevents this at training
time through `_get_ltor_masks_and_position_ids()` (gpt_dataset.py:709):

```python
def _get_ltor_masks_and_position_ids(
    data, eod_token, reset_position_ids, reset_attention_mask,
    eod_mask_loss, create_attention_mask,
):
    seq_length = data.numel()

    # Step 1: Start with standard causal (lower-triangular) attention mask
    if create_attention_mask:
        attention_mask = torch.tril(
            torch.ones((seq_length, seq_length), device=data.device)
        ).unsqueeze(0)

    # Step 2: Loss mask — optionally zero out loss on EOD tokens
    loss_mask = torch.ones(seq_length, dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Step 3: Position IDs
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)

    # Step 4: Find EOD positions and apply resets
    if reset_position_ids or reset_attention_mask:
        eod_index = position_ids[data == eod_token]

        prev_index = 0
        for j in range(eod_index.numel()):
            i = eod_index[j]

            # Zero out cross-document attention
            if reset_attention_mask and attention_mask is not None:
                attention_mask[0, (i + 1) :, : (i + 1)] = 0

            # Reset position counter after each document
            if reset_position_ids:
                position_ids[(i + 1) :] -= i + 1 - prev_index
                prev_index = i + 1

    # Convert to binary mask
    if attention_mask is not None:
        attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids
```

The critical line is `attention_mask[0, (i+1):, :(i+1)] = 0` — this zeros out all attention
weights from positions **after** the EOD to positions **at or before** the EOD. Visually:

```
Sequence: [A₁  A₂  A₃  EOD  B₁  B₂  B₃]

Standard causal mask:           With reset_attention_mask:

      A₁ A₂ A₃ EOD B₁ B₂ B₃         A₁ A₂ A₃ EOD B₁ B₂ B₃
  A₁ [ 1  ·  ·  ·  ·  ·  · ]    A₁ [ 1  ·  ·  ·  ·  ·  · ]
  A₂ [ 1  1  ·  ·  ·  ·  · ]    A₂ [ 1  1  ·  ·  ·  ·  · ]
  A₃ [ 1  1  1  ·  ·  ·  · ]    A₃ [ 1  1  1  ·  ·  ·  · ]
 EOD [ 1  1  1  1  ·  ·  · ]   EOD [ 1  1  1  1  ·  ·  · ]
  B₁ [ 1  1  1  1  1  ·  · ]    B₁ [ ·  ·  ·  ·  1  ·  · ]  ← zeroed
  B₂ [ 1  1  1  1  1  1  · ]    B₂ [ ·  ·  ·  ·  1  1  · ]  ← zeroed
  B₃ [ 1  1  1  1  1  1  1 ]    B₃ [ ·  ·  ·  ·  1  1  1 ]  ← zeroed
```

With `reset_position_ids`, position IDs also restart from 0 after each EOD, so the model's
positional encoding treats each document segment as starting from position 0 — preventing
positional information from leaking across documents.


### Configuration Flags

| Flag | Effect |
|------|--------|
| `reset_attention_mask` | Zero out attention from post-EOD tokens to pre-EOD tokens |
| `reset_position_ids` | Restart position counter at 0 after each EOD |
| `eod_mask_loss` | Exclude EOD tokens from loss computation |
| `create_attention_mask` | Generate the mask tensor (disable if the attention kernel handles masking internally) |

All four flags live in `GPTDatasetConfig` (gpt_dataset.py).


### Enabling Cross-Document Masking in MegatronBridge

MegatronBridge's `GPTDatasetConfig` inherits these four flags directly from Megatron-Core.
The **upstream default is `False` for all reset flags** — meaning packed multi-document
sequences get full causal attention across document boundaries, which is semantically
incorrect.

**Our launcher sets the correct defaults.** In `configs/config.yaml`, all three flags are
`true` by default, and `resolve_config.py` wires them to `cfg.dataset.*`. Any job launched
through our pipeline gets cross-document masking automatically:

```yaml
# configs/config.yaml (already set)
defaults:
  reset_attention_mask: true    # block cross-doc attention
  reset_position_ids: true      # restart pos encoding per document
  eod_mask_loss: true           # don't compute loss on EOD tokens
```

To disable (e.g. for debugging or reproducing upstream behavior), override via CLI:

```bash
reset_attention_mask=false reset_position_ids=false eod_mask_loss=false
```

**Config chain**: `config.yaml` → `launcher.py` (resolved YAML) → `resolve_config.py`
`FLAT_TO_CONFIG` mapping → `cfg.dataset.reset_attention_mask` etc. on `GPTDatasetConfig`
→ `GPTDataset.__init__()` → each `__getitem__()` call invokes
`_get_ltor_masks_and_position_ids()` with the flag values → the function scans for EOD
tokens and applies the masking logic shown above.

MegatronBridge validates that all three reset flags are explicitly set during config
finalization — if you set one, set all three to avoid assertion errors.

> **Memory cost warning (Method A — dense mask)**: This approach materializes a full
> `[seq_length, seq_length]` attention mask tensor per sample. At `seq_length=4096` this is
> manageable (~64 MB in float32), but at `seq_length=128K` a single mask would require ~64 GB
> — clearly impractical.
>
> **Method B — `cu_seqlens` / THD format**: For long-context pretraining, the efficient
> alternative is to pass document boundary indices directly to a FlashAttention kernel that
> applies block-diagonal masking without materializing the full matrix. FlashAttention
> supports this via `cu_seqlens` arguments. MegatronBridge's forward pass already has
> infrastructure for `cu_seqlens` (used in the SFT/finetuning path via `GPTSFTPackedDataset`),
> but it is **not yet wired into the standard pretraining path**. When
> `create_attention_mask=False`, Megatron delegates masking to the attention kernel — but the
> dataset must emit `cu_seqlens` tensors for this to work correctly with document packing.
>
> In practice: use Method A (the three config flags above) for moderate sequence lengths
> (≤8K). For long-context pretraining (32K+), Method B requires patching `GPTDataset` to scan
> EOD positions and emit `cu_seqlens` at sample retrieval time.

---

## 4. Index Caching and Reproducibility

The three index arrays (`document_index`, `sample_index`, `shuffle_index`) are saved as `.npy`
files under the cache directory. The filename is keyed by an **MD5 hash** of a unique
description string encoding: dataset path, split, `random_seed`, `sequence_length`, tokenizer,
and split ratios. Any change to any input produces a different hash and triggers a rebuild.

On subsequent runs, the indices are loaded lazily via mmap:

```python
document_index = numpy.load(path_to_document_index, allow_pickle=True, mmap_mode='r')
sample_index   = numpy.load(path_to_sample_index,   allow_pickle=True, mmap_mode='r')
shuffle_index  = numpy.load(path_to_shuffle_index,  allow_pickle=True, mmap_mode='r')
```

The `defer_npy_index_mmap=True` option delays even opening these files until the first
`__getitem__` call. This is useful for multi-node jobs where rank 0 builds the indices while
all other ranks wait at a barrier — they don't attempt to mmap files that may not exist yet.

---

## What's Next

Now that you understand the `.bin/.idx` format and how `GPTDataset` slices it into training
samples, [1-data_convert.md](1-data_convert.md) walks through the preprocessing pipeline that
creates these files from raw JSONL.
