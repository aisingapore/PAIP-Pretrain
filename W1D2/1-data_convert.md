# Data Preprocessing Pipeline

Converts raw JSONL datasets into Megatron-LM's binary `IndexedDataset` format (`.bin` + `.idx` file pairs) for pretraining.

## Quick Start

```bash
cd /mnt/weka/aisg/model_training_team/code_forge/yuli/megatron_bridge

export PYTHONPATH=/mnt/weka/aisg/source_files/megatron_yuli:$PYTHONPATH

python scripts/data_prep/run_pipeline.py all \
  --input-dir  /mnt/weka/aisg/model_training_team/code_forge/yuli/shared_fs/data/raw_knowledge/EN/EN_Wikibooks/raw/text \
  --output-dir /mnt/weka/aisg/model_training_team/code_forge/yuli/shared_fs/data/megatron/qwen3/EN_Wikibooks \
  --dataset-name EN_Wikibooks \
  --text-key   raw_content \
  --tokenizer-model Qwen/Qwen3-4B \
  --workers    20
```

This runs the full pipeline: **tokenize → merge → verify → count**.

---

## Files

```
scripts/data_prep/
  run_pipeline.py          # Orchestrator — the main entry point
  preprocess_data_spark.py # PySpark tokenizer backend (requires Java 17+)
  merge_datasets.py        # Merges per-file shards into a single dataset
  verify_data.py           # Partition + merge verification checks
  count_tokens.py          # Reports document and token counts
  utils.py                 # Shared: text key validation, file grouping, config loader
  config.yaml              # Default configuration
```

---

## Configuration

`config.yaml` holds the defaults. Any value can be overridden via CLI:

```yaml
tokenizer_model: Qwen/Qwen3-4B
tokenizer_type: HuggingFaceTokenizer

# "mp" = Python multiprocessing (default, no Java required)
# "spark" = PySpark (requires Java 17+ and JAVA_HOME set)
preprocess_backend: mp

megatron_lm_dir: /mnt/weka/aisg/source_files/megatron_yuli

workers: 20
log_interval: 100000
data_file_extension: jsonl
skip_processed: true

group_target_size_gb: 10   # PySpark only: target batch size per Spark job
group_mean_threshold_gb: 5 # PySpark only: files larger than this are not batched

verification_num_samples: 20
```

---

## Sub-commands

`run_pipeline.py` exposes individual steps so you can re-run any stage independently:

```bash
# Full pipeline
python run_pipeline.py all      --input-dir ... --output-dir ... --dataset-name ... --text-key ...

# Individual steps
python run_pipeline.py tokenize --input-dir ... --output-dir ... --dataset-name ... --text-key ...
python run_pipeline.py merge    --output-dir ... --dataset-name ...
python run_pipeline.py verify   --input-dir ... --output-dir ... --dataset-name ... --text-key ...
python run_pipeline.py count    --output-dir ... --dataset-name ...
```

### Key CLI Options

| Flag | Default | Description |
|---|---|---|
| `--input-dir` | — | Directory containing raw JSONL files |
| `--output-dir` | — | Output root (will create `converted/` and `merged/` subdirs) |
| `--dataset-name` | — | Name used for output file naming |
| `--text-key` | `text` | JSON field containing the document text |
| `--tokenizer-model` | from config | HuggingFace model name/path |
| `--workers` | from config | Number of parallel workers |
| `--backend` | from config | `mp` or `spark` |
| `--config` | `config.yaml` | Path to config YAML |

---

## Output Structure

```
output-dir/
  converted/
    {dataset}_{filename}_{text_key}_document.bin   # per-file binary token data
    {dataset}_{filename}_{text_key}_document.idx   # per-file index (doc lengths + offsets)
  merged/
    {dataset}_merged.bin                           # final merged token data
    {dataset}_merged.idx                           # final merged index
```

The `merged/` files are what training jobs consume (referenced in `data_config.yaml`).

---

## Pipeline Steps Explained

### Step 1: Tokenize

**What it does:** Reads each raw JSONL file, tokenizes the text field of every document using the specified HuggingFace tokenizer, and writes the results as Megatron `IndexedDataset` binary files.

Each document is stored as a flat integer array of token IDs followed by an end-of-document (`<eod>`) token. The `.bin` file stores the raw token ID arrays back-to-back; the `.idx` file stores the document boundaries (offsets and lengths) so any document can be retrieved in O(1) by index.

**File grouping (PySpark backend only):** For datasets with many small files, the pipeline first groups them into ~10 GB batches using `input_jsonl_group_by_size()`. Each batch becomes a single Spark job and produces one `.bin/.idx` pair. Files larger than 5 GB mean are processed individually.

**Backends:**

- **`mp` (multiprocessing, default):** Calls `megatron_yuli/tools/preprocess_data.py` directly. Uses Python's `multiprocessing.Pool` to parallelise tokenization across CPU cores. One worker per `--workers` processes documents concurrently and writes to a shared `IndexedDatasetBuilder`. No Java required.

- **`spark`:** Calls `preprocess_data_spark.py`. Uses PySpark's distributed execution model. The JSONL is read into a Spark DataFrame, then each Spark partition is tokenized independently (one `IndexedDatasetBuilder` per partition), and the per-partition shards are merged within the script using `IndexedDatasetBuilder.add_index()`. Requires Java 17+.

#### PySpark conversion workflow in detail

```
Raw JSONL
    │
    ▼
SparkSession.read.json()
    │  Reads all input files into a single DataFrame.
    │  Each Row corresponds to one JSON document.
    │  Ordering metadata (__file_index__, __line_number__) is added to
    │  preserve the original document order across partitions.
    │
    ▼
rdd.mapPartitionsWithIndex(process_and_write_partition)
    │  Each Spark executor processes one partition independently.
    │  Within each partition:
    │    - A fresh Encoder is initialised (loads the tokenizer).
    │    - Each Row is converted to JSON, then tokenized.
    │    - Token IDs are written to a per-partition shard:
    │        {output_prefix}_{text_key}_document_{partition_index}.bin/.idx
    │    - Long documents that cause OOM are chunked into 300k-char pieces.
    │
    ▼
Phase 2: Shard merge (driver)
    │  After all partitions complete, the driver iterates the per-partition
    │  shards in sorted order and calls IndexedDatasetBuilder.add_index()
    │  on each, producing a single merged file for this input group:
    │    {output_prefix}_{text_key}_document.bin/.idx
    │
    ▼
Phase 3: Cleanup
    │  The intermediate per-partition .bin/.idx files are deleted,
    │  leaving only the merged output.
    │
    ▼
Final output: {output_prefix}_{text_key}_document.{bin,idx}
```

The key advantage of Spark over plain multiprocessing is horizontal scalability: when a dataset has dozens of large files, each file group can be processed by a separate Spark job, and within each job, the Spark executor model handles memory pressure by controlling partition sizes (configured via `spark.sql.files.maxPartitionBytes`).

### Step 2: Merge

**What it does:** Combines all per-file `.bin/.idx` pairs from `converted/` into a single `merged/{dataset}_merged.{bin,idx}` that training can reference as one contiguous dataset.

Uses `IndexedDatasetBuilder.add_index()` from Megatron's core, which appends the binary content of each shard and adjusts the index offsets so the merged file presents a unified document namespace.

**Special case:** If `converted/` contains only one `.bin/.idx` pair (single-file dataset, as with EN_Wikibooks), the merge step copies instead of merging, avoiding the overhead of reading and rewriting the same data.

### Step 3: Verify

Two checks run back-to-back:

**Partition verification** (checks `converted/` against the raw input):
1. All `.bin` and `.idx` files are non-zero in size.
2. Total raw JSONL line count == total document count across all shards.
3. For single-file datasets: randomly samples 20 documents, decodes their token IDs back to text, and compares the first 500 characters against the original JSONL line.

**Merge verification** (checks `merged/` against `converted/`):
1. Sum of `(num_docs, num_tokens)` across all shards in `converted/` equals the totals in `merged/`.
2. Randomly samples 20 global indices from the merged dataset, maps each to its source partition and local index, and asserts the token sequence lengths match.

### Step 4: Count

Reads the merged `.idx` file and reports:
- **Documents**: number of entries in `dataset.sequence_lengths`
- **Tokens**: sum of `dataset.sequence_lengths`

---

## Choosing a Backend

| | `mp` | `spark` |
|---|---|---|
| Java required | No | Yes (17+) |
| Setup | None | `apt-get install openjdk-17-jdk-headless` + `export JAVA_HOME=...` |
| Best for | Single large files, simple setups | Many small files, grouped batches |
| Overhead | Low | Higher (JVM startup ~10s) |
| EN_Wikibooks (469 MB, 1 file) | ~7 min | ~10 min |

For this container, install Java once per session:
```bash
apt-get install -y openjdk-17-jdk-headless
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
```

Then pass `--backend spark` to `run_pipeline.py`.

---

## Test Run Results (EN_Wikibooks)

Dataset: `enwikibooks_dedup.jsonl` — 489 MB, 69,695 documents, text key `raw_content`

| Backend | Time | Documents | Tokens |
|---|---|---|---|
| `mp` | ~7 min | 69,695 | 113,384,260 |
| `spark` | ~10 min | 69,695 | 113,384,260 |

Both backends produce identical output. All verification checks passed (20/20 sample decodes, 20/20 index checks).
