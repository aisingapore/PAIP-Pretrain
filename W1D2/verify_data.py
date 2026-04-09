"""
Verify Megatron-format preprocessed data against raw JSONL source.

Two verification modes:
  partition  — verify that converted/ shards match the raw input files
  merge      — verify that merged/ output matches the sum of converted/ shards
"""
import argparse
import json
import os
import random

import numpy as np
from megatron.core.datasets.indexed_dataset import IndexedDataset
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def get_stats(prefix):
    """Return (num_docs, num_tokens) for an IndexedDataset at prefix."""
    ds = IndexedDataset(prefix)
    num_docs = ds.sequence_lengths.shape[0]
    num_tokens = int(np.sum(ds.sequence_lengths))
    return num_docs, num_tokens


def decode_sample(tokenizer, prefix, idx):
    """Decode a single sample from an IndexedDataset."""
    ds = IndexedDataset(prefix)
    return tokenizer.decode(ds[idx], skip_special_tokens=True)


def find_converted_prefixes(converted_dir):
    """Return sorted list of unique .bin/.idx prefixes in converted_dir."""
    prefixes = set()
    for fname in os.listdir(converted_dir):
        if fname.endswith(".bin") or fname.endswith(".idx"):
            prefixes.add(os.path.splitext(fname)[0])
    return sorted(prefixes)


# ---------------------------------------------------------------------------
# Partition verification
# ---------------------------------------------------------------------------

def verify_partition(input_dir, converted_dir, dataset_name, text_key,
                     tokenizer_model, num_samples):
    """Verify converted/ shards match the raw JSONL input."""
    print(f"\n{'='*60}")
    print(f"PARTITION VERIFICATION")
    print(f"  input_dir:     {input_dir}")
    print(f"  converted_dir: {converted_dir}")
    print(f"  dataset_name:  {dataset_name}")
    print(f"  text_key:      {text_key}")
    print(f"{'='*60}\n")

    # 1. Find all converted prefixes
    prefixes = find_converted_prefixes(converted_dir)
    print(f"Found {len(prefixes)} shard(s) in converted/:")
    for p in prefixes:
        print(f"  {p}")

    # 2. Check all files are non-zero
    errors = 0
    for p in prefixes:
        for ext in (".bin", ".idx"):
            path = os.path.join(converted_dir, p) + ext
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                print(f"  ERROR: {path} is missing or zero-size")
                errors += 1
    if errors == 0:
        print("  PASS: all shard files are non-zero")
    else:
        print(f"  FAIL: {errors} file(s) missing or zero-size")

    # 3. Check total line count matches total processed samples
    raw_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".jsonl") or f.endswith(".json")
    ])
    total_raw_lines = 0
    for rf in raw_files:
        with open(rf) as f:
            n = sum(1 for _ in f)
        total_raw_lines += n
        print(f"  Raw {os.path.basename(rf)}: {n:,} lines")

    total_processed = 0
    for p in prefixes:
        full_prefix = os.path.join(converted_dir, p)
        n_docs, _ = get_stats(full_prefix)
        total_processed += n_docs
        print(f"  Shard {p}: {n_docs:,} docs")

    print(f"\n  Total raw lines:       {total_raw_lines:,}")
    print(f"  Total processed docs:  {total_processed:,}")
    if total_raw_lines == total_processed:
        print("  PASS: line count matches")
    else:
        diff = abs(total_raw_lines - total_processed)
        print(f"  WARN: mismatch by {diff:,} (may be due to empty lines being skipped)")

    # 4. Sample decode check (single-file datasets only, where mapping is 1:1)
    if len(raw_files) == 1 and len(prefixes) == 1:
        print(f"\n  Sample decode check ({num_samples} samples)...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        raw_file = raw_files[0]
        shard_prefix = os.path.join(converted_dir, prefixes[0])
        ds = IndexedDataset(shard_prefix)
        dataset_len = len(ds)

        # Read all raw lines into memory (for random access)
        with open(raw_file) as f:
            raw_lines = f.readlines()

        passes = 0
        for _ in range(num_samples):
            i = random.randint(0, min(dataset_len, len(raw_lines)) - 1)
            try:
                raw_text = json.loads(raw_lines[i])[text_key]
            except (json.JSONDecodeError, KeyError):
                continue
            decoded = tokenizer.decode(ds[i], skip_special_tokens=True)
            if raw_text[:500] == decoded[:500]:
                passes += 1
            else:
                print(f"    MISMATCH at index {i}")
                print(f"      raw:     {raw_text[:200]!r}")
                print(f"      decoded: {decoded[:200]!r}")
        print(f"  Sample check: {passes}/{num_samples} passed")
    else:
        print(f"\n  Skipping sample decode check (multiple files — mapping not 1:1 by line index)")

    print("\nPartition verification complete.")


# ---------------------------------------------------------------------------
# Merge verification
# ---------------------------------------------------------------------------

def verify_merge(converted_dir, merged_prefix, tokenizer_model, num_samples):
    """Verify merged output matches the sum of converted/ shards."""
    print(f"\n{'='*60}")
    print(f"MERGE VERIFICATION")
    print(f"  converted_dir:  {converted_dir}")
    print(f"  merged_prefix:  {merged_prefix}")
    print(f"{'='*60}\n")

    prefixes = find_converted_prefixes(converted_dir)
    print(f"Partition stats:")
    sum_docs = 0
    sum_tokens = 0
    prefix_doc_counts = []
    for p in prefixes:
        full_prefix = os.path.join(converted_dir, p)
        n_docs, n_tokens = get_stats(full_prefix)
        print(f"  {p}: {n_docs:,} docs, {n_tokens:,} tokens")
        sum_docs += n_docs
        sum_tokens += n_tokens
        prefix_doc_counts.append((p, n_docs))

    merged_docs, merged_tokens = get_stats(merged_prefix)
    print(f"\nMerged stats:")
    print(f"  {os.path.basename(merged_prefix)}: {merged_docs:,} docs, {merged_tokens:,} tokens")

    print(f"\nPartitions sum: {sum_docs:,} docs, {sum_tokens:,} tokens")

    if sum_docs == merged_docs and sum_tokens == merged_tokens:
        print("PASS: doc and token counts tally")
    else:
        print(f"FAIL: mismatch — docs {sum_docs} vs {merged_docs}, tokens {sum_tokens} vs {merged_tokens}")
        return

    # Random index spot-check
    print(f"\nRandom index check ({num_samples} samples)...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    merged_ds = IndexedDataset(merged_prefix)
    passes = 0

    # Build cumulative doc count for partition lookup
    cumulative = []
    running = 0
    for p, n_docs in prefix_doc_counts:
        cumulative.append((p, running, running + n_docs))
        running += n_docs

    for _ in range(num_samples):
        global_idx = random.randint(0, merged_docs - 1)
        # Find which partition this index belongs to
        for p, start, end in cumulative:
            if start <= global_idx < end:
                local_idx = global_idx - start
                part_prefix = os.path.join(converted_dir, p)
                part_ds = IndexedDataset(part_prefix)
                merged_len = len(merged_ds[global_idx])
                part_len = len(part_ds[local_idx])
                if merged_len == part_len:
                    passes += 1
                else:
                    print(f"  MISMATCH at global_idx={global_idx}: "
                          f"merged len={merged_len}, partition len={part_len}")
                break

    print(f"Index check: {passes}/{num_samples} passed")
    print("\nMerge verification complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify Megatron preprocessed data")
    parser.add_argument("--mode", choices=["partition", "merge", "all"], default="all",
                        help="Verification mode")
    parser.add_argument("--input-dir", type=str,
                        help="Directory with raw JSONL files (for partition mode)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Dataset output dir containing converted/ and merged/ subdirs")
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="Dataset name (used to locate merged prefix)")
    parser.add_argument("--text-key", type=str, default="text",
                        help="JSON key for text content")
    parser.add_argument("--tokenizer-model", type=str, default="Qwen/Qwen3-4B",
                        help="HuggingFace tokenizer model for decode checks")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Number of random samples to check")
    args = parser.parse_args()

    converted_dir = os.path.join(args.output_dir, "converted")
    merged_prefix = os.path.join(args.output_dir, "merged", f"{args.dataset_name}_merged")

    if args.mode in ("partition", "all"):
        assert args.input_dir, "--input-dir is required for partition verification"
        verify_partition(
            input_dir=args.input_dir,
            converted_dir=converted_dir,
            dataset_name=args.dataset_name,
            text_key=args.text_key,
            tokenizer_model=args.tokenizer_model,
            num_samples=args.num_samples,
        )

    if args.mode in ("merge", "all"):
        verify_merge(
            converted_dir=converted_dir,
            merged_prefix=merged_prefix,
            tokenizer_model=args.tokenizer_model,
            num_samples=args.num_samples,
        )


if __name__ == "__main__":
    main()
