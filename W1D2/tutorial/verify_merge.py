"""
Tutorial: Merge Verification for Megatron-format Preprocessed Data

In this exercise you will implement two verification checks that ensure
the merged dataset faithfully combines all per-shard datasets produced
during tokenization.

See guide.md for detailed instructions and hints.
"""
import argparse
import os
import random

import numpy as np
from megatron.core.datasets.indexed_dataset import IndexedDataset


# ---------------------------------------------------------------------------
# Helpers (provided — use these in your solution)
# ---------------------------------------------------------------------------

def get_stats(prefix):
    """Return (num_docs, num_tokens) for an IndexedDataset at *prefix*.

    Usage:
        n_docs, n_tokens = get_stats("/path/to/dataset_prefix")

    Internally this loads the IndexedDataset and reads its
    ``sequence_lengths`` array, which stores the token count of every
    document.
    """
    ds = IndexedDataset(prefix)
    num_docs = ds.sequence_lengths.shape[0]
    num_tokens = int(np.sum(ds.sequence_lengths))
    return num_docs, num_tokens


def find_converted_prefixes(converted_dir):
    """Return sorted list of unique .bin/.idx prefixes in *converted_dir*.

    Each prefix can be passed to ``IndexedDataset(prefix)`` to load
    that shard.
    """
    prefixes = set()
    for fname in os.listdir(converted_dir):
        if fname.endswith(".bin") or fname.endswith(".idx"):
            prefixes.add(os.path.splitext(fname)[0])
    return sorted(prefixes)


# ---------------------------------------------------------------------------
# Exercise: implement the two verification checks below
# ---------------------------------------------------------------------------

def verify_merge(converted_dir, merged_prefix, num_samples):
    """Verify that the merged dataset matches the sum of converted/ shards.

    This function performs two checks:

    1. **Count tally** — the total number of documents and tokens across
       all per-shard datasets in ``converted_dir`` must equal the counts
       in the merged dataset at ``merged_prefix``.

    2. **Random index spot-check** — for a number of randomly chosen
       document indices, the token-sequence length in the merged dataset
       must equal the length in the originating shard.

    Parameters
    ----------
    converted_dir : str
        Directory containing per-shard ``.bin/.idx`` files.
    merged_prefix : str
        Path prefix of the merged dataset (e.g. ``merged/dataset_merged``).
    num_samples : int
        How many random indices to spot-check in check 2.
    """
    print(f"\n{'='*60}")
    print(f"MERGE VERIFICATION")
    print(f"  converted_dir:  {converted_dir}")
    print(f"  merged_prefix:  {merged_prefix}")
    print(f"{'='*60}\n")

    prefixes = find_converted_prefixes(converted_dir)

    # ===================================================================
    # Exercise 1 — Count Tally
    #
    # Goal: verify that the sum of documents and tokens across all shards
    #       equals the document and token counts in the merged dataset.
    #
    # Steps:
    #   a) Loop over each prefix in ``prefixes``.  For each one, call
    #      ``get_stats(full_prefix)`` (where full_prefix is the absolute
    #      path built from ``converted_dir`` and the prefix name) to get
    #      its (num_docs, num_tokens).  Accumulate the sums.
    #      Also build a list ``prefix_doc_counts`` of (prefix, num_docs)
    #      tuples — you will need this in Exercise 2.
    #
    #   b) Call ``get_stats(merged_prefix)`` to get the merged totals.
    #
    #   c) Compare. Print PASS if both doc and token sums match, or FAIL
    #      with the mismatched numbers.  Return early on failure.
    #
    # Hints:
    #   - get_stats() returns a tuple: (num_docs, num_tokens)
    #   - Build the full prefix path:
    #         full_prefix = os.path.join(converted_dir, p)
    #   - Print per-shard stats for visibility, e.g.:
    #         print(f"  {p}: {n_docs:,} docs, {n_tokens:,} tokens")
    # ===================================================================

    print("Partition stats:")
    sum_docs = 0
    sum_tokens = 0
    prefix_doc_counts = []

    # TODO: implement count tally (steps a-c above)
    pass

    # ===================================================================
    # Exercise 2 — Random Index Spot-Check
    #
    # Goal: for ``num_samples`` randomly chosen global document indices,
    #       verify that the token-sequence length in the merged dataset
    #       matches the length in the originating shard.
    #
    # Steps:
    #   a) Load the merged dataset:
    #          merged_ds = IndexedDataset(merged_prefix)
    #
    #   b) Build a cumulative document-count table from
    #      ``prefix_doc_counts`` so you can map a global index to the
    #      correct shard.  For example, if shard A has 100 docs and
    #      shard B has 200 docs, global indices 0-99 belong to A and
    #      100-299 belong to B.
    #
    #   c) For each of ``num_samples`` iterations:
    #        - Pick a random global index in [0, total_merged_docs).
    #        - Walk the cumulative table to find which shard owns it,
    #          and compute the local index within that shard.
    #        - Load the shard's IndexedDataset and compare:
    #              len(merged_ds[global_idx])  vs  len(part_ds[local_idx])
    #        - Count how many match.
    #
    #   d) Print the result, e.g.:
    #          print(f"Index check: {passes}/{num_samples} passed")
    #
    # Hints:
    #   - ``prefix_doc_counts`` is a list of (prefix_name, num_docs).
    #   - To build the cumulative table you need a running total.
    #   - ``random.randint(a, b)`` is inclusive on both ends.
    #   - Access a document's tokens: ``merged_ds[idx]`` returns a
    #     numpy array of token IDs.
    # ===================================================================

    print(f"\nRandom index check ({num_samples} samples)...")

    # TODO: implement random index spot-check (steps a-d above)
    pass

    print("\nMerge verification complete.")


# ---------------------------------------------------------------------------
# CLI (provided)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tutorial: verify merged Megatron dataset")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Dataset output dir containing converted/ and merged/")
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="Dataset name (used to locate merged prefix)")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Number of random samples to spot-check")
    args = parser.parse_args()

    converted_dir = os.path.join(args.output_dir, "converted")
    merged_prefix = os.path.join(args.output_dir, "merged",
                                 f"{args.dataset_name}_merged")

    verify_merge(
        converted_dir=converted_dir,
        merged_prefix=merged_prefix,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
