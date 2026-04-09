"""Count documents and tokens in a Megatron IndexedDataset."""
import argparse

import numpy as np
from megatron.core.datasets.indexed_dataset import IndexedDataset


def count_tokens(megatron_prefix):
    """
    Returns (num_docs, num_tokens) for the IndexedDataset at megatron_prefix.
    megatron_prefix should be the path without .bin/.idx extension.
    """
    dataset = IndexedDataset(megatron_prefix)
    num_docs = dataset.sequence_lengths.shape[0]
    num_tokens = int(np.sum(dataset.sequence_lengths))
    return num_docs, num_tokens


def main():
    parser = argparse.ArgumentParser(description="Count tokens in a Megatron IndexedDataset")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path prefix to .bin/.idx files (without extension)",
    )
    args = parser.parse_args()

    num_docs, num_tokens = count_tokens(args.input)
    print(f"Dataset:    {args.input}")
    print(f"Documents:  {num_docs:,}")
    print(f"Tokens:     {num_tokens:,}")


if __name__ == "__main__":
    main()
