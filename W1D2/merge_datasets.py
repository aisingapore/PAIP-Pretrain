"""Merge multiple Megatron IndexedDataset .bin/.idx pairs into a single dataset."""
import argparse
import os

from megatron.core.datasets.indexed_dataset import (
    IndexedDataset,
    IndexedDatasetBuilder,
    get_bin_path,
    get_idx_path,
)


def get_args():
    parser = argparse.ArgumentParser(description="Merge Megatron IndexedDataset files")

    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to directory containing .bin/.idx pairs to merge",
    )

    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Path to output file without suffix (parent dir must exist)",
    )

    group = parser.add_argument_group(title="miscellaneous")
    group.add_argument(
        "--multimodal",
        action="store_true",
        help="Whether the datasets are multimodal",
    )

    args = parser.parse_args()

    assert os.path.isdir(args.input), f"Input directory does not exist: {args.input}"
    assert os.path.isdir(os.path.dirname(args.output_prefix)), (
        f"Output parent directory does not exist: {os.path.dirname(args.output_prefix)}"
    )

    return args


def main():
    args = get_args()

    # Collect unique prefixes (without extension) for all .bin/.idx pairs
    prefixes = set()
    for basename in os.listdir(args.input):
        prefix, ext = os.path.splitext(basename)
        if ext not in (".bin", ".idx"):
            continue
        if prefix in prefixes:
            continue
        if not os.path.isfile(os.path.join(args.input, basename)):
            continue
        ext_pair = ".bin" if ext == ".idx" else ".idx"
        assert os.path.isfile(os.path.join(args.input, prefix) + ext_pair), (
            f"Missing {ext_pair} counterpart for {os.path.join(args.input, prefix)}"
        )
        prefixes.add(prefix)

    prefixes = sorted(prefixes)
    print(f"Found {len(prefixes)} dataset(s) to merge:")
    for p in prefixes:
        print(f"  {p}")

    builder = None
    for prefix in prefixes:
        full_prefix = os.path.join(args.input, prefix)
        if builder is None:
            dataset = IndexedDataset(full_prefix, multimodal=args.multimodal)
            builder = IndexedDatasetBuilder(
                get_bin_path(args.output_prefix),
                dtype=dataset.index.dtype,
                multimodal=args.multimodal,
            )
            del dataset
        print(f"  Adding {prefix} ...")
        builder.add_index(full_prefix)

    if builder is None:
        raise RuntimeError(f"No .bin/.idx pairs found in {args.input}")

    builder.finalize(get_idx_path(args.output_prefix))
    print(f"Merged output written to: {args.output_prefix}{{.bin,.idx}}")


if __name__ == "__main__":
    main()
