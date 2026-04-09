"""
Data preparation pipeline for Megatron pretraining.

Sub-commands:
  tokenize  — convert raw JSONL to Megatron .bin/.idx format (via PySpark)
  merge     — merge per-file shards into a single dataset
  verify    — verify converted and merged data
  count     — count documents and tokens
  all       — run tokenize → merge → verify → count

Usage example:
  python run_pipeline.py all \\
    --input-dir /path/to/raw/text \\
    --output-dir /path/to/output/EN_Wikibooks \\
    --dataset-name EN_Wikibooks \\
    --text-key raw_content \\
    --tokenizer-model Qwen/Qwen3-4B \\
    --workers 20
"""
import argparse
import glob
import logging
import os
import subprocess
import sys

from utils import check_text_key, input_jsonl_group_by_size, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "config.yaml")
DEFAULT_MEGATRON_LM_DIR = "/mnt/weka/aisg/source_files/megatron_yuli"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pipeline_config(config_path, args):
    """Merge YAML config defaults with CLI args (CLI takes precedence)."""
    cfg = load_config(config_path)

    # Overlay CLI args that were explicitly set
    if args.tokenizer_model:
        cfg["tokenizer_model"] = args.tokenizer_model
    if args.tokenizer_type:
        cfg["tokenizer_type"] = args.tokenizer_type
    if args.workers:
        cfg["workers"] = args.workers
    if getattr(args, "backend", None):
        cfg["preprocess_backend"] = args.backend

    return cfg


def find_input_files(input_dir, extension="jsonl"):
    files = sorted(glob.glob(os.path.join(input_dir, f"*.{extension}")))
    if not files:
        raise FileNotFoundError(f"No .{extension} files found in {input_dir}")
    return files


def run_subprocess(command, label="subprocess"):
    """Run a command, streaming its output to the logger."""
    logger.info(f"[{label}] Running: {' '.join(command)}")
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    for line in process.stdout:
        logger.info(f"[{label}] {line.rstrip()}")
    rc = process.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, command)


def count_bin_idx_pairs(directory):
    """Return the number of .bin/.idx pairs in directory."""
    bins = glob.glob(os.path.join(directory, "*.bin"))
    idxs = glob.glob(os.path.join(directory, "*.idx"))
    return min(len(bins), len(idxs))


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_tokenize(args, cfg):
    logger.info("=" * 60)
    logger.info("STEP: tokenize")

    input_dir = args.input_dir
    output_dir = args.output_dir
    dataset_name = args.dataset_name
    text_key = args.text_key
    extension = cfg.get("data_file_extension", "jsonl")
    skip_processed = cfg.get("skip_processed", True)

    converted_dir = os.path.join(output_dir, "converted")
    os.makedirs(converted_dir, exist_ok=True)

    # Validate text key
    check_text_key(input_dir, text_key)

    # Discover and group input files
    input_files = find_input_files(input_dir, extension)
    logger.info(f"Found {len(input_files)} input file(s)")
    file_groups = input_jsonl_group_by_size(
        input_files,
        target_size_gb=cfg.get("group_target_size_gb", 10),
        mean_threshold_gb=cfg.get("group_mean_threshold_gb", 5),
    )
    logger.info(f"Processing {len(file_groups)} file group(s)")

    backend = cfg.get("preprocess_backend", "mp")
    if backend == "spark":
        preprocess_script = os.path.join(SCRIPT_DIR, "preprocess_data_spark.py")
        logger.info(f"Backend: PySpark (preprocess_data_spark.py)")
    else:
        megatron_lm_dir = cfg.get("megatron_lm_dir", DEFAULT_MEGATRON_LM_DIR)
        preprocess_script = os.path.join(megatron_lm_dir, "tools", "preprocess_data.py")
        logger.info(f"Backend: multiprocessing (megatron_yuli/tools/preprocess_data.py)")

    for i, file_group in enumerate(file_groups, 1):
        input_basename = os.path.splitext(os.path.basename(file_group[0]))[0]
        output_prefix = os.path.join(converted_dir, f"{dataset_name}_{input_basename}")

        # Skip if output already exists
        expected_bin = f"{output_prefix}_{text_key}_document.bin"
        expected_idx = f"{output_prefix}_{text_key}_document.idx"
        if skip_processed and os.path.exists(expected_bin) and os.path.exists(expected_idx):
            logger.info(f"  Group {i}/{len(file_groups)}: skipping (output already exists)")
            continue

        logger.info(f"  Group {i}/{len(file_groups)}: {[os.path.basename(f) for f in file_group]}")

        if backend == "spark":
            command = [
                sys.executable,
                preprocess_script,
                "--input", *file_group,
                "--json-keys", text_key,
                "--tokenizer-type", cfg["tokenizer_type"],
                "--tokenizer-model", cfg["tokenizer_model"],
                "--output-prefix", output_prefix,
                "--log-interval", str(cfg.get("log_interval", 100000)),
                "--workers", str(cfg["workers"]),
                "--append-eod",
                "--keep-sequential-samples",
            ]
        else:
            # mp backend: megatron_yuli preprocess_data.py takes a single --input file
            assert len(file_group) == 1, (
                "mp backend processes one file at a time. "
                "Use spark backend for grouped multi-file inputs."
            )
            command = [
                sys.executable,
                preprocess_script,
                "--input", file_group[0],
                "--json-keys", text_key,
                "--tokenizer-type", cfg["tokenizer_type"],
                "--tokenizer-model", cfg["tokenizer_model"],
                "--output-prefix", output_prefix,
                "--log-interval", str(cfg.get("log_interval", 100000)),
                "--workers", str(cfg["workers"]),
                "--append-eod",
            ]

        run_subprocess(command, label=f"tokenize/{input_basename}")

    logger.info("Tokenization complete.")


def step_merge(args, cfg):
    logger.info("=" * 60)
    logger.info("STEP: merge")

    output_dir = args.output_dir
    dataset_name = args.dataset_name
    converted_dir = os.path.join(output_dir, "converted")
    merged_dir = os.path.join(output_dir, "merged")
    merged_prefix = os.path.join(merged_dir, f"{dataset_name}_merged")

    num_pairs = count_bin_idx_pairs(converted_dir)
    if num_pairs == 0:
        raise FileNotFoundError(f"No .bin/.idx pairs found in {converted_dir}")

    if num_pairs == 1:
        logger.info(f"Only 1 shard in converted/ — copying instead of merging")
        os.makedirs(merged_dir, exist_ok=True)
        # Find the single pair and copy it
        bins = glob.glob(os.path.join(converted_dir, "*.bin"))
        idxs = glob.glob(os.path.join(converted_dir, "*.idx"))
        assert len(bins) == 1 and len(idxs) == 1
        import shutil
        shutil.copy2(bins[0], merged_prefix + ".bin")
        shutil.copy2(idxs[0], merged_prefix + ".idx")
        logger.info(f"Copied to {merged_prefix}{{.bin,.idx}}")
        return

    os.makedirs(merged_dir, exist_ok=True)
    merge_script = os.path.join(SCRIPT_DIR, "merge_datasets.py")
    command = [
        sys.executable,
        merge_script,
        "--input", converted_dir,
        "--output-prefix", merged_prefix,
    ]
    run_subprocess(command, label="merge")
    logger.info(f"Merge complete → {merged_prefix}{{.bin,.idx}}")


def step_verify(args, cfg):
    logger.info("=" * 60)
    logger.info("STEP: verify")

    verify_script = os.path.join(SCRIPT_DIR, "verify_data.py")
    command = [
        sys.executable,
        verify_script,
        "--mode", "all",
        "--input-dir", args.input_dir,
        "--output-dir", args.output_dir,
        "--dataset-name", args.dataset_name,
        "--text-key", args.text_key,
        "--tokenizer-model", cfg["tokenizer_model"],
        "--num-samples", str(cfg.get("verification_num_samples", 20)),
    ]
    run_subprocess(command, label="verify")


def step_count(args, cfg):
    logger.info("=" * 60)
    logger.info("STEP: count")

    merged_prefix = os.path.join(args.output_dir, "merged", f"{args.dataset_name}_merged")
    count_script = os.path.join(SCRIPT_DIR, "count_tokens.py")
    command = [
        sys.executable,
        count_script,
        "--input", merged_prefix,
    ]
    run_subprocess(command, label="count")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_common_args(parser):
    parser.add_argument("--input-dir", type=str,
                        help="Directory containing raw JSONL input files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory (will contain converted/ and merged/ subdirs)")
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="Dataset name used for output file naming")
    parser.add_argument("--text-key", type=str, default="text",
                        help="JSON key for text content (default: text)")
    parser.add_argument("--tokenizer-model", type=str, default=None,
                        help="HuggingFace tokenizer model (overrides config)")
    parser.add_argument("--tokenizer-type", type=str, default=None,
                        help="Megatron tokenizer type (overrides config)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of PySpark workers (overrides config)")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help=f"Path to config YAML (default: {DEFAULT_CONFIG})")
    parser.add_argument("--backend", type=str, choices=["mp", "spark"], default=None,
                        help="Preprocessing backend: 'mp' (multiprocessing, default) or 'spark' (PySpark, requires Java)")


def main():
    parser = argparse.ArgumentParser(
        description="Megatron data preparation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for cmd in ("tokenize", "merge", "verify", "count", "all"):
        sub = subparsers.add_parser(cmd, help=f"Run the {cmd} step")
        add_common_args(sub)

    args = parser.parse_args()
    cfg = load_pipeline_config(args.config, args)

    if args.command == "tokenize":
        step_tokenize(args, cfg)
    elif args.command == "merge":
        step_merge(args, cfg)
    elif args.command == "verify":
        step_verify(args, cfg)
    elif args.command == "count":
        step_count(args, cfg)
    elif args.command == "all":
        step_tokenize(args, cfg)
        step_merge(args, cfg)
        step_verify(args, cfg)
        step_count(args, cfg)

    logger.info("Done.")


if __name__ == "__main__":
    main()
