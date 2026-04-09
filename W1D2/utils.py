import json
import logging
import os

import yaml

logger = logging.getLogger(__name__)


def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def check_text_key(input_dir, text_key):
    """
    Validates that text_key exists in JSONL files in input_dir.
    Samples the first line of each file and checks for the key.
    Returns the verified text_key, or raises ValueError if not found.
    """
    jsonl_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".jsonl") or f.endswith(".json")
    ]
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {input_dir}")

    for file_path in jsonl_files[:3]:  # check up to 3 files
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if text_key not in data:
                    available = list(data.keys())
                    raise ValueError(
                        f"text_key '{text_key}' not found in {file_path}. "
                        f"Available keys: {available}"
                    )
                logger.info(f"Confirmed text_key '{text_key}' in {os.path.basename(file_path)}")
                break

    return text_key


def input_jsonl_group_by_size(input_files, target_size_gb=10, mean_threshold_gb=5):
    """
    Groups JSONL files into batches for efficient PySpark processing.

    Files with mean size >= mean_threshold_gb are processed individually.
    Otherwise, files are grouped into ~target_size_gb batches.
    Parquet files are always processed individually.

    Returns a nested list where each inner list is one processing group.
    """
    if not input_files:
        return []

    if all(f.lower().endswith(".parquet") for f in input_files):
        logger.info("All inputs are Parquet — processing individually")
        return [[f] for f in input_files]

    total_size = 0
    valid_files = []
    for f in input_files:
        if os.path.exists(f):
            size = os.path.getsize(f)
            total_size += size
            valid_files.append((f, size))
        else:
            logger.warning(f"File not found, skipping: {f}")

    if not valid_files:
        raise ValueError("No valid input files found")

    mean_gb = (total_size / len(valid_files)) / (1024 ** 3)
    logger.info(f"Mean file size: {mean_gb:.2f} GB")

    if mean_gb >= mean_threshold_gb:
        logger.info(f"Large files detected — processing individually")
        return [[f] for f, _ in valid_files]

    # Group into ~target_size_gb batches
    target_bytes = target_size_gb * (1024 ** 3)
    groups = []
    current_group = []
    current_size = 0

    for f, size in valid_files:
        if current_size + size > target_bytes and current_group:
            groups.append([fp for fp, _ in current_group])
            current_group = [(f, size)]
            current_size = size
        else:
            current_group.append((f, size))
            current_size += size

    if current_group:
        groups.append([fp for fp, _ in current_group])

    logger.info(f"Created {len(groups)} file group(s)")
    for i, g in enumerate(groups):
        gb = sum(os.path.getsize(fp) for fp in g) / (1024 ** 3)
        logger.info(f"  Group {i+1}: {len(g)} file(s), {gb:.2f} GB")

    return groups
