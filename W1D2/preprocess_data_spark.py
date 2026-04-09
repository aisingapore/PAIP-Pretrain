# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import logging

# Suppress verbose indexed_dataset logs
logging.getLogger('megatron.core.datasets.indexed_dataset').setLevel(logging.WARNING)

"""Processing large data for pretraining using PySpark."""
import argparse
import json
import os
import sys
from pyspark.sql import SparkSession
import time
import glob
import multiprocessing

import functools
try:
    import nltk
    from nltk.tokenize.punkt import PunktLanguageVars
    nltk_available = True
except ImportError:
    PunktLanguageVars = object
    nltk_available = False

from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.training.arguments import _add_tokenizer_args
from megatron.core.datasets import indexed_dataset


def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"{time.strftime('%H:%M:%S', time.localtime())} {func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper


# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(PunktLanguageVars):
    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""


class IdentitySplitter(object):
    def tokenize(self, *text):
        return text


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = build_tokenizer(self.args)
        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            if os.environ.get("NLTK_DATA"):
                library = os.path.join(os.environ.get("NLTK_DATA"), "tokenizers", "punkt", f"{self.args.lang}.pickle")
                url = f"file:{library}"
            else:
                library = os.path.join("tokenizers", "punkt", f"{self.args.lang}.pickle")
                url = f"nltk:{library}"
            splitter = nltk.load(url)
            if self.args.keep_newlines:
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text=splitter._params,
                    lang_vars=CustomLanguageVars())
            else:
                Encoder.splitter = splitter
        else:
            Encoder.splitter = IdentitySplitter()

    def split(self, json_line):
        data = json.loads(json_line)
        output = {}
        for key in self.args.json_keys:
            text = data[key]
            max_len = 1000000
            tokens_list = [Encoder.splitter.tokenize(text[i:i+max_len]) for i in range(0, len(text), max_len)]
            output[key] = [tokens for partial in tokens_list for tokens in partial]
        return json.dumps(output), len(json_line)

    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {}
        lens = {}
        for key in self.args.json_keys:
            text = data[key]
            if isinstance(text, list):
                sentences = text
            else:
                sentences = [text]
            doc_ids = []
            sentence_lens = []
            for sentence in sentences:
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.extend(sentence_ids)
                    sentence_lens.append(len(sentence_ids))
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids.append(Encoder.tokenizer.eod)
                sentence_lens[-1] += 1
            ids[key] = doc_ids
            lens[key] = sentence_lens
        return ids, lens, len(json_line)


class Partition(object):
    @timing_decorator
    def __init__(self, args, workers):
        self.args = args
        self.workers = workers

    def print_processing_stats(self, count, proc_start, total_bytes_processed):
        if count % self.args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {count} documents ({count/elapsed:.1f} docs/s, {mbs:.1f} MB/s).",
                  file=sys.stderr)

    @timing_decorator
    def split_sentences(self, file_name):
        input_file_name, output_file_name = file_name
        print("Opening", input_file_name)
        fin = open(input_file_name, 'r', encoding='utf-8')
        fout = open(output_file_name, 'w')

        encoder = Encoder(self.args)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        split_docs = pool.imap(encoder.split, fin, 32)

        proc_start = time.time()
        total_bytes_processed = 0
        for i, (doc, bytes_processed) in enumerate(split_docs, start=1):
            total_bytes_processed += bytes_processed
            fout.write(doc + "\n")
            self.print_processing_stats(i, proc_start, total_bytes_processed)

        fin.close()
        fout.close()

    def process_input_file(self, file_name):
        input_file_name, output_prefix = file_name
        print("Opening", input_file_name)

        file_open_start = time.time()

        if hasattr(self.args, 'workers') and self.args.workers is not None:
            print(f"Using {self.args.workers} workers")
            n_workers = self.args.workers
        else:
            print("Using default number of workers")
            n_workers = multiprocessing.cpu_count() - 5

        executor_memory_gb = 32
        per_worker_RAM_mb = min(executor_memory_gb * 1024 // n_workers, 256)

        print(f"n_workers: {n_workers}")
        print(f"pyspark RAM per worker: {per_worker_RAM_mb}m")

        spark = SparkSession.builder \
            .master(f"local[{n_workers}]") \
            .config("spark.driver.memory", f"{executor_memory_gb}g") \
            .config("spark.executor.memory", f"{executor_memory_gb}g") \
            .config("spark.sql.files.maxPartitionBytes", f"{per_worker_RAM_mb}m") \
            .getOrCreate()
        sc = spark.sparkContext

        if input_file_name[0].endswith('.parquet'):
            print(f"Loading parquet file: {input_file_name}")
            assert len(input_file_name) == 1, "Expected a single parquet file"
            df = spark.read.parquet(input_file_name[0])
            from pyspark.sql.functions import monotonically_increasing_id
            df = df.withColumn("__original_order__", monotonically_increasing_id())
            rdd = df.rdd
        elif input_file_name[0].endswith('.jsonl') or input_file_name[0].endswith('.json'):
            print(f"Loading {len(input_file_name)} jsonl file(s) into DataFrame with explicit ordering")

            from pyspark.sql.functions import monotonically_increasing_id, lit

            ordered_files = [(i, fname) for i, fname in enumerate(input_file_name)]
            print(f"Processing files in order: {[f[1] for f in ordered_files]}")

            dfs = []
            for file_idx, file_path in ordered_files:
                print(f"Reading file {file_idx}: {file_path}")
                file_df = spark.read.option("multiline", "false").json([file_path])
                file_df = file_df.withColumn("__file_index__", lit(file_idx))
                file_df = file_df.withColumn("__line_number__", monotonically_increasing_id())
                dfs.append(file_df)

            if len(dfs) == 1:
                df = dfs[0]
            else:
                df = dfs[0]
                for df_part in dfs[1:]:
                    df = df.unionByName(df_part, allowMissingColumns=True)

            df = df.withColumn("__global_order__", monotonically_increasing_id())
            print(f"DataFrame created with ordering metadata")
            rdd = df.rdd
        else:
            raise ValueError(f"Unsupported input file type: {input_file_name}. Expected .jsonl, .json, or .parquet")

        file_open_end = time.time()
        print(f"{time.strftime('%H:%M:%S', time.localtime())} Opening file took {file_open_end - file_open_start:.2f}s")

        startup_start = time.time()
        encoder_dummy = Encoder(self.args)
        tokenizer = build_tokenizer(self.args)

        level = "document"
        if self.args.split_sentences:
            level = "sentence"

        def process_and_write_partition(index, iterator):
            local_encoder = Encoder(self.args)
            local_encoder.initializer()

            local_builders = {}
            for key in self.args.json_keys:
                bin_file = "{}_{}_{}_{}.bin".format(output_prefix, key, level, index)
                os.makedirs(os.path.dirname(bin_file), exist_ok=True)
                local_builders[key] = indexed_dataset.IndexedDatasetBuilder(
                    bin_file,
                    dtype=indexed_dataset.DType.optimal_dtype(build_tokenizer(self.args).vocab_size)
                )

            for row in iterator:
                row_dict = row.asDict()
                json_line = json.dumps(row_dict)

                try:
                    doc, sentence_lens, bytes_processed = local_encoder.encode(json_line)
                    for key in doc.keys():
                        local_builders[key].add_document(doc[key], sentence_lens[key])
                except Exception as e:
                    print(f"Encoding failed, trying chunking: {str(e)}")
                    try:
                        data = json.loads(json_line)
                        chunk_size = 300000

                        for key in self.args.json_keys:
                            if key in data and data[key]:
                                text = data[key]
                                if isinstance(text, list):
                                    text = ' '.join(str(s) for s in text if s)

                                chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
                                for chunk in chunks:
                                    if chunk.strip():
                                        chunk_data = {key: chunk}
                                        chunk_json = json.dumps(chunk_data)
                                        chunk_doc, chunk_sentence_lens, _ = local_encoder.encode(chunk_json)
                                        for doc_key in chunk_doc.keys():
                                            local_builders[doc_key].add_document(chunk_doc[doc_key], chunk_sentence_lens[doc_key])
                    except Exception as chunk_error:
                        print(f"Chunking also failed, skipping: {str(chunk_error)}")
                        continue

            for key in local_builders:
                idx_file = "{}_{}_{}_{}.idx".format(output_prefix, key, level, index)
                local_builders[key].finalize(idx_file)

            return iter([])

        rdd.mapPartitionsWithIndex(process_and_write_partition).count()

        startup_end = time.time()
        print(f"{time.strftime('%H:%M:%S', time.localtime())} Partition processing took {startup_end - file_open_end:.2f}s")

        # Phase 2: Merge per-partition shards into one final dataset per key
        merge_start_time = time.time()
        for key in self.args.json_keys:
            output_full_prefix = "{}_{}_{}".format(output_prefix, key, level)
            output_bin_file = "{}.bin".format(output_full_prefix)
            output_idx_file = "{}.idx".format(output_full_prefix)

            builder = indexed_dataset.IndexedDatasetBuilder(
                output_bin_file,
                dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size)
            )
            partitions = sorted(
                glob.glob(f"{output_prefix}_{key}_{level}_*.idx"),
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            print(f"{time.strftime('%H:%M:%S', time.localtime())} Found {len(partitions)} shard(s) for {output_full_prefix}")
            for partition in partitions:
                partition_name = os.path.splitext(partition)[0]
                builder.add_index(partition_name)
            builder.finalize(output_idx_file)

        print(f"{time.strftime('%H:%M:%S', time.localtime())} Merging shards completed in {time.time() - merge_start_time:.2f}s")

        # Phase 3: Clean up intermediate partition files
        cleanup_start = time.time()
        deleted_count = 0
        for idx_file in glob.glob(f"{output_prefix}_*_{level}_*.idx"):
            os.remove(idx_file)
            deleted_count += 1
        for bin_file in glob.glob(f"{output_prefix}_*_{level}_*.bin"):
            os.remove(bin_file)
            deleted_count += 1
        print(f"{time.strftime('%H:%M:%S', time.localtime())} Deleted {deleted_count} intermediate files in {time.time() - cleanup_start:.2f}s")


def get_args():
    parser = argparse.ArgumentParser()
    parser = _add_tokenizer_args(parser)
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', nargs='+', type=str, required=True,
                       help='One or more input files (jsonl/parquet)')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='Space-separated list of keys to extract from JSON')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')
    group = parser.add_argument_group(title='tokenization process')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of each document.')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK sentence splitting.')
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=None, required=False,
                       help='Number of worker processes to launch.')
    group.add_argument('--partitions', type=int, default=1,
                       help='Number of file partitions')
    group.add_argument('--log-interval', type=int, default=1000,
                       help='Interval between progress updates')
    group.add_argument('--keep-sequential-samples', action='store_true',
                       help='Preserve ordering of samples in .jsonl files when using partitions>1.')
    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert') and not args.split_sentences:
        print("Are you sure you don't want to split sentences?")

    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


@timing_decorator
def check_files_exist(in_ss_out_names, key, num_partitions):
    for i in range(num_partitions):
        if not os.path.exists(in_ss_out_names[i][key]):
            return False
    return True


def main():
    args = get_args()

    if args.split_sentences:
        if nltk_available:
            nltk.download("punkt", quiet=True, download_dir=os.environ.get("NLTK_DATA"))
        else:
            raise Exception("nltk library required for sentence splitting is not available.")

    in_ss_out_names = []

    print(f"{time.strftime('%H:%M:%S', time.localtime())} Sentence splitting is {'enabled' if args.split_sentences else 'disabled'}")

    representative_file_name, extension = os.path.splitext(args.input[0])
    sentence_split_file = representative_file_name + "_ss" + extension
    in_ss_out_name = {
        'partition': args.input,
        'sentence_split': sentence_split_file,
        'output_prefix': args.output_prefix
    }
    in_ss_out_names.append(in_ss_out_name)

    partition = Partition(args, args.workers)

    split_sentences_present = check_files_exist(in_ss_out_names, 'sentence_split', args.partitions)

    if args.split_sentences and not split_sentences_present:
        for in_ss_out_name in in_ss_out_names:
            partition.split_sentences((in_ss_out_name['partition'], in_ss_out_name['sentence_split']))
        return

    input_key = 'sentence_split' if args.split_sentences else 'partition'

    process_json_start = time.time()
    for in_ss_out_name in in_ss_out_names:
        partition.process_input_file((in_ss_out_name[input_key], in_ss_out_name['output_prefix']))

    print(f"{time.strftime('%H:%M:%S', time.localtime())} Total processing took {time.time() - process_json_start:.2f}s")


if __name__ == '__main__':
    main()
