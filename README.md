# Syllabus Outline

## Week 1: Data for PreTraining

### Day 1: data pipeline in general, (EN & ID)

- pdf extraction pipeline (pdf/img -> text)

- data preprocess pipeline (raw text -> clean and curated dataset)
    - dedup
    - filtering
    - language detection
    - quality classification
    - toxicity removal
    - PII protection
    - data generation

- (tutorial) walkthrough of data preprocess pipeline

- data classification
    - rule-used in SeaPile (language specific)
    - ken-LM based classification (training and inference)
    - FineWebEdu classification with SOTA opensource models (toy data)

### Day 2b: data preprocessing for Pretraining Framework

- working principle of megatron dataset
    - variable sequence length handling
    - sequence packing
    - flexibility to draw at different sequence length
    - efficienty spinning

- the data conversion pipeline
    - optimize resource utilization during the data preprocess
        - Multiprocessing vs pyspark
        - CPU, RAM, Disk space, Disk I/O

    - our data conversion pipeline
        - pull raw data from S3 based on data_config
        - convert (tokenize and binarize), multiprocessing vs pyspark
        - cleanup intermediate conversion files
        - token count and integrity check

- (tutorial) data integrity check after merging shards of converted dataset

### Day 3: dataloading and datamix

- SOTA datamix approaches
    - mitigate catastrophic forgetting
    - balance the representation of different languages
    - ensure domain diversity
    - curriculum learning
    - decaying

- Data blending and sampling in Megatron stack (Walkthrough)
    - how it ensures the sampling follows the specified weights of each dataset
    - how it samples homogeneously from all locations in the entire dataset
    - efficient RAM usage by not loading whole dataset into RAM
    - cyclic loader to enable multi-epoch of individual data
    - how it ensures determinism in the data sampling trajectory, which random seed to set
    - resume from the same deterministic data sampling trajectory in the new train leg.

- Our configs and launch scripts
    - translate handcrafted datamix strategy to Megatron Blended Dataset

- (tutorial) multi-stage training implementation - loaders.py

### Day 4-5: readup on SOTA model training stuff

- tokenization of non-latin scripts and challenges
- forgetting mitigation
    - replay (reference to datamix)
    - model merging
    - LoRA
- context extension
    - ABN & YaRN 
    - long context data composition

## Week 2: Model Training, ref: ARF-Training/train/scripts/smc_megatron_bridge

### Day 1: logs, checkpointing, resuming

- log file structure and intention
- artifacts saving
- W&B logging
- checkpoint working principle (code walkthrough)
    - anatomy of checkpoint (data_load, parameter, optimizer state)
    - hf to megatron import
    - sharding the model according to sharding strategy
    - save the checkpoint
    - load the checkpoint
    - export the checkpoints to hf
- (tutorial) custom W&B metric

### Day 2: training dynamics, etc.

- optimizer
- scheduler, implementation of custom scheduler in MegatronBridge
- hyperparameters (GBS, lr, weight decay, epsilon, layer-specific)
- (tutorial) WSD scheduler implementation

### Day 3: parallelism, memory management, throughput

- efficiency (tutorial integrated)
    - MBS
    - recomputation
    - activation offloading
    - communication/computation overlap
    - torch-profiling

- parallelsim (tutorial integrated)
    - DDP
    - FSDP
    - TP
    - CP
    - EP

### Day 4-5: readup on SOTA model training stuff

- hyperparameter transfer
- Attention Mechanism & Code Implementation
- KV-cache & Code Implementation
