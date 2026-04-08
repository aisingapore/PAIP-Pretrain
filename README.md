# Syllabus Outline

## Week 1: Data for PreTraining
day 1: data pipeline in general, (EN & ID)
- pdf extraction
- dedup
- filtering
- language detection
- quality classification
- toxicity removal
- PII protection
- data generation

day 2a: rule-based and model-based quality classification
- rule-used in SeaPile (language specific)
- ken-LM based classification (training and inference)
- FineWebEdu classification with SOTA opensource models (toy data)

day 2b: data preprocessing for Pretraining Framework, 
ref: ARF-Training/data_prep/megatron
- working principle of megatron dataset
    - variable sequence length handling
    - sequence packing
    - flexibility to draw at different sequence length
    - efficienty spinning
- the data conversion pipeline
    - resource utilization during the data preprocess
        - CPU
        - RAM
        - Disk I/O
        - Disk capacity
    - pain-points and design requirements
        - 200+ dataset
        - organizing them hierarchically by language, domain, subset, quality, etc.
        - effective token counts 
        - token budgetting and datamix
    - our data conversion pipeline
        - generate data_config.yaml from google sheet
        - pull raw data from S3 based on data_config
        - convert (tokenize and binarize), multiprocessing vs pyspark
        - upload the converted to S3
        - cleanup intermediate conversion files
        - integrity check of dataset at each step (pull/convert/upload/final verification)

day 3: dataloading and datamix, 
ref: 
    ARF-Training/train/configs/data_config.yaml, 
    ARF-Training/train/scripts/common/datamix
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

day 4-5 (optional): readup on SOTA model training stuff
- tokenization of non-latin scripts and challenges
- forgetting mitigation
    - replay (part of datamix)
    - LoRA
    - model merging
- context extension
    - RoPE
    - YaRN
    - Context-Parallel (hyperlink)

## Week 2: Model Training, ref: ARF-Training/train/scripts/smc_megatron_bridge

day 1: logs, checkpointing, resuming
- log file structure and intention
- artifacts saving
- W&B logging
- (exercise) custom W&B metric
- checkpoint working principle (code walkthrough)
    - anatomy of checkpoint (data_load, parameter, optimizer state)
    - hf to megatron import
    - sharding the model according to sharding strategy
    - save the checkpoints
    - export the checkpoints to hf

day 2: training dynamics, etc.
- optimizer
- scheduler, implementation of custom scheduler in MegatronBridge
- hyperparameters (GBS, lr, weight decay, epsilon, layer-specific)

day 3: parallelism, memory management, throughput
- 

day 4-5 (optional): readup on SOTA model training stuff
- Attention Mechanism & Code Implementation (Karparthy or LLM from scratch)
- KV-cache & Code Implementation
- hyperparameter transfer