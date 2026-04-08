# Checkpoint Workflows in MegatronBridge

This document explains the logic and design rationale behind each checkpoint workflow in MegatronBridge. For full API references, config tables, and CLI commands, see the copied reference docs in `W2D1/ref/`.

---

## Terminology Glossary

| Term | What it means |
|------|---------------|
| **AutoBridge** | The single entry point for all HF-Megatron conversions. It auto-detects the HuggingFace model architecture (Llama, Qwen, Gemma, ...) and dispatches to the correct converter. |
| **Bridge** | The architecture-specific converter (e.g. `LlamaBridge`). AutoBridge selects the right one for you. |
| **Provider** (`GPTModelProvider`) | A factory that holds model config + parallelism settings and can produce a distributed Megatron model on demand. Think of it as a "recipe card" — you tweak the ingredients (TP, PP, fusions) before baking (creating the model). |
| **`finalize()`** | Triggers deferred initialization on a Provider. Megatron Core computes many derived fields (padded vocab size, pipeline dtypes, etc.) in `__post_init__`. `finalize()` delays that computation so you can modify TP/PP/other fields *after* construction but *before* derived fields are locked in. |
| **`distcp`** | Short for "distributed checkpoint". The file extension (`.distcp`) used by MCore's `torch_dist` format. Each file holds one rank's shard of the model/optimizer state. |
| **`torch_dist`** | MCore's distributed checkpoint format. Each rank writes its own `.distcp` shard in parallel. Supports async save, fully parallel save/load, and resharding across different TP/PP/DP configurations. |
| **Resharding** | Automatically redistributing checkpoint shards when the parallelism layout changes between save and load (e.g. saved with TP=4, resumed with TP=8). No manual step needed — MCore handles it during `load()`. |
| **RNG state** | The random number generator state for Python, NumPy, CPU torch, CUDA torch, and Megatron's custom RNG tracker. Saving and restoring this ensures bitwise-reproducible training across restarts. |
| **`async_save`** | Non-blocking checkpoint writes. Training continues while data is flushed to disk in the background. Only supported with `torch_dist` format. |
| **`common.pt`** | An unsharded file written by rank 0 in `torch_dist` checkpoints. Contains metadata that is the same across all ranks (e.g. config hashes, format markers). |
| **`train_state.pt`** | Per-iteration metadata: current step, consumed samples/tokens, accumulated FLOPs, LR scheduler state. Used by the training loop to know where it left off. |
| **`run_config.yaml`** | A full snapshot of the `ConfigContainer` (model, optimizer, data, checkpoint settings). Stored in every checkpoint so the exact training setup can be reproduced or compared at load time. |
| **`latest_train_state.pt`** | A top-level tracker file pointing to the most recent checkpoint iteration. The load path reads this to find which `iter_N/` directory to load. |
| **SafeTensors** | HuggingFace's tensor serialization format (`.safetensors`). Used when exporting Megatron models back to HF format. Supports memory-mapped, streaming reads. |

---

## Workflow 1: Import HuggingFace checkpoint → Megatron

### Why this step exists

HuggingFace models store weights in a single monolithic format (or a few large shards). Megatron needs weights distributed across a parallelism grid (TP/PP/DP) in its own `distcp` format. The import converts the weight layout and wraps them in Megatron's model abstractions.

### What happens behind the scenes

```
import_ckpt("meta-llama/Llama-3.2-1B", "./megatron_ckpt")
│
├─ from_hf_pretrained("meta-llama/Llama-3.2-1B")
│   ├─ Thread-safe config load (avoids race conditions across ranks)
│   ├─ Validate architecture (must end with ForCausalLM or similar)
│   └─ Load HF weights into a PreTrainedCausalLM wrapper
│
├─ to_megatron_provider(load_weights=True)
│   ├─ Detect architecture → select bridge (e.g. LlamaBridge)
│   ├─ Create GPTModelProvider with HF config mapped to Megatron config
│   └─ Register a pre-wrap hook: "after model is created but before DDP
│      wrapping, call load_weights_hf_to_megatron() to copy HF weights
│      into the Megatron model parameter-by-parameter"
│
├─ provider.finalize()
│   └─ Now compute derived fields (padded vocab, pipeline dtype, etc.)
│
├─ provider.provide_distributed_model(wrap_with_ddp=False)
│   ├─ Instantiate MCoreGPTModel with the configured TransformerConfig
│   └─ Pre-wrap hook fires → weights are copied from HF → Megatron
│
└─ save_megatron_model(low_memory_save=True)
    ├─ Generate sharded state dict from the Megatron model
    ├─ Write distcp shards to disk
    └─ Delete model immediately after save to halve peak memory
```

### Key design insight

The **Provider pattern** decouples "what model architecture" from "how to distribute it". You get a Provider from the Bridge, tweak parallelism knobs (TP, PP, fusions), call `finalize()`, then let it build the distributed model. This means the same Bridge code works whether you're running on 1 GPU or 128.

The **pre-wrap hook** is clever: weights must be loaded *after* the model exists but *before* DDP wrapping (which would complicate direct parameter access). Hooks let the Provider orchestrate this timing.

> Full API reference, CLI commands, and code examples: [`ref/megatronbridge-bridge-guide.md`](ref/megatronbridge-bridge-guide.md)

---

## Workflow 2: Distributed Checkpointing and Resharding

### Why not just save everything on one GPU?

A 70B-parameter model with Adam optimizer states takes ~1 TB of memory. Gathering all of that to a single rank for saving would OOM. Instead, each rank writes only its own shard to disk in parallel. This is both faster (parallel I/O) and uses no extra memory beyond what each rank already holds.

### How `distcp` shards work

Each rank in the parallelism grid (TP x PP x DP) holds a slice of the model. During save:

1. Each rank calls `model.sharded_state_dict()` — this returns the rank's local tensors annotated with their *logical* position in the full model (e.g. "I hold columns 0-2047 of layer 5's QKV weight, out of 8192 total columns").
2. `dist_checkpointing.save()` writes each rank's tensors to separate `.distcp` files.
3. A `metadata.json` file records the mapping: logical parameter name → which `.distcp` file → byte offset.

### How resharding works on load

When you resume with a *different* TP/PP layout (e.g. saved TP=4, loading TP=8):

1. The new run calls `generate_state_dict()` which describes what each rank in the *new* grid needs.
2. `dist_checkpointing.load()` reads the old shard metadata, figures out which old shards overlap with each new rank's needs, reads the relevant pieces, and repartitions them.
3. This is fully automatic — no separate resharding script needed.

### Limitation: optimizer resharding

Optimizer states (Adam momentum, variance) can be sharded in two ways:
- **`dp_zero_gather_scatter`** (default): shards only across DP ranks. If TP/PP changes, the optimizer can't be resharded — you'll get an error, and training restarts with fresh optimizer state.
- **`fully_sharded_model_space`** (opt-in via `dist_ckpt_optim_fully_reshardable: true`): shards across all dimensions. This allows full TP/PP/EP/DP resharding but uses more communication during save/load.

> Config parameters and options: [`ref/megatronbridge-checkpointing.md`](ref/megatronbridge-checkpointing.md)

---

## Workflow 3: Saving Checkpoints During Training

### Why periodic saves

Training runs can die from hardware failures, job time limits, or NaN loss spikes. Saving every N steps bounds the maximum amount of lost work to N steps. The trade-off: saving is expensive (disk I/O stalls training), so you pick an interval that balances risk vs. throughput.

### What happens during a save

```
save_checkpoint(iteration, model, optimizer, scheduler)
│
├─ 1. Collect RNG state
│   ├─ Gather: Python random, NumPy, CPU torch, CUDA torch, Megatron tracker
│   └─ Shard decision:
│       ├─ With Expert Parallelism (EP > 1):
│       │   Shard by (PP, TP, DP) — each EP rank initializes different
│       │   experts with different random seeds, so DP ranks differ
│       └─ Without EP:
│           Shard by (PP, TP), DP is replica — all DP ranks compute
│           identical forward passes, so they share the same RNG
│
├─ 2. Generate state dict
│   ├─ model.sharded_state_dict()     → weight shards with logical positions
│   ├─ optimizer.sharded_state_dict() → momentum/variance shards
│   ├─ RNG state (sharded as above)
│   ├─ LR scheduler state
│   └─ Dataloader state (per-DP-rank iteration position)
│
├─ 3. Write to disk
│   ├─ dist_checkpointing.save() → each rank writes its .distcp files
│   ├─ Rank 0 writes: common.pt, metadata.json, run_config.yaml, train_state.pt
│   └─ Tokenizer files copied into checkpoint (self-contained, portable)
│
└─ 4. If async_save=True:
    ├─ save() returns an AsyncRequest instead of blocking
    ├─ Training continues immediately
    └─ Finalized later via maybe_finalize_async_save()
        (called at next save interval or at training end)
```

### What each file in a checkpoint IS

```
checkpoint_dir/
├── latest_train_state.pt       ← "The latest checkpoint is iter_5000"
└── iter_0005000/
    ├── __0_0.distcp, ...       ← Actual weight/optimizer tensor shards (one per rank)
    ├── .metadata               ← PyTorch DCP: which shard holds which tensor slice
    ├── metadata.json           ← MCore: logical param → shard mapping
    ├── common.pt               ← Unsharded metadata from rank 0 (format markers, etc.)
    ├── run_config.yaml         ← Full config snapshot (so you can reproduce or diff)
    ├── train_state.pt          ← Step count, consumed samples, FLOPs (where training was)
    ├── tokenizer/              ← Tokenizer files (no external dependency needed)
    └── dataloader_state/       ← Per-DP-rank data iterator position
        ├── train_dataloader_dprank000.pt
        └── ...
```

### Why RNG sharding differs with Expert Parallelism

Without EP, all DP replicas run the same forward pass on different data — their model-side RNG must be identical (e.g. for dropout masks). With EP, different ranks hold different experts that were initialized with different random seeds, so each (DP, EP) combination has unique RNG that must be preserved independently.

### Why async save matters

A synchronous save of a large checkpoint can stall training for minutes. Async save (`async_save: true`) hands the data off to a background thread and returns immediately. The training loop continues computing the next steps while I/O happens in parallel. The only constraint: you must finalize the previous async save before starting a new one (to avoid write conflicts).

> All save config parameters and options: [`ref/megatronbridge-checkpointing.md`](ref/megatronbridge-checkpointing.md)

---

## Workflow 4: Loading a Checkpoint on Resume

### Why every piece of state matters

Resuming isn't just loading weights. If you only restore weights but not the optimizer, Adam's momentum/variance estimates reset to zero — the model effectively "forgets" the optimization trajectory and loss spikes. If you don't restore the dataloader position, you re-train on data the model already saw. If you don't restore RNG state, dropout masks differ and training diverges from where it would have been.

### What happens during a load

```
load_checkpoint(model, optimizer, scheduler)
│
├─ 1. Find the checkpoint
│   ├─ Read latest_train_state.pt → get iteration number
│   ├─ Or use ckpt_step=N to load a specific iteration
│   └─ If not found and exit_on_missing_checkpoint=True → fail immediately
│      (prevents silently training from scratch when you meant to resume)
│
├─ 2. Read metadata (rank 0 only, then broadcast)
│   ├─ run_config.yaml → extract saved TP, PP sizes
│   ├─ Broadcast to all ranks (avoids filesystem contention at scale —
│   │   100+ ranks hitting the same file simultaneously would be slow)
│   └─ Compare saved TP/PP vs current TP/PP
│
├─ 3. Decide what to restore
│   ├─ RNG: load only if TP/PP match AND not finetuning AND load_rng=True
│   │   (mismatched TP/PP means different ranks hold different layers,
│   │    so old RNG states don't map to the right ranks anymore)
│   ├─ Optimizer: load only if TP/PP match AND not finetuning AND load_optim=True
│   │   AND optimizer sharding is compatible (fully_reshardable if TP/PP changed)
│   └─ Weights: always loaded (this is the whole point)
│
├─ 4. Build target scaffold
│   └─ generate_state_dict() with current model/optimizer → describes what
│      each rank in the NEW grid expects to receive
│
└─ 5. Load and redistribute
    └─ dist_checkpointing.load() reads old shards, matches them to the
       target scaffold, reshards if TP/PP changed, and populates each
       rank's model/optimizer tensors
```

### `pretrained_checkpoint` vs `load`

These serve different purposes:

- **`load`**: Resume the full training state (weights + optimizer + RNG + step count + dataloader position). Used for crash recovery or continuing a run.
- **`pretrained_checkpoint`**: Load only the frozen base weights for fine-tuning. Optimizer starts fresh, step count resets, dataloader starts from the beginning. Used for fine-tuning or PEFT.

Both can be set simultaneously for PEFT: `pretrained_checkpoint` provides the frozen base, `load` resumes the adapter checkpoint.

### `dist_ckpt_strictness`: handling key mismatches

When loading a checkpoint, the saved keys may not exactly match what the current model expects (e.g. different software version added a new buffer). The strictness parameter controls how to handle this:
- `assume_ok_unexpected` (default): silently accept extra keys — most permissive, good for most resumption cases
- `raise_all`: error on any mismatch — strictest, useful for debugging

> All load config parameters and strictness options: [`ref/megatronbridge-checkpointing.md`](ref/megatronbridge-checkpointing.md)

---

## Workflow 5: Export Megatron checkpoint → HuggingFace

### Why this step exists

After training in Megatron's distributed format, you need standard HF format for deployment (vLLM, TGI), sharing (HuggingFace Hub), or evaluation (lm-eval-harness). The distributed `.distcp` shards must be gathered and converted back into monolithic SafeTensors files with HF-compatible parameter names.

### What happens behind the scenes

```
bridge.export_ckpt("./megatron_ckpt", "./hf_export")
│
├─ 1. temporary_distributed_context(backend="gloo")
│   ├─ Initialize a single-rank torch.distributed process group
│   ├─ Initialize Megatron parallel state (TP=1, PP=1)
│   └─ Why needed: MCore's checkpoint loading API requires torch.distributed
│      to be initialized, even when running on a single GPU for export
│
├─ 2. load_megatron_model("./megatron_ckpt")
│   ├─ Read run_config.yaml → reconstruct GPTModelProvider
│   ├─ provider.finalize() → compute derived fields
│   ├─ provider.provide_distributed_model() → create model
│   └─ dist_checkpointing.load() → populate weights from distcp shards
│      (single-rank = effectively "gather all shards into one model")
│
└─ 3. save_hf_pretrained(model, "./hf_export")
    ├─ Save model config (config.json)
    ├─ Save tokenizer files
    ├─ stream_weights_megatron_to_hf()
    │   └─ Iterate parameter-by-parameter, converting Megatron names/layouts
    │      to HF names/layouts (e.g. fused QKV → separate Q, K, V)
    └─ Write to SafeTensors format
```

### Three export modes and when to use each

| Method | What it saves | When to use |
|--------|--------------|-------------|
| `save_hf_pretrained()` | Config + tokenizer + weights | Deployment, sharing on HF Hub — produces a complete, loadable HF model |
| `save_hf_weights()` | Weights only (SafeTensors) | When you already have the config/tokenizer elsewhere and just need updated weights |
| `export_hf_weights()` | Streams `(name, tensor)` pairs | When you don't want to write to disk at all — e.g. feeding weights directly into an RL framework or eval pipeline |

### Gotcha: `from_hf_pretrained` vs `from_hf_config`

For export, always use `from_hf_pretrained()`. The `from_hf_config()` method only loads the architecture config (layer count, hidden size, etc.) but *not* the tokenizer or other artifacts. If you try to `save_hf_pretrained()` from a config-only bridge, the exported checkpoint will be incomplete (missing tokenizer files, special tokens map, etc.).

`from_hf_config()` is only useful for architecture introspection — checking supported models, inspecting transformer config, etc.

> Full API reference, CLI commands, and code examples: [`ref/megatronbridge-bridge-guide.md`](ref/megatronbridge-bridge-guide.md)

---

## Coverage Summary

| Workflow | MegatronBridge docs | Notes |
|---|---|---|
| 1. Import HF → Megatron | `ref/megatronbridge-bridge-guide.md` | CLI + Python API + one-liner |
| 2. Distributed checkpointing & resharding | `ref/megatronbridge-checkpointing.md` | Automatic via `torch_dist`; no standalone reshard tool |
| 3. Save during training | `ref/megatronbridge-checkpointing.md` | Full config table + dir structure |
| 4. Load on resume | `ref/megatronbridge-checkpointing.md` | `load`, `ckpt_step`, `pretrained_checkpoint` |
| 5. Export Megatron → HF | `ref/megatronbridge-bridge-guide.md` | Shards merged automatically; 3 export methods |
