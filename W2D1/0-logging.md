# Week 2, Day 1: Logs, Checkpointing, Resuming

## Key Knowledge-Points

### 1. Log Files and Log Directory Organization
- **Questions**: When a multi-node training job crashes at 2am, how do you figure out which node failed? If you need to reproduce a run from 3 weeks ago, how do you recover the exact config and code that was used? Why does rank-0 show output in the terminal but other ranks don't?
- **Intuition**: Every training submission creates a dedicated `LOG_DIR` at `{shared_fs}/log/mb/{sweep_name}/{job_name}/{job_id}/`. The `{job_id}` makes each submission's logs unique — you always get a fresh directory even when resuming. The `CKPT_DIR` (at `{shared_fs}/ckpt/mb/{sweep_name}/{job_name}/`) deliberately omits `{job_id}` so resumed jobs continue writing to the same checkpoint.

  **Full directory layout:**
  ```
  {shared_fs}/log/mb/{sweep_name}/{job_name}/{job_id}/
  ├── slurm.log  OR  pbs.log                   # scheduler stdout/stderr
  ├── launch.slurm  OR  launch.pbs             # copy of the generated job script
  ├── launcher.py                               # copy of launcher at submission time
  ├── resolve_config.py                         # copy of training entry point
  ├── utils.py                                  # copy of utility module
  ├── config.yaml                               # copy of the config used
  ├── data_config.yaml                          # copy of data config (if exists)
  ├── calc_proportion.py                        # copy of proportion calculator (if exists)
  ├── <rank>_python_master.log                  # master node torchrun output (tee'd to terminal)
  ├── <rank>_python.log                         # non-master node torchrun output (per node)
  ├── <rank>_<hostname>_sh.log                  # python/framework version info (first node only)
  ├── recipe/
  │   ├── original.yaml                         # recipe before overrides
  │   └── override.yaml                         # recipe after all overrides applied
  ├── nccl/
  │   └── <hostname>.log                        # per-node NCCL debug log
  ├── env/
  │   ├── EnvVar_hostOS.log                     # host OS environment variables
  │   └── EnvVar_<hostname>.log                 # per-node container environment variables
  └── tb_logs/
      └── events.out.tfevents.*                 # TensorBoard event files
  ```

  **Execution context — where each artifact is created:**

  Understanding _where_ each file is created helps you reason about what's available when a job fails at different stages:

  | File / Directory | Created by | Runs on |
  |---|---|---|
  | `slurm.log` / `pbs.log` | Scheduler stdout redirect (`#SBATCH --output`) | Head node (job shell) |
  | `launch.slurm` / `launch.pbs` | `cp "$0"` in outer bash script | Head node (job shell) |
  | `launcher.py`, `resolve_config.py`, `utils.py`, `config.yaml`, `data_config.yaml`, `calc_proportion.py` | `cp` commands in outer bash script | Head node (job shell) |
  | `EnvVar_hostOS.log` | `printenv` in outer bash script | Head node (job shell) |
  | `<rank>_python_master.log` | torchrun output via `tee` on master node | Master node (SLURM: last node; PBS: node 0) |
  | `<rank>_python.log` | torchrun output redirect on non-master nodes | Each non-master node |
  | `<rank>_<hostname>_sh.log` | Shell block gated on `node_rank == 0` | First node only |
  | `EnvVar_<hostname>.log` | `env` inside container | Each node |
  | `nccl/<hostname>.log` | `NCCL_DEBUG_FILE` env var | Each node |
  | `recipe/original.yaml`, `recipe/override.yaml` | `resolve_config.py` saves these during config resolution | DP_rank 0 (inside training process) |
  | `tb_logs/events.*` | TensorBoard writer | Last rank (logger rank) |

  The key distinction: the **outer bash script** (the generated SLURM/PBS job script) runs once on the head node and handles directory creation, artifact copying, and environment variable setup. Then it launches `torchrun` (via `srun`/`mpirun`) which spawns processes on **each node**. Within each node, torchrun creates `gpus_per_node` worker processes, but log files are per-node (not per-GPU) — only the logger rank writes TensorBoard and WandB metrics.

  **Why we structure logs this way:**

  1. **Group by nature and source**: Scheduler logs, training output, NCCL communication logs, and environment dumps are separated into different files and directories. NCCL debug output is especially verbose — isolating it in `nccl/` prevents it from drowning out training output. Environment variables go in `env/` so you can quickly check container configuration without scrolling through training logs.
  2. **Identify hardware failures from individual nodes/GPUs**: Per-node `<rank>_python.log` files and per-node `nccl/<hostname>.log` files let you pinpoint exactly which machine failed. In a 16-node job, if one node has a bad GPU, its `nccl/<hostname>.log` will show the NCCL timeout while all other nodes show normal communication. Without per-node logs, you'd only see "NCCL timeout" in the master log with no indication of the faulty node.

  **Why we save artifact copies (scripts and configs):**

  1. **Immediate script iteration**: After `sbatch`/`qsub`, the generated job script copies all source files (`launcher.py`, `resolve_config.py`, `config.yaml`, etc.) into `LOG_DIR`. The running job reads from these copies, not from your working directory. This means you can immediately edit scripts in your workdir and submit a new experiment — the already-running job is not affected, and there is no risk of contaminating its configuration mid-training.
  2. **Full traceability**: Weeks later, when you need to understand why a run behaved differently, the `LOG_DIR` contains the exact `config.yaml`, `resolve_config.py`, and `launcher.py` that were used. Combined with `recipe/original.yaml` (before overrides) and `recipe/override.yaml` (after all overrides), you have complete reproducibility without relying on git history or memory of what you changed.

  **Where local log folders live for each logging backend:**

  | Backend | Directory | Set by | Lifetime |
  |---|---|---|---|
  | TensorBoard | `{LOG_DIR}/tb_logs/` | `cfg.logger.tensorboard_dir` in `resolve_config.py` | Per-submission (fresh each job ID) |
  | WandB | `{CKPT_DIR}/wandb/` | `WANDB_DIR={CKPT_DIR}` env var in launcher.py | Persistent across resumes |
  | Checkpoints (NeMo) | `{CKPT_DIR}/` | `cfg.checkpoint.save` and `cfg.checkpoint.load` | Persistent across resumes |

  TensorBoard lives in `LOG_DIR` because each submission gets its own TB event files (matching the job's log lifetime). WandB lives in `CKPT_DIR` because WandB auto-resumes the run using its local state in `wandb/` — if the job is resubmitted, WandB picks up where it left off rather than creating a new run.

- **Exercise Steps**:
  1. Navigate to a real `LOG_DIR` and list its contents; match each file to its purpose in the directory layout above.
  2. Open `<rank>_python_master.log` and a non-master `<rank>_python.log` side-by-side — what's different? Why does only the master node get tee'd?
  3. Examine `env/EnvVar_hostOS.log` — which secrets are filtered out and how? Find the grep pattern in `launcher.py` that does the filtering.
  4. Compare `recipe/original.yaml` vs `recipe/override.yaml` — which fields changed? What does this tell you about how feature flags modify the base recipe?
  5. Explain why `LOG_DIR` includes `{job_id}` but `CKPT_DIR` does not — what would break if both used job_id, or if both omitted it?
  6. Using the execution-context table, identify which files would be missing if a job was killed before torchrun started on any node. Which files would still exist?

### 2. W&B Integration and Metrics Dashboard
- **Questions**: If `launcher.py` contains no `wandb.init()` calls, how does W&B know about the run? What does `WANDB_RUN_GROUP` control and why does it matter when you have dozens of experiments? If throughput suddenly drops by 30%, which W&B metric would you look at first?
- **Intuition**: W&B is wired entirely through environment variables set in the generated bash script — `WANDB_PROJECT`, `WANDB_RUN_GROUP` (= `SWEEP_NAME`), `WANDB_EXP_NAME` (= `{JOB_NAME}-{JOB_ID}`), `WANDB_MODE=online`, and `WANDB_DIR` (= `CKPT_DIR`). The actual `wandb.init()` and `wandb.log()` calls live inside MegatronBridge's training loop. The `resolve_config.py` `apply_logging_config()` function configures which metric groups to enable — all of the following are enabled by default in our setup:

  ```python
  cfg.logger.log_throughput = True
  cfg.logger.log_progress = True
  cfg.logger.log_l2_norm_grad_to_tensorboard = True
  cfg.logger.log_memory_to_tensorboard = True
  cfg.logger.log_params_norm = True
  cfg.logger.log_runtime_to_tensorboard = True
  cfg.logger.log_throughput_to_tensorboard = True
  cfg.logger.throughput_window_size = 20
  cfg.logger.runtime_time_unit = "hours"
  ```

  Below is a comprehensive walkthrough of all major metrics logged during MegatronBridge training, grouped by their original grouping in the codebase.

  ---

  **Core Training Metrics** (logged every `tensorboard_log_interval` steps by the training loop)

  **`lm loss`** (unit: nats) — Language model cross-entropy loss. This is the primary training signal.

  The loss goes through a multi-stage reduction pipeline before it reaches W&B:
  1. **Per-token loss**: The model's `forward_step()` produces per-token cross-entropy values in `output_tensor` (shape: `[seq_length]`).
  2. **Mask and sum**: The loss function multiplies element-wise by `loss_mask` (which zeros out padding tokens) and sums: `loss = torch.sum(losses * loss_mask)`. It also counts non-padded tokens: `num_tokens = loss_mask.sum()`.
  3. **Reporting format**: The loss is packed as a 2-element tensor `[loss_sum, num_tokens]` — this format allows correct averaging later.
  4. **Microbatch accumulation**: Each training step processes multiple microbatches. The 2-element tensors from all microbatches are stacked and summed column-wise: `val = torch.vstack(val).sum(dim=0)`.
  5. **DP reduction**: The summed `[loss_sum, num_tokens]` tensor is all-reduced (SUM) across all data-parallel and context-parallel ranks.
  6. **Final average**: `lm_loss = loss_sum / num_tokens` — a properly token-weighted average across all microbatches and all DP ranks.

  This token-weighted reduction is why the loss is accurate even when microbatches have different numbers of non-padded tokens (e.g., with variable-length packing).

  **`mtp_1 loss`, `mtp_2 loss`, ...** (unit: nats) — Per-head auxiliary loss for Multi-Token Prediction (only when MTP is enabled). Each MTP prediction head computes its own cross-entropy following the same pipeline as `lm loss`. Logged via `MTPLossLoggingHelper.track_mtp_metrics()`, which scales each loss by `1 / num_microbatches` before logging.

  **`learning-rate`** (unit: float) — Current learning rate from the LR scheduler. Read directly from the optimizer's parameter groups: iterates through `optimizer.param_groups` and returns the `'lr'` value from the first group marked `default_config=True`. All default-config groups share the same LR schedule, so any one is representative. On ranks without trainable parameters, this is `None` and gets filled in via `reduce_max_stat_across_model_parallel_group`.

  **`grad-norm`** (unit: float) — Global L2 norm of all gradients, computed **before** gradient clipping. The computation:
  1. On each rank, collect all gradient tensors (`p.main_grad` for each parameter with `requires_grad`).
  2. Compute the local L2 norm using Apex's `multi_tensor_l2norm` kernel (a fused CUDA operation that processes all gradient tensors in a single kernel launch for efficiency).
  3. Square the local norm: `total_norm = local_l2_norm ** 2`.
  4. All-reduce (SUM) the squared norm across all data-parallel and model-parallel ranks.
  5. Take the square root: `final_grad_norm = total_norm ** 0.5`.

  After this norm is computed, gradient clipping scales all gradients by `clip_coeff = max_norm / (grad_norm + 1e-6)` if `clip_coeff < 1.0`. The logged `grad-norm` is the **pre-clipping** value — it tells you the "raw" gradient magnitude before any clipping intervention.

  **`params-norm`** (unit: float) — L2 norm of all model parameters. Computed similarly to `grad-norm` but over parameter values instead of gradients:
  1. Parameters are separated into three groups: dense parameters, MoE (Mixture-of-Experts) parameters, and sharded main parameters (for FSDP). Each group needs different reduction strategies because they are sharded differently across ranks.
  2. For each group, the local L2 norm is computed via `multi_tensor_l2norm`, squared, and all-reduced (SUM) across the appropriate parallel groups (DP + TP + PP for dense; DP + TP + EP + PP for MoE).
  3. BF16 parameters are converted to FP32 (via the optimizer's `.main_param` attribute) before norm computation to avoid precision issues.
  4. All squared norms are summed and the final result is `sqrt(total_squared_norm)`.

  **`loss-scale`** (unit: float) — Dynamic loss scaler value for mixed-precision (FP16/BF16) training. The loss scaler is a state machine that prevents gradient underflow/overflow:
  - **Scale-down**: When NaN or Inf values are detected in gradients, a `hysteresis_tracker` counts down. After `hysteresis` consecutive NaN iterations, the scale is reduced: `scale = max(scale * backoff_factor, min_scale)`. The `min_scale` is the floor (e.g., 1.0) below which the scaler cannot go.
  - **Scale-up**: When no NaN/Inf is detected, a `growth_tracker` increments. After `growth_interval` consecutive clean iterations, the scale increases: `scale = scale * growth_factor`.
  - The logged value is `optimizer.get_loss_scale().item()` — the current scale at that step.

  A monotonically decreasing `loss-scale` that hits the `min_scale` floor means the model is consistently producing NaN/Inf gradients — a sign of severe training instability (typically caused by too-high learning rate, data corruption, or numerical issues in the model architecture).

  **`batch-size`** (unit: count) — Global batch size for the step, computed as `micro_batch_size * data_parallel_size * num_microbatches`. This is the total number of samples processed across all GPUs in one training step. Logged from config, not measured.

  **`iteration-time`** (unit: seconds) — Average wall-clock time per training step over the most recent logging interval. Computed as:
  ```
  iteration-time = elapsed_time / total_iterations
  ```
  where `elapsed_time` comes from a barrier-synchronized timer (`timers('interval-time').elapsed(barrier=True)`) that includes all-reduce, optimizer step, and data loading time. The barrier ensures all ranks have finished before measuring, so this captures the slowest rank's time (i.e., straggler effects are visible here).

  **`samples vs steps`** (unit: count) — Cumulative consumed training samples, tracked by the training state counter (`train_state.consumed_train_samples`). Incremented by `global_batch_size` each step. This provides an alternative x-axis for W&B charts — useful when comparing runs with different batch sizes.

  **`skipped-train-samples`** (unit: count) — Samples skipped when the loss scaler detects NaN/Inf in gradients and the optimizer step is skipped. Only logged when `> 0`.

  ---

  **Throughput Metrics** (rolling window average, from `report_throughput()`)

  These metrics use a rolling window to smooth out per-step variance. The implementation maintains a `deque(maxlen=window_size+1)` that stores the absolute wall-clock timestamp (`time.time() - start_time`) at each step. Once the deque has enough entries, throughput is computed over the most recent `window_size` steps (default: 20, set by `cfg.logger.throughput_window_size`):

  ```python
  elapsed_wct = history_wct[-1] - history_wct[0]   # wall-clock span of the window
  elapsed_samples = (iteration * GBS) - ((iteration - window_size) * GBS)
  samples_per_sec = elapsed_samples / elapsed_wct
  ```

  | Metric | Unit | Scope | Formula |
  |---|---|---|---|
  | `throughput/batches_per_sec` | batches/sec | Global (all GPUs) | `window_size / elapsed_wct` |
  | `throughput/samples_per_sec` | samples/sec | Global | `(window_size * global_batch_size) / elapsed_wct` |
  | `throughput/tokens_per_sec` | tokens/sec | Global | `(window_size * global_batch_size * seq_length) / elapsed_wct` |
  | `throughput/device/batches_per_sec` | batches/sec | Per GPU | `throughput/batches_per_sec / world_size` |
  | `throughput/device/samples_per_sec` | samples/sec | Per GPU | `throughput/samples_per_sec / world_size` |
  | `throughput/device/tokens_per_sec` | tokens/sec | Per GPU | `throughput/tokens_per_sec / world_size` |

  No data is logged until the deque fills up (i.e., the first `window_size` steps produce no throughput metrics). If the window's elapsed wall-clock time is zero or negative (can happen during checkpoint resumption), the calculation is skipped entirely to avoid division by zero.

  **`throughput/tflops/device`** (unit: TFLOP/s per GPU) — Model FLOP utilization per GPU. This is the key hardware efficiency indicator. Unlike the rolling-window metrics above, this is computed from the logging-interval timer:

  ```
  throughput/tflops/device = num_floating_point_operations(config, batch_size)
                             / elapsed_time_per_iteration
                             / world_size
                             / 1e12
  ```

  **`throughput/tflops`** (unit: TFLOP/s total) — `throughput/tflops/device * world_size`.

  The `num_floating_point_operations` function counts all multiply-accumulate operations in the model. For a standard Transformer:

  ```
  flops = batch_size * seq_length * (
      # Attention (per layer): Q/K/V projections + attention scores + output projection
      12 * num_layers * hidden_size^2 * (1 + GQA_ratio + seq_length / (2 * hidden_size))
      # MLP (per layer): up-projection + down-projection (3/2x for SwiGLU gating)
    + 12 * num_layers * hidden_size * ffn_hidden_size * gated_multiplier
      # Logits: embedding-to-vocabulary projection (+ MTP heads if enabled)
    + 6  * hidden_size * padded_vocab_size * (1 + mtp_num_layers)
  )
  ```

  The 12x factor comes from: 3x (forward + backward wgrad + backward dgrad) * 2x (two stacked GEMMs per block) * 2x (multiply-accumulate = 2 FLOPs per element). See [Narayanan et al., 2021, Appendix](https://arxiv.org/abs/2104.04473) for the derivation. Note this counts **model FLOPs only** — it excludes data loading, communication, and optimizer overhead, so the reported TFLOP/s will always be lower than the GPU's theoretical peak.

  Interpretation: For H100 GPUs with BF16, well-optimized configs achieve 150-180 MODEL_TFLOP/s per GPU. A sudden 30% drop usually indicates a straggler node (check per-node NCCL logs) or a data loading bottleneck.

  ---

  **Memory Metrics** (from `report_memory()`, converted to GB)

  All memory metrics are read from `torch.cuda.memory_stats()` and converted from bytes to gigabytes (`GB = bytes / 1e9`, rounded to 5 significant digits). To understand these metrics, you need to know how PyTorch's CUDA caching allocator works:

  - When a tensor is created, PyTorch requests a memory block from the caching allocator. The allocator either reuses a cached free block or calls `cudaMalloc` to get a new one from the GPU driver.
  - When a tensor is freed (`del tensor` or goes out of scope), the memory is returned to the **cache**, not back to the GPU driver. This avoids expensive `cudaMalloc`/`cudaFree` calls.
  - **Reserved** memory = everything the allocator has obtained from the driver (both in-use and cached free blocks).
  - **Allocated** memory = blocks that have been handed out to tensors (some may have been freed but the block isn't returned to the free pool yet).
  - **Active** memory = blocks currently containing at least one live tensor.
  - **Inactive split** memory = free sub-blocks created when a large cached block is split to satisfy a smaller allocation. These fragments cannot be returned to the driver because they're part of a larger `cudaMalloc` region.

  The relationship is: `reserved >= allocated >= active`, and `inactive_split` represents fragmentation overhead within allocated blocks.

  | Metric | What it measures | What to watch for |
  |---|---|---|
  | `memory/current_allocated_gigabytes` | Total bytes in blocks that have been allocated (handed out by the caching allocator), converted to GB. | Baseline memory footprint — this is your working set. |
  | `memory/current_active_gigabytes` | Bytes in blocks containing at least one live (non-freed) tensor. | Memory actually being used by tensors right now. The gap between allocated and active indicates recently-freed tensors whose blocks haven't been recycled. |
  | `memory/current_inactive_gigabytes` | Bytes in free sub-blocks created by block splitting (fragmentation). These blocks are cached but cannot be merged back. | High values mean the allocator has fragmented memory. This is normal during training warm-up but should stabilize. |
  | `memory/current_reserved_gigabytes` | Total bytes reserved from the GPU driver by the caching allocator. | This is the actual GPU memory consumed as seen by `nvidia-smi`. The gap between reserved and allocated is the free cache (blocks available for reuse without calling `cudaMalloc`). |
  | `memory/peak_allocated_gigabytes` | High-water mark for allocated memory since training started. | Maximum memory needed during training — usually peaks during backward pass when both activations and gradients exist simultaneously. |
  | `memory/peak_active_gigabytes` | High-water mark for active memory. | Maximum simultaneously-live tensors. |
  | `memory/peak_inactive_gigabytes` | High-water mark for inactive (fragmented) memory. | Worst-case fragmentation during training. |
  | `memory/peak_reserved_gigabytes` | High-water mark for reserved memory. | How close you got to OOM — compare with GPU total memory (e.g., 80 GB for H100). |
  | `memory/alloc_retries` | Count of failed `cudaMalloc` calls that triggered a cache flush and retry. The allocator frees all cached blocks and retries the allocation. | **Any value > 0 is a warning**: you are at the edge of OOM. The allocator had to evict its entire cache to satisfy a request. Frequent retries will degrade throughput significantly due to the cache flush overhead. |

  ---

  **Runtime Metrics** (from `report_runtime()`, time unit = hours)

  | Metric | Unit | How obtained |
  |---|---|---|
  | `time/remaining_estimate` | hours | Extrapolates from current progress: computes `elapsed_fraction = current_step / train_iters`, then `remaining = (elapsed_time / elapsed_fraction) * (1 - elapsed_fraction) / 3600`. This assumes constant throughput — the estimate will be inaccurate if throughput changes (e.g., after adding/removing nodes, or during evaluation pauses). |
  | `time/tokens` | count | `consumed_train_samples * seq_length`. This is the total number of tokens the model has been trained on, useful for comparing runs with different batch sizes or sequence lengths. |
  | `time/samples` | count | Directly from the training state counter `train_state.consumed_train_samples`. Incremented by `global_batch_size` each step. |
  | `time/batches` | count | `train_state.step` — the global step counter. Equivalent to the number of optimizer updates performed. |
  | `time/total` | hours | `(time.time() - start_time) / 3600`. Wall-clock time since the training process started (not since the job was submitted — excludes queue wait time and initialization). |

  ---

  **Per-Layer Gradient Norms** (from `report_l2_norm_grad()`)

  | Metric | Description |
  |---|---|
  | `l2_norm/grad/global` | L2 norm across all model parameters — `sqrt(sum of all per-layer squared norms)`. |
  | `l2_norm/grad/<layer_name>` | L2 norm for each individual named parameter that has a gradient (e.g., `l2_norm/grad/decoder.layers.0.self_attention.linear_qkv.weight`). |

  The computation iterates over `model.named_parameters()` and for each parameter with a non-`None` `main_grad`, computes `torch.linalg.vector_norm(p.main_grad)`. The global norm is then `sqrt(sum(per_layer_norm ** 2))`. Tensor results are converted to Python floats via `.item()` before logging.

  Note: Unlike `grad-norm` (which is computed inside the optimizer's gradient clipping routine using the fused `multi_tensor_l2norm` kernel), `l2_norm/grad/*` is computed separately in `report_l2_norm_grad()` **after gradient unscaling**. The values should be very close but may differ slightly due to floating-point ordering.

  Interpretation: This is the most granular diagnostic for gradient health. If `grad-norm` spikes, use per-layer norms to identify the responsible layer. Common culprits: embedding layers, the final LM head, or the first attention layer. Watch for layers where the norm is orders of magnitude larger than peers — this indicates localized instability that global `grad-norm` alone would not pinpoint.

  ---

  **Energy Metrics** (hardware-dependent, requires NVML support)

  | Metric | Unit | How obtained |
  |---|---|---|
  | `iter-energy/gpu` | Joules/iter/GPU | The energy monitor reads cumulative GPU energy consumption via NVML (NVIDIA Management Library) at each logging interval. The per-iteration, per-GPU energy is: `energy_monitor.lap() / total_iterations / world_size`. The `.lap()` method returns the total energy consumed since the last call, measured by the GPU's onboard power sensor. |
  | `power/gpu` | Watts/GPU | Derived from energy: `power = energy / elapsed_time_per_iteration`. This gives the average power draw per GPU during the logging interval (not instantaneous — it's smoothed over `log_interval` steps). |

  These metrics require compatible GPU hardware and drivers (NVML). On systems without NVML support (e.g., some container configurations), they are simply not logged. Typical H100 power draw during training is 500-700W depending on workload intensity.

  ---

  **Validation Metrics** (logged every `eval_interval` steps)

  | Metric | When logged |
  |---|---|
  | `lm loss validation` | Every `eval_interval` steps (single validation set). Computed identically to training `lm loss` — cross-entropy averaged over tokens — but on validation data without gradient computation. |
  | `lm loss validation <dataset_name>` | Per-dataset validation loss (when `multiple_validation_sets: true`). The `<dataset_name>` is the basename of the validation data path. Each dataset is evaluated independently using `eval_iters` batches. |
  | `lm loss validation (aggregated)` | Unweighted mean across all per-dataset validation losses. |

  See Section 3 below for the rationale behind multiple validation datasets and how to detect domain collapse.

- **Exercise Steps**:
  1. Open a real W&B run for one of our training jobs. Find the W&B run using `Group = {SWEEP_NAME}` and `Name = {JOB_NAME}-{JOB_ID}`. Identify which metric groups are present.
  2. Locate where `WANDB_EXP_NAME` is set in `launcher.py` and trace how it flows into `cfg.logger.wandb_exp_name` in `resolve_config.py`.
  3. Look at `throughput/tflops/device` over the first 100 steps of a run — does it stabilize? What causes the initial ramp-up?
  4. Compare `grad-norm` across two training runs with different learning rates — what do high vs. healthy grad-norm curves look like?
  5. If `loss-scale` is monotonically decreasing and hitting the minimum floor, what does that indicate about training stability?
  6. Find `report_throughput()` and `report_memory()` in the MegatronBridge `train_utils.py` source. What is the `throughput_window_size`? How would changing it affect the smoothness of the throughput curve?
  7. Calculate the expected `throughput/tflops/device` for a known model config (e.g., Qwen3-4B with `seq_length=8192`, `GBS=1024`) using the TFLOP formula above. Compare your calculation with the actual W&B value — what accounts for the difference?

### 3. Multiple Validation Datasets
- **Questions**: If you're training on a mix of English Wikipedia, Malay news, and math code, and validation loss improves overall — how do you know the model isn't getting worse on math? Why might a single aggregated validation loss hide important regression? What's the minimum number of validation sets you'd want for a multilingual continual pretraining run?
- **Intuition**: LLM pretraining uses mixed data from many domains. A single validation set gives one number that hides per-domain behavior. We enable multiple validation datasets via `multiple_validation_sets: true` in the config plus a list of separate `valid_data` paths. During evaluation (every `eval_interval` steps), the trainer iterates over each validation dataset independently and logs per-dataset losses. W&B metrics follow the pattern `lm loss validation <dataset_name>` (where `<dataset_name>` is the path basename), plus `lm loss validation (aggregated)` (mean across all datasets). This is critical for catching **domain collapse** — a failure mode where training on a new domain pushes the model to forget previously learned domains, which the aggregated loss may not reveal until it's severe.
- **Exercise Steps**:
  1. Find `multiple_validation_sets` in `resolve_config.py` and `launcher.py` — trace the full path from the `--multival` CLI flag through the config preset to the final `cfg.dataset.multiple_validation_sets = True`.
  2. On a W&B run with multi-validation enabled, plot `lm loss validation EN_Wikipedia` and `lm loss validation MY_Fineweb2` on the same chart alongside `lm loss validation (aggregated)` — do they diverge?
  3. Design a scenario where `lm loss validation (aggregated)` improves while one domain's loss gets worse. What type of training data mix would cause this?
  4. What is `eval_iters` and how does it affect the reliability of per-dataset validation loss estimates? What's the tradeoff with evaluation frequency?
