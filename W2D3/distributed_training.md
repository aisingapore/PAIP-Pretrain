
# Procedures to follow for exercises

The exercises in this section will require launching multiple training jobs using the [Nvidia Nemo Framework v25.11](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo?version=25.11) container. The steps to launch a training job are as follows:

1. Launch an interactive Slurm job with 4 GPUs: `srun --gres=gpu:4 --pty bash`
2. Create enroot container using the provided sqsh archive: `enroot create -n test /mnt/weka/aisg/sqsh/nemo:25.11.sqsh`.
3. Launch the container: `enroot start --rw test`.
4. Activate the python environment: `source /opt/venv/bin/activate`.
5. Navigate to the working directory, which should be automatically mounted to the enroot container: `cd /mnt/weka/aisg/model_training_team/code_forge/yuli/repos/PAIP-Pretrain/W2D3`.
6. Create a `logs` folder in the working directory: `mkdir logs`.


7. Run the training job after setting the necessary parameters in the [qwen3_pretrain_override.yaml](qwen3_pretrain_override.yaml) configuration file. It is recommended to save the training logs to a file for future reference: `torchrun --nproc_per_node 2 qwen3_pretrain.py --config-file qwen3_pretrain_override.yaml | tee -a logs/test.log`.

8. Retrieve the throughput and memory usage metrics by running the `mean_flops.sh` bash script: `bash mean_flops.sh`.

Note that steps 1-6 only need to be performed once. 

Upon completing the exercises, exit both the container and the interactive Slurm job session. The `exit` command can be used in both cases.


# Distributed Training of Large Language Models

Modern Large Language Models (LLMs) are trained in parallel across multiple GPUs. The objective of this module is to explain the considerations relevant for training models at scale. 

# Challenges associated with LLM training

1. **Memory**: LLM training requires performing multiple forward and backward passes of numerous batches of data through a model containing numerous layers. During training, the model weights, gradients, activations and optimizer states need to be regularly accessed and updated. This can quickly overwhelm the video Random Access Memory (vRAM) available on a single GPU. The vRAM specification of a graphics card determines the maximum volume of data it can process at any given time.

2. **Throughput**: The need to process large volumes of data implies that the training cannot be completed within a reasonable time unless the throughput is sufficiently high. The batch size required for training at scale typically requires the training to be distributed across multiple GPUs, to enable high throughput without exceeding the memory available on a single GPU.


# Single GPU Training

## Theory

Read the section entitled "First Steps: Single GPU Training" in the [Nanotron UltraScale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=high-level_overview).

Answer the following questions:


1. How does the batch size affect the overall training time?

2. What is activation recomputation? What is the key trade-off associated with it?

3. What is gradient accumulation? What is the relationship between the global and micro batch sizes?


## Practical

The hands-on exercises will involve training the Qwen3-4B model using the [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main) framework. The training parameters can be configured using the [qwen3_pretrain_override.yaml](qwen3_pretrain_override.yaml) file. The existing set of parameters should be used as a baseline for comparison. All training runs in this section should be performed using a single H200 GPU.


Perform an initial training run without changing any parameters in the [qwen3_pretrain_override.yaml](qwen3_pretrain_override.yaml) file. Take note of the following results from the training logs:

- Theoretical memory footprint (MB)
- Allocated and reserved memory after the first iteration
- Step time (s) 
- Throughput per GPU (TFLOP/s/GPU)
- Validation and test losses


---Answer Begin---
```
Theoretical memory footprints: weight and optimizer=69053.29 MB
[Rank 0] (after 1 iterations) memory (MB) | allocated: 69454.74462890625 | max allocated: 69454.76025390625 | reserved: 73472.0 | max reserved: 73472.0
Average Step Time: 4.097s
Average GPU Utilization: 200.33 TFLOP/s/GPU
```
---Answer End---

Perform the following experiments:

1. Try running the training with the `global_batch_size` parameter in the `train` section (i.e. `train.global_batch_size`) set to 16 and 64. How do the `Step time` and `Throughput per GPU` metrics vary with global batch size?


---Answer Begin---

| Global batch size | Step time (s) | Throughput per GPU (TFLOP/s/GPU) |
| --- | --- | --- |
| 16 | 2.11 | 194.68 |
| 32 | 4.10 | 200.33 |
| 64 | 8.09 | 202.99 |


There is a slight increase in throughput with increasing global batch size because the optimizer step can be executed after peforming the forward and backward passes for more samples. Since the time required for the optimizer step depends only on the number of model parameters, the number of samples processed per second is slightly greater for larger global batch sizes.

The extent of increase in throughput decreases with increasing global batch size. Let $b$, $x$ and $y$ denote the global batch size, total time required for forward and backward passes per sample and time required for the optimizer step respectively. The number of samples processed per second is given by:

$$
Number of samples processed per second = \frac{b}{bx + y} = \frac{1}{x + \frac{y}{b}}
$$

As the global batch size becomes very large, the number of samples processed per second approaches the constant value of $1/x$. Hence, increasing global batch size does not lead to a significant increase in throughput when this quantity is already very large.

---Answer End---

2. Try running the training with the `micro_batch_size` parameter in the `train` section (i.e. `train.micro_batch_size`) set to 2, 4, 8 and 16. How does the throughput vary with micro batch size? If you obtain an error for any of these cases, explain the likely reasons.

---Answer Begin---

| Micro batch size | Step time (s) | Throughput per GPU (TFLOP/s/GPU) |
| --- | --- | --- |
| 1 | 4.10 | 200.33 |
| 2 | 2.25 | 364.83 |
| 4 | 2.06 | 398.66 |
| 8 | 1.96 | 419.49 |

Increasing the micro batch size results in a considerable increase in throughput. The extent of increase is significantly greater than that obtained by increasing the global batch size. This is because tensors of larger size can be used to perform the computation in each forward and backward pass. Hence, the time required for each forward and backward pass increases less than proportionately to the extent of increase in batch size, which improves the training efficiency.

However, there is a limit to how large the micro batch size can be. For a micro batch size of 16, a CUDA out of memory error is encountered. In this case, the vRAM available on a single GPU is insufficient to store the activations for the entire batch.

__Answer End__



3. Megatron-Bridge supports activation recomputation using the `recompute_granularity` parameter in the `model` section. The effect of activating this setting during training will be investigated in this task.

    a. Try running the training with `recompute_granularity` set to `full` and `selective`. This will require uncommenting the line containing `recompute_granularity` in the [qwen3_pretrain_override.yaml](qwen3_pretrain_override.yaml) file. How do the `Step time` and `Throughput per GPU` metrics change as compared to the baseline run for which this setting was not enabled? Explain the reasons for the observed differences.

    ---Answer Begin---

    | Setting | Step time (s) | Throughput per GPU (TFLOP/s/GPU) |
    | --- | --- | --- |
    | None | 4.10 | 200.33 |
    | Full | 6.67 | 122.97 |
    | Selective | 5.11 | 160.72 |

    Enabling full activation recomputation decreases the throughput considerably. In this case, none of the activations are stored during the forward pass, requiring them to be recomputed during the backward pass. As the activations account for a small proportion of the GPU memory usage for short sequence lengths, there is no benefit in enabling activation recomputation in such cases.

    The throughput is better for selective as compared to full recomputation as it only recomputes those activations which with a larger memory footprint and which are cheaper to recompute. See the [Megatron documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.transformer_config.html) for more details.
    
    ---Answer End---



    b. Increase `model.seq_length` and `dataset.sequence_length` to 16384. Run the training without activation recomputation, and with `recompute_granularity` set to `full` and `selective`. How do the `Step time` and `Throughput per GPU` metrics compare for these three cases? If you encounter an error for any of these cases, explain the likely reasons. 

    ---Answer Begin---

    One encounters errors related to insufficient memory if activation recomputation is disabled or set to selective. For long sequence lengths, the activations account for a significant proportion of the vRAM usage and attempting to store even a subset of them may be infeasible. The training only works when activation recomputation is set to full. In this case, the average step time and throughput per GPU per second are 58.03 seconds and 349.04 TFLOP/s/GPU respectively.

    ---Answer End---

    c. Apart from the case of large sequence lengths, what are some other scenarios where activation recomputation may be useful?

    ---Answer Begin---
    It may be needed when training models with a large number of parameters as the hidden size would be large in such cases. It may also be required when using large batch size.

    ---Answer End---

TODO: Try to run the training for long sequence lengths with activation offloading to CPU. Enabling this setting causes a ListIndexOutOfRange error when using the `nemo:25.09.nemotron_nano_v2_vl` container.


# Data Parallelism

A total of 4 H200 GPUs will be used for this task.

## Theory

Read the section entitled "Data Parallelism" in the [Nanotron UltraScale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=data_parallelism).

Answer the following questions:


1. What is the problem with a naive implementation of data parallelism in which the all-reduce operation to aggregate gradients is only triggered after the backward pass?

2. What are the 3 methods that can be used to improve the efficiency of distributed data parallelism (DDP)?

3. What is the relationship between global batch size, micro batch size, number of gradient accumulation steps and number of data parallel ranks?

4. What are some limitations of using DDP with the model parameters, gradients and optimizer states replicated across all ranks?

5. How does Zero-1 improve the efficiency of DDP? How does it change the type of communication operations performed during training?

6. What is the difference between Zero-1 and Zero-2?

7. Explain the difference between Zero-2 and Zero-3. 

8. What is prefetching?

9. What is the key limitation of the ZeRO technique?


## Practical

1. Run a DDP training job using the [qwen3_pretrain_override.yaml](qwen3_pretrain_override.yaml) file with the number of data parallel (DP) ranks set to 2 and 4. How does the throughput change with varying number of DP ranks? 

Note that Megatron-LM internally calculates the number of DP ranks as `dp_size = world_size/(tp_size * pp_size * cp_size)` in `/megatron/core/parallel_state.py`. In this section, the number of DP ranks will always be equal to the world size since `tp_size = pp_size = cp_size = 1`. The `world_size` parameter can be adjusted using the `--nproc-per-node` flag in the torchrun command used to run the training.

---Answer Begin---

For some reason, the Megatron-Bridge throughput is slightly different for the distributed and non-distributed optimizer cases even when running on a single GPU (i.e. world size = 1). All the results above are for the distributed optimizer case. The results for the single DP rank case in the following table were obtained for the non-distributed optimizer case.


| DP Ranks | Step time (s) | Throughput per GPU (TFLOP/s/GPU) |
| --- | --- | --- |
| 1 | 3.92 | 209.16 |
| 2 | 2.05 | 199.92 |
| 4 | 1.10 | 186.76 |

The throughput per GPU decreases slightly with increasing number of DP ranks. This is because the communication overhead of the all-reduce operation used to synchronize gradients across devices increases as the number of DP ranks increases, since the total volume of data exchanged increases.

---Answer End---

2. Repeat the 3 runs as question 1 using the same numbers of DP ranks but with the `overlap_grad_reduce` parameter set to `False` in the [qwen3_pretrain_override.yaml](qwen3_pretrain_override.yaml) file. Does the throughput change as compared to the corresponding runs in question 1? Do not worry about trying to explain the reasons for your observations as they will be explored in the next question.


---Answer Begin---

| DP Ranks | Step time (s) | Throughput per GPU (TFLOP/s/GPU) |
| --- | --- | --- |
| 1 | 3.90 | 210.23 |
| 2 | 2.05 | 199.86 |
| 4 | 1.10 | 186.07 |

There is no significant change in the throughput when `overlap_grad_reduce` is set to `False`. The throughput per device still decreases to the same extent with increasing number of DP ranks. 

There are two reasons for this:

i. The communication of gradients cannot be completely overlapped with the backward pass since the gradients for the first few layers can only be all reduced after the backward pass is complete.

ii. Overlapping computation with communication results in concurrent execution of multiple kernels in different CUDA streams. As explained [here](https://anakli.inf.ethz.ch/papers/gpu_interference_socc25.pdf), this can lead to competition for GPU resources such as L2 cache, memory bandwidth, warp scheduling and compute pipelines.

---Answer End---

3. Profiling is a useful tool to understand the effects of various parallelism configurations. Repeat the training run in question 1 for the 2 GPU case with Pytorch profiling enabled. 

This requires setting the `use_pytorch_profiler` parameter to 'true' in the [qwen3_pretrain_override.yaml](qwen3_pretrain_override.yaml) file. The profiling traces will be generated as files with the `.pt.trace.json` extension in the `nemo_experiments/default/tb_logs` subfolder of your working directory. Download one of the files and visualize them using the [Perfetto UI](ui.perfetto.dev).


When analyzing the traces, pay specific attention to the following points:

a. How many streams are present in the trace? What is the purpose of each stream? More information regarding CUDA streams can be found [here](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams) and [here](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-semantics).

b. Which are the operations that account for the largest proportion of total wall duration in each stream? How many times were these operations executed? What is the average wall duration of these operations?

c. Do any of the operations identified in (b) take longer than average when overlapped with an operation from a different stream?

d. Are the executions of kernels on different streams perfectly overlapped? If not, what is preventing this from happening?

---Answer Begin---

a. There are 2 streams in the trace. Stream 7 is for computation while stream 35 is for communication. In this case, communication occurs through the all-reduce operation.

b. In the computation stream, the `nvjet_tss_128x256_64x4_2x1_v_badd_coopA_NTN` kernel accounts for the largest proportion of total wall duration. It is executed 1152 times. The average wall duration of this kernel is 157.9 $\mu$s. In the communication stream, the `all_reduce` operation accounts for the largest proportion of total wall duration. It is executed 72 times. The average wall duration of this operation is 792.3 $\mu$s.

c. The `nvjet_tss_128x256_64x4_2x1_v_badd_coopA_NTN` kernel execution time takes significantly longer than average when overlapped with the nccl kernel responsible for the all-reduce operation. This is likely caused by the two kernels competing for the same resources on the GPU as explained [here](https://anakli.inf.ethz.ch/papers/gpu_interference_socc25.pdf).

d. The all-reduce operation is not entirely overlapped with the execution of the computation kernels. The gradients for the first few layers can only be synchronized after the backward pass is complete.

---Answer End---

4. Use the findings from question 3 to explain why the throughput per device does not change significantly when the `overlap_grad_reduce` parameter is set to `false` in the [qwen3_pretrain_override.yaml](qwen3_pretrain_override.yaml) file.

---Answer Begin---

The answer is stated in question 2.

---Answer End---

5. Run the training on 1, 2 and 4 GPUs using FSDP with optimizer state sharding only. This requires setting the `use_megatron_fsdp` and `use_distributed_optimizer` parameters to 'true' in the [qwen3_pretrain_override.yaml](qwen3_pretrain_override.yaml) file. The `data_parallel_sharding_strategy` parameter should be set to `optim`. The `ckpt_format` parameter should be set to `fsdp_dtensor`. Compare the throughput and memory usage for DDP and FSDP with optimizer state sharding. Run the PyTorch profiler for the FSDP case to obtain additional insights into the results. In this case, pay special attention to the profiling trace for the host (CPU) side. The traces for the various CPU threads are typically labelled using large integers such as `python 1789435`. You may also find it useful to go through [this paper](https://arxiv.org/abs/2304.11277), which explains the fundamentals of FSDP. 


---Answer Begin---

All results from this point onwards are obtained using the `nemo:25.11` container because neither the Megatron nor the Torch variants of FSDP work correctly with the `nemo:25.09.nemotron_nano_v2_vl` container.

Results for DDP with non-distributed optimizer using `nemo:25.11` container. The `Memory per GPU (GB)` column refers to the 'mem-allocated-gigabytes' field from the logs after one training iteration:

| DP Ranks | Step time (s) | Throughput per GPU (TFLOP/s/GPU) | Memory per GPU (GB) |
| --- | --- | --- | --- |
| 1 | 4.10 | 200.11 | 73.65 |
| 2 | 2.12 | 193.52 | 73.65 |
| 4 | 1.13 | 181.03 | 73.65 |


Results for FSDP:

| DP Ranks | Step time (s) | Throughput per GPU (TFLOP/s/GPU) | Memory per GPU (GB) |
| --- | --- | --- | --- |
| 1 | 5.43 | 151.27 | 72.60 |
| 2 | 2.77 | 147.96 | 48.46 |
| 4 | 1.48 | 138.62 | 36.39 |

The throughput is consistently smaller for FSDP as compared to DPP, even for the single GPU case. The PyTorch CPU profiling trace shows that the FSDP run invokes several hooks before and/or after the forward and backward passes through each FSDP unit. The need to process these hooks on the CPU side before and/or after every forward and backward pass through each FSDP unit increases the time interval between the launch of kernels that perform computation on the GPU, which reduces the throughput as compared to DDP. Two of the important functions performed by these hooks include: 

* Coordination of computation and communication operations. See section 4.3 of the [FSDP paper](https://arxiv.org/abs/2304.11277).

* Performing flatten and unflatten operations on parameters to facilitate computation using `FlatParameter` objects.

The key advantage of FSDP is the lower memory footprint per GPU as compared to DDP. For large models, this consideration becomes important.

---Answer End---

6. Perform training using FSDP on 1, 2 and 4 GPUs for two additional cases. In the first case, shard the optimizer states and gradients by setting `data_parallel_sharding_strategy` to `optim_grads`. In the second case, include the model parameters in the sharding by setting this parameter to `optim_grads_params`. Compare the results for these two cases with those obtained in question 5. 

---Answer Begin---

Results for FSDP with optimizer state and gradient sharding:

| DP Ranks | Step time (s) | Throughput per GPU (TFLOP/s/GPU) | Memory per GPU (GB) |
| --- | --- | --- | --- |
| 1 | 5.43 | 151.12 | 72.60 |
| 2 | 2.78 | 147.78 | 40.33 |
| 4 | 1.46 | 140.59 | 24.25 |

Results for FSDP with optimizer state, gradients and model parameter sharding:

| DP Ranks | Step time (s) | Throughput per GPU (TFLOP/s/GPU) | Memory per GPU (GB) |
| --- | --- | --- | --- |
| 1 | 5.74 | 142.86 | 72.67 |
| 2 | 3.05 | 134.39 | 38.22 |
| 4 | 1.55 | 132.60 | 20.04 |

The memory footprint per GPU progressively decreases as the number of data parallel ranks increases and as more quantities are sharded. 

The throughput per GPU for the case in which both optimizer states and gradients are sharded is similar to the case in which only optimizer states are sharded. In comparison, the throughput per device decreases slightly if the model parameters are also sharded across data parallel ranks.

The sharding of model parameters requires all-gather operations to be performed during the forward and backward passes to materialize the parameters of the layer for which the computation is being performed. If the gradients are also sharded, reduce-scatter operations are also required to assign each DP rank its shard of the gradients. When the backward pass is overlapped with these communication operations, the resulting kernel interference increases the time required to perform the computation for the backward pass. This results in a slight decrease in throughput.

---Answer End---


# Tensor Parallelism

## Theory

Read the section entitled "Tensor Parallelism" in the [Nanotron UltraScale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism).

Answer the following questions:

1. What are the two main types of sharding strategies used in tensor parallelism? What are the distributed communication operations used in each type of tensor parallelism?

2. What is the optimal ordering of these sharding strategies in a feedforward multi-layer perceptron (MLP) block?

3. Explain the trade-off between the memory savings and communication overhead arising from the use of tensor parallelism.

4. Can tensor parallelism (TP) scale across nodes effectively? Why or why not?

5. How does sequence parallelism (SP) complement tensor parallelism?

6. How does the communication cost of using TP and SP together compare with the cost of using TP alone?


## Practical

All results from this point onwards were obtained on SMC. The preceding results were obtained on Hopper.

1. Investigate the effect of setting the TP degree to 1, 2 and 4 with all other parameters set to their default values. The relevant parameter to be changed in [qwen3_pretrain_override.yaml](./qwen3_pretrain_override.yaml) is `tensor_model_parallel_size`. Ensure to set the world size to the same value as the TP degree. Explain the observed trends in throughput and memory utilization per GPU.

---Answer Begin---

| TP Degree | Step time (s) | Throughput per GPU (TFLOP/s/GPU) | Memory per GPU (GB) |
| --- | --- | --- | --- |
| 1 | 3.34 | 245.69 | 73.65 |
| 2 | 3.84 | 106.72 | 37.03 |
| 4 | 3.86 | 53.21 | 18.61 |

The throughput per GPU decreases by at least 50% each time the TP degree is doubled. This is a direct consequence of the communication cost associated with TP, since the communication operations cannot be effectively overlapped with the forward and backward passes. 

The memory allocated per GPU decreases by approximately 50% each time the TP degree is doubled. This is because the model parameters and activations are sharded across increasing number of TP ranks. 


---Answer End---

2. Repeat the runs in question 1 with sequence parallelism enabled for the cases TP=2 and 4 only. How do the throughput and memory utilization compare with the corresponding runs in question 1? Use the PyTorch profiler to explain the observed changes in the throughput between the TP only and TP + SP cases. 

---Answer Begin---

| TP Degree | Step time (s) | Throughput per GPU (TFLOP/s/GPU) | Memory per GPU (GB) |
| --- | --- | --- | --- |
| 2 | 4.21 | 97.48 | 37.03 |
| 4 | 4.34 | 47.22 | 18.63 |

Enabling sequence parallelism reduces the throughput slightly as compared to using TP only. The PyTorch profiling trace shows that there is better overlap of the all-reduce communication operations with computation for the TP only case. When using TP in combination with SP, there are significant segments of the profiler trace where the communication blocks the computation, especially for the all-gather operation. 

The memory utilization remains unchanged since the activations only account for a small proportion of the VRAM utilization, since a short sequence length of 1024 was used.

---Answer End---

3. Is it feasible to run the training on a single GPU for a long sequence length of 16384? How does the use of TP help in such cases?

---Answer Begin---

One encounters a CUDA Out of Memory error if the training is performed on a single GPU. At long sequence lengths, the activations account for a significant proportion of the vRAM usage. Even sharding the activations across just two GPUs using TP = 2 is sufficient to make the training feasible.

---Answer End---

# Context Parallelism

## Theory

Read the section entitled "Context Parallelism" in the [Nanotron UltraScale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=context_parallelism).

Answer the following questions:
