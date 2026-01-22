# Distributed Training

> Parent: [AI Infrastructure](../00_AI_Infra.md)

## Overview

Distributed training enables training models that exceed single-GPU memory capacity and accelerates training through parallelization. Modern LLMs (e.g., GPT-4, Llama 3, DeepSeek-V3) require combining multiple parallelism strategies across thousands of GPUs.

```
┌──────────────────────────────────────────────────────────────────┐
│                    Distributed Training Stack                    │
├──────────────────────────────────────────────────────────────────┤
│  Training Systems    │ DeepSpeed, Megatron-LM, FSDP, ColossalAI  │
├──────────────────────────────────────────────────────────────────┤
│  Parallelism         │ Data, Tensor, Pipeline, Sequence, Expert  │
├──────────────────────────────────────────────────────────────────┤
│  Communication       │ NCCL, AllReduce, AllGather, All-to-All    │
├──────────────────────────────────────────────────────────────────┤
│  Hardware            │ NVLink, NVSwitch, InfiniBand, RoCE        │
└──────────────────────────────────────────────────────────────────┘
```

---

## Topics

| Document | Description |
|----------|-------------|
| [Parallelism Strategies](Parallelism.md) | 5D Parallelism: Data, Tensor, Pipeline, Sequence, Expert |
| [Communication](communication.md) | NCCL, collective operations, overlap strategies |
| [Training Systems](Training_Systems.md) | DeepSpeed ZeRO, Megatron-LM, mixed precision |

---

## Core Concepts

### Why Distributed Training?

Two fundamental bottlenecks drive the need for distributed training:

1. **Memory Capacity**: Model parameters + optimizer states + activations + gradients exceed single GPU memory
2. **Compute Time**: Training on billions of tokens requires parallelization for reasonable wall-clock time

### Memory Analysis for a Transformer

For a model with `P` parameters in mixed precision training:

| Component | Memory (bytes) |
|-----------|----------------|
| Parameters (FP16) | 2P |
| Gradients (FP16) | 2P |
| Optimizer states (Adam FP32) | 12P |
| Activations | O(batch × seq × hidden × layers) |
| **Total (excluding activations)** | **16P** |

Example: 70B parameter model → **1.12 TB** just for parameters/optimizer/gradients

### The 5D Parallelism Framework

Modern training combines multiple parallelism dimensions:

```
                        ┌─────────────────┐
                        │  Expert Parallel│  (MoE only)
                        │     (EP)        │
                        └────────┬────────┘
                                 │
     ┌───────────────────────────┼───────────────────────────┐
     │                           │                           │
┌────▼────┐              ┌───────▼───────┐            ┌──────▼──────┐
│  Data   │              │    Tensor     │            │  Pipeline   │
│Parallel │              │   Parallel    │            │  Parallel   │
│  (DP)   │              │    (TP)       │            │    (PP)     │
└────┬────┘              └───────┬───────┘            └──────┬──────┘
     │                           │                           │
     └───────────────────────────┼───────────────────────────┘
                                 │
                        ┌────────▼────────┐
                        │    Sequence     │
                        │    Parallel     │
                        │      (SP)       │
                        └─────────────────┘
```

| Parallelism | What is Split | Communication Pattern | When to Use |
|-------------|---------------|----------------------|-------------|
| **Data (DP)** | Input batches | AllReduce gradients | Always; baseline |
| **Tensor (TP)** | Weight matrices | AllReduce/AllGather activations | Large hidden dimensions; within node |
| **Pipeline (PP)** | Model layers | Point-to-point (P2P) | Very deep models; across nodes |
| **Sequence (SP)** | Sequence dimension | AllGather/ReduceScatter | Long context; memory for activations |
| **Expert (EP)** | MoE experts | All-to-All | MoE architectures |

### Communication-Computation Tradeoff

Key insight: **Minimize communication, maximize computation overlap**

```
Ideal: ████████████████████████  (compute)
           ░░░░░░░░░░░░░░░░░░░   (communication, overlapped)

Reality: ████████░░░░░████████░░░░░████████
         compute  comm  compute  comm  compute
```

Strategies:
- **Gradient bucketing**: Batch small gradients to amortize communication overhead
- **Async AllReduce**: Start communication while still computing gradients for earlier layers
- **Activation recomputation**: Trade compute for memory, reduce communication of activations

---

## Hierarchy: How Parallelism Strategies Layer

### Intra-Node vs Inter-Node

```
┌─────────────────────── Node 0 ───────────────────────┐
│  GPU0 ◄──NVLink──► GPU1 ◄──NVLink──► GPU2 ◄──► GPU3 │
│    │                                           │     │
│    └──────────── Tensor Parallel ──────────────┘     │
└──────────────────────────────────────────────────────┘
                          │
                     InfiniBand (slower)
                          │
┌─────────────────────── Node 1 ───────────────────────┐
│  GPU4 ◄──NVLink──► GPU5 ◄──NVLink──► GPU6 ◄──► GPU7 │
└──────────────────────────────────────────────────────┘

Inter-node: Data Parallel or Pipeline Parallel (less communication)
Intra-node: Tensor Parallel (high bandwidth needed)
```

### Typical Configuration (Example: 64 GPUs, 8 nodes)

```python
# Megatron-LM style configuration
world_size = 64                    # Total GPUs
tensor_parallel_size = 8           # Within each node (NVLink)
pipeline_parallel_size = 4         # Across 4 groups of 2 nodes
data_parallel_size = 2             # Remaining parallelism

# Constraint: TP × PP × DP = world_size
assert 8 * 4 * 2 == 64
```

---

## Memory Optimization Stack

### ZeRO (Zero Redundancy Optimizer)

Progressive sharding of training state across data parallel ranks:

```
┌────────────────────────────────────────────────────────────┐
│                    Memory per GPU                          │
├────────────────────────────────────────────────────────────┤
│ Baseline:    [Params][Grads][Opt States]  = 16P bytes     │
├────────────────────────────────────────────────────────────┤
│ ZeRO-1:      [Params][Grads][Opt/N]       = 4P + 12P/N    │
├────────────────────────────────────────────────────────────┤
│ ZeRO-2:      [Params][Grads/N][Opt/N]     = 2P + 14P/N    │
├────────────────────────────────────────────────────────────┤
│ ZeRO-3:      [Params/N][Grads/N][Opt/N]   = 16P/N         │
└────────────────────────────────────────────────────────────┘
                            N = number of data parallel ranks
```

### FSDP (Fully Sharded Data Parallel)

PyTorch-native implementation of ZeRO-3:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
    cpu_offload=CPUOffload(offload_params=True),    # Optional
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    ),
)
```

---

## Key Papers

| Paper | Contribution | Link |
|-------|--------------|------|
| Megatron-LM (2019) | Tensor + Pipeline parallelism | [arXiv:1909.08053](https://arxiv.org/abs/1909.08053) |
| ZeRO (2019) | Memory-efficient data parallelism | [arXiv:1910.02054](https://arxiv.org/abs/1910.02054) |
| GPipe (2019) | Pipeline parallelism with micro-batches | [arXiv:1811.06965](https://arxiv.org/abs/1811.06965) |
| Megatron-LM v2 (2021) | Sequence parallelism + selective recompute | [arXiv:2104.04473](https://arxiv.org/abs/2104.04473) |
| DeepSpeed ZeRO++ (2023) | Communication optimization | [arXiv:2306.10209](https://arxiv.org/abs/2306.10209) |
| Ring Attention (2023) | Blockwise attention for infinite context | [arXiv:2310.01889](https://arxiv.org/abs/2310.01889) |

---

## Code References

| Codebase | Focus | Link |
|----------|-------|------|
| Megatron-LM | NVIDIA's reference for TP/PP/SP | [GitHub](https://github.com/NVIDIA/Megatron-LM) |
| DeepSpeed | Microsoft's ZeRO + optimizations | [GitHub](https://github.com/microsoft/DeepSpeed) |
| PyTorch FSDP | Native sharded training | [Docs](https://pytorch.org/docs/stable/fsdp.html) |
| ColossalAI | Unified parallelism framework | [GitHub](https://github.com/hpcaitech/ColossalAI) |
| Nanotron | Clean implementation for learning | [GitHub](https://github.com/huggingface/nanotron) |

---

## Further Reading

- [Nanotron Ultrascale Playbook](https://nanotron-ultrascale-playbook.static.hf.space) - Interactive guide to distributed training
- [HuggingFace Parallelism Guide](https://huggingface.co/docs/transformers/parallelism) - Practical tutorial
- [DeepSpeed Tutorials](https://www.deepspeed.ai/tutorials/) - ZeRO and optimization techniques
