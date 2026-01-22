# Collective Communication

> Parent: [Distributed Training](00_Distributed.md)

## Overview

Collective communication operations are the building blocks of distributed training. Understanding when and how data moves between GPUs is essential for optimizing training throughput.

![Collective Operations](/AI_Infra/images/collective_operation.png)

---

## Learning Objectives

- [ ] NCCL and collective operations
- [ ] AllReduce, AllGather, ReduceScatter, All-to-All
- [ ] Communication-computation overlap
- [ ] Ring AllReduce algorithm
- [ ] Bandwidth analysis and optimization

---

## NCCL (NVIDIA Collective Communications Library)

NCCL is the de facto standard for multi-GPU communication in deep learning.

```python
import torch.distributed as dist

# Initialize process group
dist.init_process_group(backend="nccl")

# Basic collective operations
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
dist.all_gather(output_list, tensor)
dist.reduce_scatter(output, input_list)
```

---

## Collective Operations

### AllReduce

Reduce values across all ranks and broadcast result to all ranks.

```
Before:                     After (SUM):
GPU 0: [1, 2]              GPU 0: [10, 20]
GPU 1: [3, 4]    ──────►   GPU 1: [10, 20]
GPU 2: [2, 6]              GPU 2: [10, 20]
GPU 3: [4, 8]              GPU 3: [10, 20]
```

**Use case**: Synchronizing gradients in Data Parallel training

### AllGather

Gather tensors from all ranks to all ranks.

```
Before:                     After:
GPU 0: [A]                 GPU 0: [A, B, C, D]
GPU 1: [B]      ──────►    GPU 1: [A, B, C, D]
GPU 2: [C]                 GPU 2: [A, B, C, D]
GPU 3: [D]                 GPU 3: [A, B, C, D]
```

**Use case**: Gathering sharded parameters in FSDP/ZeRO-3 before forward pass

### ReduceScatter

Reduce and scatter: opposite of AllGather.

```
Before:                     After (SUM):
GPU 0: [1,2,3,4]           GPU 0: [10]  (sum of all [0])
GPU 1: [2,3,4,5]  ──────►  GPU 1: [14]  (sum of all [1])
GPU 2: [3,4,5,6]           GPU 2: [18]  (sum of all [2])
GPU 3: [4,5,6,7]           GPU 3: [22]  (sum of all [3])
```

**Use case**: Scattering reduced gradients in FSDP/ZeRO-2

### All-to-All

Each rank sends different data to each other rank.

```
Before:                     After:
GPU 0: [A0,A1,A2,A3]       GPU 0: [A0,B0,C0,D0]
GPU 1: [B0,B1,B2,B3]  ──►  GPU 1: [A1,B1,C1,D1]
GPU 2: [C0,C1,C2,C3]       GPU 2: [A2,B2,C2,D2]
GPU 3: [D0,D1,D2,D3]       GPU 3: [A3,B3,C3,D3]
```

**Use case**: Token routing in MoE (Mixture of Experts) models

---

## Ring AllReduce Algorithm

AllReduce can be decomposed into ReduceScatter + AllGather, implemented efficiently as a ring:

```
Step 1-3: ReduceScatter (each GPU gets 1/N of the reduced result)
┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
│GPU 0│───►│GPU 1│───►│GPU 2│───►│GPU 3│───┐
└─────┘    └─────┘    └─────┘    └─────┘   │
    ▲                                       │
    └───────────────────────────────────────┘

Step 4-6: AllGather (broadcast reduced chunks to all)
Same ring direction, different data
```

**Bandwidth analysis**:
- Data per GPU: `D` bytes
- Steps: `2(N-1)` for N GPUs
- Data transferred per step: `D/N`
- Total per GPU: `2(N-1) × D/N ≈ 2D` (independent of N!)

This is why ring AllReduce scales well.

---

## Hardware Topology Matters

### Intra-Node (NVLink/NVSwitch)

```
DGX A100: 8 GPUs fully connected via NVSwitch
Bandwidth: 600 GB/s per GPU

    GPU0 ──── GPU1
     │ ╲    ╱ │
     │  ╲  ╱  │
     │   ╲╱   │
     │   ╱╲   │
     │  ╱  ╲  │
     │ ╱    ╲ │
    GPU2 ──── GPU3
```

### Inter-Node (InfiniBand/RoCE)

```
Node 0 ◄─── InfiniBand (400 Gb/s = 50 GB/s) ───► Node 1
         Much slower than NVLink!
```

**Implication**: Place Tensor Parallel within node (high communication), Data/Pipeline Parallel across nodes (lower communication).

---

## Communication-Computation Overlap

### Gradient Bucketing

Group small gradients into larger buckets to amortize launch overhead:

```python
# PyTorch DDP does this automatically
# Bucket size is tunable (default: 25MB)
model = DistributedDataParallel(
    model,
    bucket_cap_mb=25,
)
```

### Async Communication

Start AllReduce while still computing:

```
Layer N:     [Backward]────────────────►
Layer N-1:              [Backward]─────►[AllReduce]
Layer N-2:                       [Backward]──►[AllReduce]

Time ───────────────────────────────────────────────────►
```

```python
# Manual async collective
handle = dist.all_reduce(tensor, async_op=True)
# ... do other computation ...
handle.wait()  # Block until complete
```

---

## Bandwidth Analysis

### Theoretical vs Achieved

| Interconnect | Theoretical | Achieved (typical) |
|--------------|-------------|-------------------|
| NVLink (A100) | 600 GB/s | 500+ GB/s |
| InfiniBand HDR | 200 Gb/s = 25 GB/s | 20-22 GB/s |
| PCIe 4.0 x16 | 32 GB/s | 25 GB/s |

### Communication Time Formula

```
T_comm = latency + (data_size / bandwidth)
```

For large messages, bandwidth dominates. For small messages, latency dominates.

---

## Key Papers & Resources

| Resource | Topic |
|----------|-------|
| [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/) | Official NCCL guide |
| [Bringing HPC to PyTorch](https://arxiv.org/abs/2006.15704) | PyTorch distributed design |
| [ZeRO++](https://arxiv.org/abs/2306.10209) | Communication optimization for ZeRO |

---

## Notes

<!-- Add your learning notes here -->
