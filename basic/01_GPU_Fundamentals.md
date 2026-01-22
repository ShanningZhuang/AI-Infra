# GPU Fundamentals

## Overview

Understanding NVIDIA GPU architecture: SMs, warps, threads, and memory hierarchy.

## Learning Objectives

- [x] NVIDIA GPU architecture overview (SM, warp, memory hierarchy)
- [x] Understanding GPU memory types: Global, Shared, L1/L2 cache, Registers
- [x] Memory bandwidth vs compute bound problems
- [x] CUDA programming model: grid, block, thread

---

## GPU Architecture

### Streaming Multiprocessors (SMs)

A GPU consists of many **Streaming Multiprocessors (SMs)**. Each SM contains:
- CUDA cores (for computation)
- Shared memory / L1 cache
- Register file
- Warp schedulers

```
GPU
├── SM 0
│   ├── CUDA Cores
│   ├── Shared Memory (48-164 KB)
│   ├── Registers (~256 KB)
│   └── Warp Schedulers
├── SM 1
│   └── ...
├── ...
└── SM N (e.g., 108 SMs on A100)
    └── ...

Shared across all SMs:
├── L2 Cache (~40 MB)
└── Global Memory / HBM (40-80 GB)
```

---

## Thread Hierarchy

### Programming Model: Grid → Block → Thread

```
Grid (launched by kernel)
├── Block (0,0)
│   ├── Thread (0,0)
│   ├── Thread (0,1)
│   └── ...
├── Block (0,1)
│   └── ...
├── Block (1,0)
│   └── ...
└── ...
```

### Key Built-in Variables

| Variable | Description |
|----------|-------------|
| `threadIdx` | Index of thread within its block (0 to blockDim-1) |
| `blockIdx` | Index of block within the grid |
| `blockDim` | Dimensions of the thread block |
| `gridDim` | Dimensions of the grid |

### Example: Computing Global Thread Index

```cuda
// For a 1D grid of 1D blocks:
int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

// For a 2D grid:
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

### Warps

A **warp** is a group of 32 threads that execute in lockstep (SIMT - Single Instruction Multiple Thread).

- All threads in a warp execute the same instruction
- Divergence (different branches) causes serialization
- Warp is the basic scheduling unit

```
Block (256 threads)
├── Warp 0: threads 0-31
├── Warp 1: threads 32-63
├── Warp 2: threads 64-95
├── ...
└── Warp 7: threads 224-255
```

---

## Memory Hierarchy

### Overview

| Memory Type | Scope | Speed | Size | Lifetime |
|-------------|-------|-------|------|----------|
| Registers | Thread | Fastest (~1 cycle) | ~256KB/SM | Thread |
| Shared Memory | Block | Fast (~20-30 cycles) | 48-164KB/SM | Block |
| L1 Cache | SM | Fast | Configurable | Automatic |
| L2 Cache | Device | Medium (~200 cycles) | ~40MB | Automatic |
| Global Memory (HBM) | Device | Slow (~400-600 cycles) | 40-80GB | Application |

### Memory Access Pattern

```
Thread registers (fastest)
        ↓
Shared memory (block-level)
        ↓
L1 Cache (SM-level)
        ↓
L2 Cache (device-level)
        ↓
Global Memory / HBM (slowest)
```

### Key Concepts

**Coalesced Memory Access**: When threads in a warp access consecutive memory addresses, the hardware can combine requests into fewer transactions.

```
Good (coalesced):     Thread 0→addr[0], Thread 1→addr[1], Thread 2→addr[2]...
                      → 1 memory transaction for 32 threads

Bad (strided):        Thread 0→addr[0], Thread 1→addr[128], Thread 2→addr[256]...
                      → Many memory transactions, wasted bandwidth
```

**Bank Conflicts**: Shared memory is divided into 32 banks. If multiple threads access the same bank, accesses are serialized.

---

## Compute vs Memory Bound

### Arithmetic Intensity

```
Arithmetic Intensity = FLOPs / Bytes transferred
```

| Kernel Type | Arithmetic Intensity | Bottleneck | Optimization Focus |
|-------------|---------------------|------------|-------------------|
| Memory-bound | Low (<10) | Bandwidth | Reduce memory traffic, coalescing |
| Compute-bound | High (>10) | Compute | Increase ILP, use Tensor Cores |

### Roofline Model

```
                    Peak Compute (FLOPS)
                   ────────────────────────
                  /
                 /
Performance     /
(FLOPS)        /
              /
             ──────────────────────────────
            Memory Bandwidth Ceiling
           /
          /
         /
        └───────────────────────────────────
             Arithmetic Intensity (FLOP/byte)
```

---

## Example: Parallel Reduction

A classic GPU pattern - summing an array using parallel reduction.

```
Initial shared memory (8 threads for simplicity):
tid:   0    1    2    3    4    5    6    7
     +----+----+----+----+----+----+----+----+
     | v0 | v1 | v2 | v3 | v4 | v5 | v6 | v7 |
     +----+----+----+----+----+----+----+----+

Step 1: offset=4 (threads 0-3 add from threads 4-7)
     +----+----+----+----+----+----+----+----+
     |v0+v4|v1+v5|v2+v6|v3+v7| v4 | v5 | v6 | v7 |
     +----+----+----+----+----+----+----+----+

Step 2: offset=2 (threads 0-1 add from threads 2-3)
     +--------+--------+--------+--------+
     |sum0123 |sum4567 | v2+v6  | v3+v7  |
     +--------+--------+--------+--------+

Step 3: offset=1 (thread 0 adds from thread 1)
     +------------------------+
     |     FINAL SUM          |
     | (v0+v1+v2+v3+v4+v5+v6+v7) |
     +------------------------+
         ↑ stored in shared[0]
```

This pattern reduces N elements in O(log N) steps.

---

## Resources

### Blogs & Tutorials

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- "How GPUs work" by NVIDIA

### Related Notes

- [CPU-GPU Execution Model](02_CPU_GPU_Execution.md) - Streams, CUDA Graphs
- [CUDA Advanced](03_CUDA_Advanced.md) - Warp primitives, PTX/SASS, Nsight profiling
