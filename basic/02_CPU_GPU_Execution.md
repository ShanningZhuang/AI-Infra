# CPU-GPU Execution Model

## Overview

How CPU and GPU interact: kernel launching, streams, async execution, and CUDA Graphs.

## Learning Objectives

- [x] How CPU launches kernels to GPU (command queue, streams)
- [x] Kernel launch overhead (~5-10μs per launch)
- [x] Asynchronous execution and synchronization
- [x] CUDA Streams (concurrent kernel execution)
- [x] CUDA Graphs (capture & replay to reduce launch overhead)

## Resources

### Papers

-

### Blogs & Tutorials

- [NVIDIA CUDA Programming Guide - Streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)
- [NVIDIA CUDA Graphs Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)

### Videos

-

### Code References

-

---

## Notes

### CPU-GPU Interaction Model

```
CPU (Host)                    GPU (Device)
    │                              │
    │── Launch Kernel ──────────▶  │
    │   (~5-10μs overhead)         │ [Execute Kernel]
    │                              │
    │── Launch Kernel ──────────▶  │
    │                              │ [Execute Kernel]
    │                              │
```

### CUDA Streams

Streams allow concurrent kernel execution:

```
Stream 0: [Kernel A] ────────────────────
Stream 1:        [Kernel B] ─────────────
Stream 2:             [Kernel C] ────────
```

### CUDA Graphs

Captures a sequence of operations, replays with single launch:

```
Without Graph:           With Graph:
CPU: launch → launch     CPU: launch_graph (once)
        ↓        ↓              ↓
GPU:  [K1]     [K2]      GPU: [K1][K2][K3]
```

