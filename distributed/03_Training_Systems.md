# Large-Scale Training Systems

> Parent: [Distributed Training](00_Distributed.md)

## Overview

Systems for training large models: DeepSpeed, Megatron-LM, mixed precision, gradient checkpointing.

## Learning Objectives

- [x] DeepSpeed ZeRO stages (1, 2, 3) (know but not code)
- [x] Megatron-LM architecture (know but not code)
- [x] Mixed precision training (FP16, BF16, FP8) (know but not code)
- [x] Gradient checkpointing (Not so familiar and no code reference)
- [x] ZMQ sockets and pipes. IPC? (Know the basic knowledge but not dive into the code)

## Resources

### Papers

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [Megatron-LM: Training Multi-Billion Parameter Language Models](https://arxiv.org/abs/1909.08053)
- [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)
- [Mixed Precision Training (Micikevicius et al.)](https://arxiv.org/abs/1710.03740)

### Blogs & Tutorials

- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)

### Videos

-

### Code References

-

---

## Notes

### DeepSpeed ZeRO Stages

| Stage | What is partitioned | Memory Saving |
|-------|---------------------|---------------|
| ZeRO-1 | Optimizer states | ~4x |
| ZeRO-2 | + Gradients | ~8x |
| ZeRO-3 | + Parameters | ~Nx (N = #GPUs) |

### Mixed Precision

```
Forward:  FP16/BF16 (fast, memory efficient)
Backward: FP16/BF16
Weights:  FP32 master copy (for precision)
```

### Gradient Checkpointing

Trade compute for memory:
```
Normal:      Save all activations ─────────▶ High memory
Checkpoint:  Save only checkpoints, recompute ─▶ Low memory
```

