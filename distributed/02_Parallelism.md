# Parallelism Strategies

> Parent: [Distributed Training](00_Distributed.md)

## Overview

Different parallelism strategies for training large models: Data, Tensor, Pipeline, Sequence, Expert.

## Learning Objectives

- [x] Data Parallelism (DDP, FSDP)
- [x] Tensor Parallelism (Megatron-style)
- [x] Pipeline Parallelism
- [x] Sequence Parallelism
- [x] Expert Parallelism (MoE)

## Resources

### Papers

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [Megatron-LM v2: Efficient Large-Scale Training](https://arxiv.org/abs/2104.04473) - Introduces sequence parallelism
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
- [Switch Transformer](https://arxiv.org/abs/2101.03961) - Expert parallelism for MoE
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) - Production 5D parallelism

### Blogs & Tutorials

**- [Nanotron Ultrascale Playbook](https://nanotron-ultrascale-playbook.static.hf.space) - Best interactive guide for 5D parallelism**
- [HuggingFace Parallelism Guide](https://huggingface.co/docs/transformers/parallelism)
- [DeepSpeed ZeRO Tutorial](https://www.deepspeed.ai/tutorials/zero/)

### Code References

- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

---

## Notes

### 5D Parallelism Overview

5D Parallelism combines five dimensions to train models at extreme scale (100B+ parameters):

| Dimension | What is Split | Communication | Bandwidth | Use Case |
|-----------|---------------|---------------|-----------|----------|
| Data (DP) | Batches | AllReduce gradients | Low | Always; baseline |
| Tensor (TP) | Weight matrices | AllReduce/AllGather activations | **High** | Large layers |
| Pipeline (PP) | Model layers | Point-to-point | Medium | Very deep models |
| Sequence (SP) | Sequence dimension | AllGather/ReduceScatter | Medium-High | Long sequences |
| Expert (EP) | MoE experts | All-to-All | High | MoE models |

### Hardware Mapping

```
Intra-Node (NVLink):   TP, SP   ← Need high bandwidth
Inter-Node (IB):       DP, PP   ← Tolerate lower bandwidth
Both:                  EP       ← Depends on expert count
```

### Data Parallelism

```
GPU 0: Model copy + Batch 0 ──┐
GPU 1: Model copy + Batch 1 ──┼── AllReduce Gradients ── Update
GPU 2: Model copy + Batch 2 ──┤
GPU 3: Model copy + Batch 3 ──┘
```

### Tensor Parallelism (Column-wise)

```
          ┌─────────┐
Input ───▶│ W[:,0:n]│──▶ Output_0 ──┐
          └─────────┘               │
          ┌─────────┐               ├── Concat/AllReduce
Input ───▶│ W[:,n:] │──▶ Output_1 ──┘
          └─────────┘
```

