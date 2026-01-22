# AI Infrastructure Learning Path

> Goal: Systematic learning for AI Infra internship at Qwen (Agent Infra focus)
> Time commitment: ~1 hour/day

## Overview: The AI Infra Stack (Bottom to Top)

```
┌─────────────────────────────────────────────┐
│           Agent Infrastructure              │  ← Your focus area
├─────────────────────────────────────────────┤
│         Serving & Inference                 │
├─────────────────────────────────────────────┤
│          RL & Alignment (RLHF)              │
├─────────────────────────────────────────────┤
│      Distributed Training Systems           │
├─────────────────────────────────────────────┤
│      Frameworks (PyTorch/JAX internals)     │
├─────────────────────────────────────────────┤
│       Compute Optimization (Kernels)        │
├─────────────────────────────────────────────┤
│         CUDA / GPU Programming              │
├─────────────────────────────────────────────┤
│        Hardware (GPU Architecture)          │
└─────────────────────────────────────────────┘
```

---

## Phase 1: Foundations (Weeks 1-2)

### 1.1 GPU Architecture Basics → [Notes](basic/GPU_Fundamentals.md)
- [x] NVIDIA GPU architecture overview (SM, warp, memory hierarchy)
- [x] Understanding GPU memory types: Global, Shared, L1/L2 cache, Registers
- [x] Memory bandwidth vs compute bound problems

### 1.2 CPU-GPU Execution Model → [Notes](basic/CPU_GPU_Execution.md)
- [x] How CPU launches kernels to GPU (command queue, streams)
- [x] Kernel launch overhead (~5-10μs per launch)
- [x] Asynchronous execution and synchronization
- [x] CUDA Streams (concurrent kernel execution)
- [x] **CUDA Graphs** (capture & replay to reduce launch overhead)

### 1.3 CUDA Programming → [Notes](basic/GPU_Fundamentals.md) | [Advanced](basic/CUDA_Advanced.md)
- [x] CUDA programming model: grid, block, thread
- [x] Basic kernel writing and launching
- [x] Memory coalescing and bank conflicts concepts

---

## Phase 2: Compute Optimization (Weeks 3-4)

### 2.1 Matrix Multiplication Optimization → [Notes](basic/matrix_multiplication.md)
- [x] Naive → Tiled → Optimized matmul progression
- [x] Understanding roofline model
- [x] cuBLAS and how it achieves peak performance

### 2.2 Flash Attention → [Notes](Flash_Attention.md) | [Implementation](basic/FlashAttention_CS336.md)
- [x] Standard attention memory bottleneck
- [x] Flash Attention algorithm (tiling, recomputation)
- [x] Flash Attention 2 & 3 improvements

### 2.3 Kernel Fusion & Triton → [Notes](basic/Triton_Programming.md)
- [x] Why kernel fusion matters (memory bandwidth)
- [x] Triton programming basics
- [x] How PyTorch 2.0 `torch.compile` uses Triton

---

## Phase 3: Distributed Training (Weeks 5-6)

> **Node**: [Distributed Training Overview](distributed/distributed.md)

### 3.1 Parallelism Strategies → [Notes](distributed/Parallelism.md)
- [x] Data Parallelism (DDP, FSDP)
- [x] Tensor Parallelism (Megatron-style)
- [x] Pipeline Parallelism
- [x] Sequence Parallelism
- [x] Expert Parallelism (MoE)

### 3.2 Communication → [Notes](distributed/communication.md)
- [x] NCCL and collective operations (AllReduce, AllGather, ReduceScatter)
- [x] Communication-computation overlap
- [x] Ring AllReduce algorithm
- [x] ZMQ sockets and pipes 

### 3.3 Large-Scale Training Systems → [Notes](distributed/Training_Systems.md)
- [x] DeepSpeed ZeRO stages (1, 2, 3)
- [x] Megatron-LM architecture
- [x] Mixed precision training (FP16, BF16, FP8)
- [x] Gradient checkpointing

---

## Phase 4: RL & Alignment (Weeks 7-8)

> **Overview**: [RL & Alignment Overview](rl/rl.md)

### 4.1 Alignment Algorithms → [Notes](rl/01_Algorithms.md)
- [ ] PPO (RLHF) - reward model, value model, clipped objective
- [ ] DPO - direct preference, no reward model
- [ ] DPO variants (IPO, KTO, ORPO, SimPO)
- [ ] GRPO - group normalization, no value model (DeepSeek)
- [ ] RLOO - leave-one-out baseline
- [ ] GDPO - distributional preferences (ICLR 2025)

### 4.2 RL Infrastructure → [Notes](rl/02_Infrastructure.md)
- [ ] Distributed PPO training
- [ ] Actor-critic parallelism
- [ ] Experience buffer management
- [ ] Rollout generation at scale

### 4.3 RL Frameworks → [Notes](rl/03_Frameworks.md)
- [ ] TRL (Transformer Reinforcement Learning)
- [ ] OpenRLHF
- [ ] DeepSpeed-Chat
- [ ] veRL

---

## Phase 5: Inference & Serving (Weeks 9-10)

> **Overview**: [Inference Overview](inference/inference.md)

### 5.1 Inference Optimization
- [x] KV Cache mechanism and memory management → [Notes](inference/01_KV_Cache.md)
- [x] PagedAttention (vLLM) → [Notes](inference/01_KV_Cache.md)
- [x] Continuous batching → [Notes](inference/02_Batching.md)
- [x] Speculative decoding → [Notes](inference/03_Speculative_Decoding.md)
- [ ] Quantization (INT8, INT4, GPTQ, AWQ) → [Notes](inference/04_Quantization.md) (Not yet, becaude this part is actually not so commonly used)
- [ ] CUDA Graphs in practice → [Notes](inference/06_Serving.md) (see 1.2 for basics)

### 5.2 Inference Frameworks → [Notes](inference/05_Frameworks.md)
- [x] vLLM architecture
- [x] TensorRT-LLM
- [x] SGLang (particularly relevant for agents)

### 5.3 Serving Infrastructure → [Notes](inference/06_Serving.md)
- [ ] Model serving patterns (online vs batch)
- [ ] Load balancing and routing strategies
- [ ] Request scheduling and prioritization

---

## Phase 6: Agent Infrastructure (Weeks 11-14) ← Core Focus

> **Overview**: [Agent Infrastructure Overview](agent/agent.md)

### 6.1 Agent Fundamentals → [Notes](agent/01_Fundamentals.md)
- [ ] ReAct pattern and tool use
- [ ] Agent memory systems (short-term, long-term, episodic)
- [ ] Planning and reasoning frameworks (CoT, ToT, GoT)
- [ ] Prompt engineering for agents

### 6.2 Agent Execution Infrastructure → [Notes](agent/02_Execution.md)
- [ ] Orchestration frameworks (LangChain internals, LlamaIndex)
- [ ] Tool calling protocols and function calling
- [ ] Parallel tool execution
- [ ] Error handling and retry strategies

### 6.3 Multi-Agent Systems → [Notes](agent/03_Multi_Agent.md)
- [ ] Agent communication patterns
- [ ] Coordination and consensus
- [ ] Multi-agent frameworks (AutoGen, CrewAI, LangGraph)

### 6.4 Agent Serving → [Notes](agent/04_Serving.md)
- [ ] Session management
- [ ] Streaming responses for agents
- [ ] Long-context handling and compression
- [ ] State persistence strategies

### 6.5 Production Considerations → [Notes](agent/05_Production.md)
- [ ] Observability and tracing for agents
- [ ] Evaluation and benchmarking
- [ ] Safety guardrails and sandboxing
- [ ] Cost monitoring and optimization

---

## Suggested Weekly Schedule

| Day | Focus |
|-----|-------|
| Mon-Tue | Core reading/paper |
| Wed-Thu | Hands-on coding/experiments |
| Fri | Review + notes organization |
| Weekend | Optional deep-dive or catch-up |

---

## Priority Matrix (Given Agent Infra Focus)

| Topic | Priority | Depth |
|-------|----------|-------|
| Agent Infrastructure | ★★★★★ | Deep |
| Inference/Serving | ★★★★☆ | Medium-Deep |
| RL & Alignment | ★★★★☆ | Medium-Deep |
| Distributed Training | ★★★☆☆ | Conceptual |
| Flash Attention/Kernels | ★★★☆☆ | Conceptual |
| CUDA/GPU Basics | ★★☆☆☆ | Light |

---

## Key Papers to Read

1. **Attention**: "Attention Is All You Need", Flash Attention 1/2
2. **Distributed**: Megatron-LM, DeepSpeed ZeRO
3. **RL/Alignment**: InstructGPT (RLHF), DPO, GRPO (DeepSeek-R1)
4. **Inference**: vLLM (PagedAttention), SGLang
5. **Agents**: ReAct, Toolformer, "Language Agents" survey papers

---

## Practical Projects (Optional but Recommended)

1. Implement a basic Flash Attention kernel (simplified)
2. Set up distributed training with FSDP
3. Deploy a model with vLLM and benchmark
4. Build a simple agent with tool use and memory