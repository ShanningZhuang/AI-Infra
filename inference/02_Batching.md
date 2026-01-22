# Batching Strategies for LLM Inference

> Parent: [Inference Overview](inference.md)

## Overview

Batching is crucial for LLM inference efficiency. Unlike training where batch sizes are fixed, inference must handle requests with variable lengths arriving at different times. The evolution from static to continuous batching represents one of the biggest improvements in LLM serving throughput.

## Learning Objectives

- [ ] Static batching limitations
- [ ] Continuous batching (iteration-level scheduling)
- [ ] Prefill-decode disaggregation
- [ ] Chunked prefill
- [ ] Scheduling algorithms

## Resources

### Papers

- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) - Introduced continuous batching
- [Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369)
- [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/abs/2311.18677)

### Blogs & Tutorials

- [Anyscale: How Continuous Batching Enables 23x Throughput](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [vLLM Scheduling Deep Dive](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html)

### Code References

- [vLLM Scheduler](https://github.com/vllm-project/vllm/blob/main/vllm/core/scheduler.py)
- [TGI Continuous Batching](https://github.com/huggingface/text-generation-inference)

---

## Notes

### Static Batching Problem

Traditional batching waits for all sequences in a batch to complete:

```
Time ──────────────────────────────────────────────────▶

Static Batch:
┌──────────────────────────────────────────────────────┐
│ Req 1: ████████████████████████████░░░░░░░░░░░░░░░░ │ Done early, waiting
│ Req 2: █████████████████████████████████████████████ │ Longest
│ Req 3: ███████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │ Done early, waiting
│ Req 4: ████████████████████████░░░░░░░░░░░░░░░░░░░░ │ Done early, waiting
└──────────────────────────────────────────────────────┘
                  ↑ GPU idle time ↑

All requests must wait for the longest one before new batch can start.
```

Problems:
- **Head-of-line blocking**: Fast requests wait for slow ones
- **Low GPU utilization**: GPU idles while waiting
- **High latency variance**: Short requests have unpredictable latency

### Continuous Batching (Iteration-Level Scheduling)

Process one iteration at a time, adding/removing requests dynamically:

```
Time ──────────────────────────────────────────────────▶

Continuous Batching:
┌──────────────────────────────────────────────────────┐
│ Req 1: ████████████████████████████                  │ Done → slot freed
│ Req 2: █████████████████████████████████████████████ │
│ Req 3: ███████████████                    ← Req 5 ██ │ Req 3 done, Req 5 joins
│ Req 4: ████████████████████████      ← Req 6 ██████ │ Req 4 done, Req 6 joins
└──────────────────────────────────────────────────────┘

GPU stays busy processing at every iteration!
```

Key insight: At each forward pass iteration, we can:
1. Remove completed sequences
2. Add new sequences (with their prefill)
3. Maintain near-100% GPU utilization

### The Two Phases Challenge

| Phase | Compute Pattern | Optimal Batch Size |
|-------|----------------|-------------------|
| Prefill | Matrix-matrix multiply | Smaller (compute-bound) |
| Decode | Matrix-vector multiply | Larger (memory-bound) |

Mixing prefill and decode in same batch creates inefficiency:
- Prefill wants smaller batches (limited by compute)
- Decode wants larger batches (hide memory latency)

### Chunked Prefill (Sarathi)

Split long prefills into chunks to mix with decode:

```
Without chunked prefill:
┌─────────────────────────────────────────┐
│ Long prefill (1000 tokens)              │ ← Blocks decode requests
│ Decode 1 │ Decode 2 │ Decode 3          │
└─────────────────────────────────────────┘

With chunked prefill (chunk_size=256):
┌─────────────────────────────────────────┐
│ Prefill[0:256] │ Decode 1 │ Decode 2    │
│ Prefill[256:512] │ Decode 1 │ Decode 2  │
│ Prefill[512:768] │ Decode 1 │ Decode 2  │
│ Prefill[768:1000] │ Decode 1 │ Decode 2 │
└─────────────────────────────────────────┘

Interleave prefill chunks with decode → better latency for decode requests
```

Benefits:
- Reduces latency spikes from long prefills
- Better resource utilization
- More predictable TPOT for decode

### Prefill-Decode Disaggregation (Splitwise)

Separate prefill and decode onto different GPUs:

```
┌─────────────────────┐     ┌─────────────────────┐
│   Prefill GPUs      │     │    Decode GPUs      │
│  (compute-heavy)    │────▶│   (memory-heavy)    │
│                     │ KV  │                     │
│ Optimized for large │cache│ Optimized for large │
│ matrix multiplies   │xfer │ batch sizes         │
└─────────────────────┘     └─────────────────────┘
```

#### Why Disaggregation Helps Despite Transfer Overhead

The key insight: **Prefill and decode have fundamentally opposite computational characteristics**.

```
Prefill Phase:                          Decode Phase:
┌─────────────────────────┐            ┌─────────────────────────┐
│  Q: [batch, seq, dim]   │            │  Q: [batch, 1, dim]     │
│  K: [batch, seq, dim]   │            │  K: [batch, seq, dim]   │
│  V: [batch, seq, dim]   │            │  V: [batch, seq, dim]   │
│                         │            │                         │
│  Matrix × Matrix        │            │  Matrix × Vector        │
│  High arithmetic        │            │  Low arithmetic         │
│  intensity              │            │  intensity              │
│                         │            │                         │
│  COMPUTE-BOUND          │            │  MEMORY-BOUND           │
└─────────────────────────┘            └─────────────────────────┘

Optimal: Small batch,                  Optimal: Large batch,
         high parallelism                       amortize memory access
```

#### The Problem with Mixed Batching

When prefill and decode share the same GPU:

```
Mixed Batch Iteration:
┌────────────────────────────────────────────────────────────────┐
│  Prefill Req A (1000 tokens) ──┐                               │
│  Decode Req B (1 token)   ─────┼──▶  Same GPU, same kernel     │
│  Decode Req C (1 token)   ─────┤                               │
│  Decode Req D (1 token)   ─────┘                               │
└────────────────────────────────────────────────────────────────┘

Problems:
1. Prefill dominates compute → decode requests starve (latency spike)
2. Can't optimize kernel for either workload (compromise)
3. Decode batch size limited by prefill memory usage
4. Unpredictable latency for decode when prefill arrives
```

#### The Transfer Cost Math

```
KV Cache Transfer Cost (one-time per request):
┌─────────────────────────────────────────────────────────────────┐
│ KV Cache Size = 2 × layers × hidden_dim × seq_len × dtype      │
│                                                                 │
│ Example (7B model, 2048 tokens, FP16):                         │
│   = 2 × 32 × 4096 × 2048 × 2 bytes                             │
│   = 1.07 GB per request                                         │
│                                                                 │
│ Transfer time by interconnect:                                  │
│   - NVLink (600 GB/s): ~1.8 ms                                 │
│   - PCIe 4.0 (64 GB/s): ~17 ms                                 │
│   - Network (100 Gbps): ~86 ms                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Why the Tradeoff is Worth It

The transfer is a **one-time cost**, but decode runs for **N iterations** (N = output tokens):

```
┌─────────────────────────────────────────────────────────────────┐
│ If output = 500 tokens, decode runs 500 iterations             │
│                                                                 │
│ Benefit per iteration from optimized batching:                 │
│   Mixed: 50 decode requests (limited by prefill memory)        │
│   Disaggregated: 200+ decode requests (full GPU memory)        │
│                                                                 │
│ Decode speedup: 2-4x throughput per iteration                  │
│ Total benefit: 500 iterations × speedup                        │
│                                                                 │
│ ∴ One-time transfer cost << Accumulated decode benefit         │
└─────────────────────────────────────────────────────────────────┘
```

| Factor | Mixed | Disaggregated |
|--------|-------|---------------|
| Prefill batch size | Limited by decode latency SLA | Can maximize (no decode waiting) |
| Decode batch size | Limited by prefill memory | Can maximize GPU memory |
| Kernel optimization | Compromise for both | Specialized per phase |
| Hardware utilization | Suboptimal for both | Near-optimal for each |
| **One-time cost** | None | KV cache transfer |
| **Per-iteration benefit** | None | 2-4x decode throughput |

#### When Disaggregation Makes Sense

```
Good scenarios:                        Bad scenarios:
┌─────────────────────────────┐       ┌─────────────────────────────┐
│ ✓ Long prompts (>1K tokens) │       │ ✗ Short prompts (<256)      │
│ ✓ Long outputs (>100 tokens)│       │ ✗ Very short outputs (<10)  │
│ ✓ High QPS (many requests)  │       │ ✗ Low QPS (few requests)    │
│ ✓ Fast interconnect (NVLink)│       │ ✗ Slow network only         │
│ ✓ Heterogeneous HW available│       │ ✗ Homogeneous cluster       │
└─────────────────────────────┘       └─────────────────────────────┘

Break-even point:
  Transfer_cost < N_decode_iterations × per_iteration_benefit

  With NVLink (~2ms transfer) and 4x decode speedup:
  Even with just 10 output tokens, disaggregation can be beneficial
```

#### Disaggregation Architecture

```
                    Load Balancer
                         │
           ┌─────────────┴─────────────┐
           ▼                           ▼
    ┌─────────────┐             ┌─────────────┐
    │  Prefill    │             │  Prefill    │
    │  Pool       │             │  Pool       │
    │  (H100s)    │             │  (H100s)    │
    └──────┬──────┘             └──────┬──────┘
           │ KV Cache                  │ KV Cache
           │ Transfer                  │ Transfer
           ▼                           ▼
    ┌─────────────┐             ┌─────────────┐
    │  Decode     │             │  Decode     │
    │  Pool       │             │  Pool       │
    │  (A100s)    │◀───────────▶│  (A100s)    │
    └─────────────┘  Load       └─────────────┘
                     Balance

Prefill Pool: Fewer GPUs, optimized for compute (tensor parallel)
Decode Pool: More GPUs, optimized for memory bandwidth (large batches)
```

#### Implementation Considerations

```python
class DisaggregatedScheduler:
    def __init__(self, prefill_workers, decode_workers):
        self.prefill_pool = prefill_workers  # Compute-optimized
        self.decode_pool = decode_workers    # Memory-optimized
        self.pending_transfers = {}

    def handle_request(self, request):
        # 1. Send to prefill pool
        prefill_worker = self.select_prefill_worker()
        kv_cache = prefill_worker.run_prefill(request.prompt)

        # 2. Transfer KV cache to decode pool
        decode_worker = self.select_decode_worker()
        transfer_start = time.time()
        decode_worker.receive_kv_cache(kv_cache)
        transfer_time = time.time() - transfer_start

        # 3. Decode runs on dedicated GPU with large batch
        return decode_worker.run_decode(request, kv_cache)

    def select_decode_worker(self):
        # Select worker with most available batch capacity
        # Large batches = better memory bandwidth utilization
        return max(self.decode_pool, key=lambda w: w.available_batch_slots)
```

### Scheduling Algorithms

**First-Come-First-Serve (FCFS):**
```python
def fcfs_schedule(waiting_queue, running_batch, max_batch_size):
    while len(running_batch) < max_batch_size and waiting_queue:
        running_batch.append(waiting_queue.pop(0))
    return running_batch
```

**Shortest-Job-First (SJF):**
```python
def sjf_schedule(waiting_queue, running_batch, max_batch_size):
    # Sort by expected output length (if known or estimated)
    waiting_queue.sort(key=lambda r: r.expected_output_len)
    while len(running_batch) < max_batch_size and waiting_queue:
        running_batch.append(waiting_queue.pop(0))
    return running_batch
```

**Priority-based:**
```python
def priority_schedule(waiting_queue, running_batch, max_batch_size):
    # Consider: wait time, priority level, preemption
    waiting_queue.sort(key=lambda r: (
        -r.priority,           # Higher priority first
        r.arrival_time,        # Then FCFS within priority
    ))
    while len(running_batch) < max_batch_size and waiting_queue:
        running_batch.append(waiting_queue.pop(0))
    return running_batch
```

### Preemption Strategies

When memory is full, must preempt running requests:

| Strategy | Description | Overhead |
|----------|-------------|----------|
| Swap | Move KV cache to CPU | High latency spike |
| Recompute | Discard KV, recompute later | Wasted compute |
| Kill | Terminate request | Lost work |

vLLM preemption order:
1. Try to swap to CPU
2. If CPU full, recompute shortest sequence
3. Only kill as last resort

### Batch Size Considerations

```
Throughput vs Batch Size:

Throughput │
    ↑      │              ┌──────────────────
           │             ╱
           │            ╱
           │           ╱
           │          ╱
           │         ╱   ← Memory-bound region
           │        ╱     (larger batch = better)
           │       ╱
           │      ╱
           │     ╱  ← Compute-bound region
           │    ╱     (diminishing returns)
           │   ╱
           └──┴────────────────────────────▶
                    Batch Size
```

Optimal batch size depends on:
- Model size (larger model = smaller optimal batch)
- Sequence length (longer = smaller batch)
- Hardware (more memory = larger batch possible)
- Latency requirements (SLA constraints)

### Practical Implementation

```python
class ContinuousBatchScheduler:
    def __init__(self, max_batch_tokens, max_batch_size):
        self.max_batch_tokens = max_batch_tokens
        self.max_batch_size = max_batch_size
        self.running = []
        self.waiting = []

    def step(self):
        # Remove completed sequences
        self.running = [r for r in self.running if not r.is_done()]

        # Calculate current batch token count
        current_tokens = sum(r.current_len() for r in self.running)

        # Add new requests if space available
        while self.waiting:
            candidate = self.waiting[0]
            new_tokens = current_tokens + candidate.prompt_len

            if (len(self.running) < self.max_batch_size and
                new_tokens <= self.max_batch_tokens):
                self.running.append(self.waiting.pop(0))
                current_tokens = new_tokens
            else:
                break

        return self.running

    def add_request(self, request):
        self.waiting.append(request)
```

### Key Metrics to Monitor

| Metric | Description | Target |
|--------|-------------|--------|
| Batch utilization | Actual vs max batch size | > 80% |
| Queue wait time | Time request waits before starting | < 100ms |
| Preemption rate | % requests preempted | < 5% |
| Throughput | Tokens generated per second | Maximize |
