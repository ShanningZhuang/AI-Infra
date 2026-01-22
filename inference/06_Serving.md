# Inference Serving Infrastructure

> Parent: [Inference Overview](inference.md)

## Overview

Serving LLMs in production requires more than just inference optimization. This covers the infrastructure layer: load balancing, request routing, scaling strategies, and operational concerns that make LLM serving reliable and cost-effective.

## Learning Objectives

- [ ] Model serving patterns (online vs batch)
- [ ] Load balancing strategies for LLMs
- [ ] Request scheduling and prioritization
- [ ] CUDA Graphs in production
- [ ] Autoscaling and cost optimization
- [ ] Observability and monitoring

## Resources

### Papers

- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving](https://arxiv.org/abs/2302.11665)
- [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/abs/2311.18677)

### Blogs & Tutorials

- [vLLM Production Deployment Guide](https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html)
- [Anyscale: Scaling LLM Inference](https://www.anyscale.com/blog/scaling-llm-inference)
- [Modal LLM Serving Guide](https://modal.com/docs/guides/llm-serving)

### Code References

- [vLLM](https://github.com/vllm-project/vllm)
- [Ray Serve](https://docs.ray.io/en/latest/serve/index.html)
- [Triton Inference Server](https://github.com/triton-inference-server/server)

---

## Notes

### Serving Patterns

| Pattern | Latency | Throughput | Use Case |
|---------|---------|------------|----------|
| Online (sync) | Low | Variable | Chat, real-time apps |
| Streaming | Low TTFT | Variable | Long generation, UX |
| Batch | High | Maximum | Bulk processing, eval |
| Async | Variable | High | Background tasks |

### Online vs Batch Processing

```
Online Serving:
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Request │────▶│ Inference│────▶│ Response│
│ arrives │     │ (fast)  │     │ (immediate)│
└─────────┘     └─────────┘     └─────────┘
                    │
                    └──── Latency SLA: < 1s TTFT

Batch Processing:
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Requests│────▶│ Queue   │────▶│ Process │────▶ Results
│ (many)  │     │ (batch) │     │ (bulk)  │     (later)
└─────────┘     └─────────┘     └─────────┘
                    │
                    └──── Throughput: Maximize tokens/sec/$
```

### Load Balancing Strategies

**Round Robin:**
```
Request 1 ──▶ GPU 0
Request 2 ──▶ GPU 1
Request 3 ──▶ GPU 2
Request 4 ──▶ GPU 0  (cycle)
```
- Simple, predictable
- Ignores current load
- Bad for variable-length requests

**Least Connections:**
```
              ┌──────┐
Request ─────▶│Router│
              └──┬───┘
                 │ Pick GPU with fewest active requests
    ┌────────────┼────────────┐
    ▼            ▼            ▼
GPU 0 (2)    GPU 1 (5)    GPU 2 (3)
   ↑
   └── Selected (lowest)
```
- Better load distribution
- Still ignores request complexity

**Prefix-Aware Routing:**
```
              ┌──────────────────┐
Request ─────▶│ Prefix Matching  │
              │ Router           │
              └────────┬─────────┘
                       │ Route to GPU with cached prefix
    ┌──────────────────┼──────────────────┐
    ▼                  ▼                  ▼
GPU 0               GPU 1               GPU 2
[System A cached]   [System B cached]   [System C cached]
```
- Maximizes KV cache reuse
- Essential for multi-tenant serving
- See [KV Cache](01_KV_Cache.md) for RadixAttention details

**Sticky Routing (Session Affinity):**
```python
class StickyScheduler:
    def __init__(self, workers):
        self.user_to_worker = {}  # user_id -> worker_id

    def dispatch(self, user_id, prompt_tokens):
        if user_id in self.user_to_worker:
            return self.user_to_worker[user_id]

        # First time: assign least loaded worker
        worker_id = self._select_least_loaded_worker()
        self.user_to_worker[user_id] = worker_id
        return worker_id
```
- O(1) lookup - extremely fast
- Perfect for sequential conversations
- Risk: load imbalance, no cross-user cache sharing

**Hybrid Routing (Production Recommended):**
```python
class HybridScheduler:
    def dispatch(self, user_id, prompt_tokens):
        # Fast path: sticky routing
        if user_id in self.user_to_worker:
            worker = self.get_worker(self.user_to_worker[user_id])
            if worker.is_available() and self._cache_hit_rate(worker, prompt_tokens) > 0.5:
                return worker.id

        # Fallback: prefix matching
        worker_id = self.trie_scheduler.dispatch(prompt_tokens)
        self.user_to_worker[user_id] = worker_id
        return worker_id
```
- Combines benefits of both approaches
- Fast path for 95% of requests
- Handles failures and context switching gracefully

### Routing Strategy Comparison

| Strategy | Latency | Cache Reuse | Load Balance | Best For |
|----------|---------|-------------|--------------|----------|
| Round Robin | O(1) | None | Good | Simple deployments |
| Least Connections | O(n) | None | Better | Variable workloads |
| Sticky | O(1) | Per-user | Poor | Single-user chat |
| Prefix Matching | O(k) | Cross-user | Natural | Multi-tenant, RAG |
| Hybrid | O(1)/O(k) | Both | Good | **Production systems** |

**When Users Switch Contexts:**
```
Morning:  "Explain Python..." → Worker 1
Afternoon: "How to use Docker..." → Worker 2
Evening:  "More about Python..." → ?

Sticky: Routes to Worker 2 ❌ (Docker cached, not Python)
Prefix: Routes to Worker 1 ✅ (Python still cached)
Hybrid: Routes to Worker 1 ✅ (detects cache mismatch)
```

### Request Scheduling Priorities

```python
class PriorityScheduler:
    def __init__(self):
        self.queues = {
            "high": [],      # Premium users, time-sensitive
            "normal": [],    # Standard requests
            "low": [],       # Background, batch jobs
        }

    def schedule(self, request):
        priority = self.get_priority(request)
        self.queues[priority].append(request)

    def get_next_batch(self, max_size):
        batch = []
        # Drain high priority first
        for priority in ["high", "normal", "low"]:
            while self.queues[priority] and len(batch) < max_size:
                batch.append(self.queues[priority].pop(0))
        return batch
```

### CUDA Graphs in Production

CUDA Graphs reduce kernel launch overhead by capturing and replaying execution:

```
Without CUDA Graphs:
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Launch   │──│ Launch   │──│ Launch   │
│ Kernel 1 │  │ Kernel 2 │  │ Kernel 3 │
└──────────┘  └──────────┘  └──────────┘
   ~5μs          ~5μs          ~5μs      = 15μs overhead

With CUDA Graphs:
┌─────────────────────────────────────────┐
│ Graph Replay (all kernels)              │
│ [K1] ──▶ [K2] ──▶ [K3]                 │
└─────────────────────────────────────────┘
   ~2μs total overhead
```

**When CUDA Graphs help:**
- Decode phase (same operation repeated)
- Fixed batch sizes
- Short sequences (overhead % is higher)

**Limitations:**
- Can't handle dynamic shapes easily
- Memory overhead for multiple graph captures
- Setup time for initial capture

### Autoscaling Strategies

**Metrics to monitor:**
| Metric | Scale Up When | Scale Down When |
|--------|---------------|-----------------|
| Queue depth | > 100 requests | < 10 requests |
| GPU utilization | > 80% sustained | < 30% sustained |
| P99 latency | > SLA threshold | Well under SLA |
| Tokens/sec/GPU | Declining | Stable |

**Scaling approaches:**
```
Horizontal Scaling (more replicas):
┌─────────────────────────────────────────────┐
│ Load Balancer                               │
├─────────────────────────────────────────────┤
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐       │
│  │GPU 0│  │GPU 1│  │GPU 2│  │GPU 3│  ...  │
│  │     │  │     │  │     │  │     │       │
│  └─────┘  └─────┘  └─────┘  └─────┘       │
└─────────────────────────────────────────────┘

Vertical Scaling (bigger instances):
┌─────────────────────────────────────────────┐
│ Single Instance (8x A100)                   │
│  ┌─────────────────────────────────────┐   │
│  │ Tensor Parallel across 8 GPUs       │   │
│  │ [TP0][TP1][TP2][TP3][TP4][TP5][TP6][TP7]│
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

### Deployment Architecture

```
                    ┌─────────────────┐
                    │   API Gateway   │
                    │   (rate limit,  │
                    │    auth, log)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Load Balancer  │
                    │  (prefix-aware) │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  vLLM Pod 1   │    │  vLLM Pod 2   │    │  vLLM Pod 3   │
│  (2x A100)    │    │  (2x A100)    │    │  (2x A100)    │
└───────────────┘    └───────────────┘    └───────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Model Storage  │
                    │  (S3/GCS/NFS)   │
                    └─────────────────┘
```

### Cost Optimization

**GPU utilization is key:**
```
Cost per token = (GPU cost per hour) / (tokens per hour)

A100 @ $2/hr generating 1000 tokens/sec:
Cost = $2 / (1000 × 3600) = $0.0000005/token

Same A100 at 30% utilization (300 tokens/sec):
Cost = $2 / (300 × 3600) = $0.0000018/token  (3.6x more expensive!)
```

**Optimization strategies:**
1. **Batching**: Higher batch size = better GPU utilization
2. **Quantization**: Fit larger batches in memory
3. **Spot instances**: 60-90% cost savings (with preemption handling)
4. **Right-sizing**: Match GPU to model size

### Observability Stack

```
┌─────────────────────────────────────────────────────────────┐
│                     Monitoring Stack                         │
├─────────────────────────────────────────────────────────────┤
│  Metrics (Prometheus/Datadog):                              │
│  - tokens_generated_total                                    │
│  - request_latency_seconds (histogram)                       │
│  - batch_size (gauge)                                        │
│  - gpu_memory_used_bytes                                     │
│  - queue_depth                                               │
├─────────────────────────────────────────────────────────────┤
│  Tracing (Jaeger/Honeycomb):                                │
│  - Request ID propagation                                    │
│  - Time per phase (queue, prefill, decode)                  │
│  - Token-level timing                                        │
├─────────────────────────────────────────────────────────────┤
│  Logging (ELK/Loki):                                        │
│  - Request/response pairs                                    │
│  - Error traces                                              │
│  - Model loading events                                      │
└─────────────────────────────────────────────────────────────┘
```

### Key Metrics Dashboard

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| TTFT P50 | < 200ms | < 500ms | > 1s |
| TTFT P99 | < 500ms | < 2s | > 5s |
| GPU util | > 70% | 40-70% | < 40% |
| Error rate | < 0.1% | < 1% | > 1% |
| Queue depth | < 50 | < 200 | > 500 |

### Handling Failures

**Graceful degradation:**
```python
async def serve_with_fallback(request):
    try:
        # Primary: full model
        return await primary_model.generate(request, timeout=30)
    except TimeoutError:
        # Fallback 1: shorter output
        return await primary_model.generate(
            request, max_tokens=request.max_tokens // 2
        )
    except GPUOOMError:
        # Fallback 2: smaller model
        return await fallback_model.generate(request)
    except Exception:
        # Fallback 3: cached/canned response
        return get_cached_response(request)
```

**Health checks:**
```python
@app.get("/health")
async def health():
    checks = {
        "gpu_available": torch.cuda.is_available(),
        "model_loaded": model is not None,
        "memory_ok": get_gpu_memory_free() > MIN_FREE_MEMORY,
        "queue_ok": queue.size() < MAX_QUEUE_SIZE,
    }
    healthy = all(checks.values())
    return {"status": "healthy" if healthy else "degraded", "checks": checks}
```

### Multi-Model Serving

When serving multiple models on shared infrastructure:

```
┌─────────────────────────────────────────────────────────────┐
│                    Model Router                              │
├─────────────────────────────────────────────────────────────┤
│  Request ──▶ Model Selection ──▶ Route to appropriate GPU   │
│                                                              │
│  Strategy options:                                           │
│  1. Dedicated GPUs per model                                 │
│  2. Time-multiplexing (swap models)                         │
│  3. Memory-multiplexing (fit multiple small models)         │
└─────────────────────────────────────────────────────────────┘

Dedicated:           Time-multiplex:      Memory-multiplex:
┌─────┐ ┌─────┐     ┌─────────────┐      ┌─────────────┐
│GPT-4│ │Llama│     │ GPT-4 ──────│      │ GPT-4 (50%) │
│     │ │     │     │ ──▶ Llama ──│      │ Llama (50%) │
└─────┘ └─────┘     └─────────────┘      └─────────────┘
```
