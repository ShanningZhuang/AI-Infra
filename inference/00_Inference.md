# LLM Inference Optimization

> Parent: [AI Infrastructure](../00_AI_Infra.md)

## Overview

LLM inference is fundamentally different from training. While training processes batches of fixed-length sequences, inference generates tokens autoregressively one at a time. This creates unique challenges: memory-bound computation, variable sequence lengths, and the need to serve thousands of concurrent requests efficiently.

## The Inference Challenge

```
Training:                          Inference:
┌──────────────────┐              ┌──────────────────┐
│ Fixed batch size │              │ Variable requests│
│ Fixed seq length │              │ Variable lengths │
│ Compute-bound    │              │ Memory-bound     │
│ Throughput focus │              │ Latency matters  │
└──────────────────┘              └──────────────────┘
```

### Why Inference is Memory-Bound

For each generated token, the model must:
1. Load all model weights from memory
2. Compute attention over all previous tokens
3. Generate just ONE token

This means:
- Small compute per memory access
- Memory bandwidth becomes the bottleneck
- GPU utilization often < 30% without optimization

## Key Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| TTFT | Time To First Token | < 500ms |
| TPOT | Time Per Output Token | < 50ms |
| Throughput | Tokens/second across all requests | Maximize |
| GPU Utilization | % of GPU compute used | > 70% |

## Learning Path

### Core Concepts

| Topic | File | Priority |
|-------|------|----------|
| KV Cache & Prefix Caching | [01_KV_Cache.md](01_KV_Cache.md) | ★★★★★ |
| Batching Strategies | [02_Batching.md](02_Batching.md) | ★★★★★ |
| Speculative Decoding | [03_Speculative_Decoding.md](03_Speculative_Decoding.md) | ★★★★☆ |
| Quantization | [04_Quantization.md](04_Quantization.md) | ★★★★☆ |

### Systems & Frameworks

| Topic | File | Priority |
|-------|------|----------|
| Inference Frameworks | [05_Frameworks.md](05_Frameworks.md) | ★★★★★ |
| Serving & Routing | [06_Serving.md](06_Serving.md) | ★★★★☆ |

---

## The Big Picture: Inference Optimization Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    Request Level                             │
│  Continuous Batching │ Request Scheduling │ Load Balancing   │
├─────────────────────────────────────────────────────────────┤
│                    Memory Level                              │
│  KV Cache Management │ PagedAttention │ Prefix Caching       │
├─────────────────────────────────────────────────────────────┤
│                    Compute Level                             │
│  Quantization │ Speculative Decoding │ CUDA Graphs           │
├─────────────────────────────────────────────────────────────┤
│                    Kernel Level                              │
│  FlashAttention │ Fused Kernels │ Custom CUDA                │
└─────────────────────────────────────────────────────────────┘
```

## Key Papers

| Paper | Key Contribution |
|-------|------------------|
| [Orca (2022)](https://www.usenix.org/conference/osdi22/presentation/yu) | Continuous batching |
| [vLLM (2023)](https://arxiv.org/abs/2309.06180) | PagedAttention for KV cache |
| [SGLang (2024)](https://arxiv.org/abs/2312.07104) | RadixAttention, prefix caching |
| [Speculative Decoding (2022)](https://arxiv.org/abs/2211.17192) | Draft-verify paradigm |

## Key Blogs & Resources

- [vLLM Blog: How Continuous Batching Works](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [Aleksa Gordić's vLLM Deep Dive](https://www.aleksagordic.com/blog/vllm)
- [HuggingFace Text Generation Inference](https://huggingface.co/docs/text-generation-inference)

---

## Notes

### Autoregressive Generation Basics

```python
# Simplified autoregressive generation
def generate(model, prompt_tokens, max_new_tokens):
    tokens = prompt_tokens

    for _ in range(max_new_tokens):
        # Forward pass through entire model for ONE token
        logits = model(tokens)

        # Sample next token
        next_token = sample(logits[:, -1, :])
        tokens = torch.cat([tokens, next_token], dim=1)

        if next_token == EOS:
            break

    return tokens
```

**Problem**: Each iteration recomputes attention for ALL previous tokens!

### The KV Cache Solution

```python
def generate_with_cache(model, prompt_tokens, max_new_tokens):
    # Prefill: process all prompt tokens at once
    logits, kv_cache = model(prompt_tokens, use_cache=True)

    next_token = sample(logits[:, -1, :])
    tokens = [next_token]

    for _ in range(max_new_tokens - 1):
        # Decode: only process ONE new token, reuse cache
        logits, kv_cache = model(next_token, past_kv=kv_cache)
        next_token = sample(logits[:, -1, :])
        tokens.append(next_token)

        if next_token == EOS:
            break

    return tokens
```

**Improvement**: O(n) → O(1) compute per token (but memory grows linearly)

### Two Phases of Inference

| Phase | Input | Compute | Memory | Characteristic |
|-------|-------|---------|--------|----------------|
| Prefill | All prompt tokens | High (parallel) | Allocate KV | Compute-bound |
| Decode | One token at a time | Low (sequential) | Append to KV | Memory-bound |

This prefill/decode split is fundamental to understanding inference optimization.
