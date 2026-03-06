# Multimodal Inference

> Parent: [Inference Overview](00_Inference.md)

## Overview

Serving multimodal models introduces challenges that the existing AR-only inference stack wasn't designed for. Models that combine text generation with image understanding or generation require handling fundamentally different compute patterns — autoregressive decoding (memory-bound, sequential) and diffusion denoising (compute-bound, iterative) — often within the same request. This page covers why multimodal inference differs and how frameworks are adapting.

## AR vs Diffusion Inference

```
Autoregressive (LLM/VLM text):        Diffusion (Image generation):
┌──────────────────────────┐          ┌──────────────────────────┐
│ Prefill → Decode loop    │          │ Denoise for N steps      │
│                          │          │                          │
│ • Sequential tokens      │          │ • Fixed N iterations     │
│ • KV cache grows         │          │ • No KV cache            │
│ • Memory-bound (decode)  │          │ • Compute-bound          │
│ • Variable length output │          │ • Fixed compute per step │
│ • Latency-sensitive      │          │ • Throughput-oriented    │
└──────────────────────────┘          └──────────────────────────┘
```

### Detailed Comparison

| Aspect | AR (Text) | Diffusion (Image) |
|--------|-----------|-------------------|
| Compute pattern | Prefill (parallel) + decode (sequential) | Iterative denoising (parallel within step) |
| Memory | KV cache grows with sequence | Fixed memory per denoising step |
| Bottleneck | Memory bandwidth (decode phase) | Compute (dense matmuls each step) |
| Batching benefit | Amortize weight loading across batch | Amortize kernel launch, high GPU util |
| Output | Variable-length token sequence | Fixed-size image tensor |
| Latency profile | TTFT + TPOT × num_tokens | Steps × time_per_step |
| Key optimization | KV cache, continuous batching | Fewer steps (distillation), faster sampler |

### Hybrid Models (AR + Diffusion)

Unified models like Janus and BAGEL combine both patterns:

```
User: "Describe this image, then generate a variation"

Phase 1 (Understanding):  Image → Visual Encoder → AR decode → text description
                          ↑ standard VLM inference (memory-bound decode)

Phase 2 (Generation):     Text → AR conditioning → Diffusion denoise → Image
                          ↑ diffusion inference (compute-bound, N steps)

Same model, same request, two completely different compute profiles!
```

## Challenges

### 1. Different Compute Profiles

```
AR decode:     Low compute, high memory bandwidth demand
               GPU utilization: 10-30% (without batching)

Diffusion:     High compute, moderate memory
               GPU utilization: 70-90%

Problem: Can't efficiently share GPU between AR and diffusion workloads
         if you batch them naively — one starves the other.
```

### 2. Scheduling Complexity

```
AR requests:                     Diffusion requests:
┌─────┐ ┌───────┐ ┌───┐        ┌──────────────────────┐
│ R1  │ │  R2   │ │R3 │        │     D1 (25 steps)    │
└─────┘ └───────┘ └───┘        └──────────────────────┘
Variable length, can            Fixed compute, can
preempt and resume              batch steps together

Mixed scheduling: when to switch between AR batch and diffusion batch?
```

### 3. Vision Encoder Preprocessing

Image inputs require preprocessing before LLM processing:

```
Raw image → Resize/Tile → ViT encoder → Project → Visual tokens
            ↑                ↑              ↑
         CPU-bound     Compute-bound   Quick (linear/MLP)
         (decode jpg)  (ViT forward)

For dynamic resolution (tiles): variable number of visual tokens per request
→ complicates padding and batching in the LLM
```

### 4. Variable Input Modalities

```
Request types that a multimodal server must handle:

1. Text-only:        "Hello" → AR generate
2. Image + text:     [image] + "What is this?" → encode + AR
3. Multi-image:      [img1] [img2] + "Compare" → multi-encode + AR
4. Text-to-image:    "Generate a cat" → AR + diffusion
5. Image-to-image:   [image] + "Make it winter" → encode + diffusion
6. Video:            [frames] + "Summarize" → multi-encode + AR

Each type has different memory/compute requirements and latency expectations.
```

## Framework Solutions

### vLLM Omni

Extending vLLM beyond AR-only serving:

```
Standard vLLM:                    vLLM Omni:
┌──────────────────┐             ┌──────────────────────────┐
│ AR Scheduler     │             │ Unified Scheduler        │
│ KV Cache Manager │             │ ├─ AR pipeline           │
│ PagedAttention   │             │ │  └─ KV cache, batching │
│                  │             │ ├─ Diffusion pipeline    │
│ Text-only models │             │ │  └─ step scheduling    │
└──────────────────┘             │ └─ Vision preprocessing  │
                                 │    └─ encoder batching   │
                                 └──────────────────────────┘
```

Key extensions:
- **Multimodal input processing**: batch image encoding separately from LLM
- **Support for VLMs**: handle variable visual token counts
- **Diffusion model support**: add diffusion inference pipeline alongside AR
- **Unified API**: single endpoint for text, VLM, and generation requests

### SGLang Diffusion

SGLang extending to diffusion model serving:

```
SGLang Core:                     + Diffusion Extension:
┌──────────────────┐             ┌──────────────────────┐
│ RadixAttention   │             │ Diffusion scheduler   │
│ Structured gen   │             │ Step-level batching   │
│ Prefix caching   │             │ Classifier-free       │
│                  │             │ guidance batching     │
└──────────────────┘             └──────────────────────┘
```

Key features:
- **Diffusion-specific scheduling**: batch denoising steps across requests
- **CFG optimization**: efficient handling of conditional + unconditional passes
- **Integration with AR**: serve unified models with mixed AR + diffusion

### Handling Mixed Pipelines

How frameworks handle a unified model request:

```
1. Request arrives: "Generate an image of a sunset"
2. Tokenize text prompt
3. AR prefill: process text tokens through LLM
4. AR decode: generate conditioning tokens (if model uses AR conditioning)
5. Switch to diffusion: initialize noise, begin denoising loop
6. Diffusion steps: iterate N times through denoising network
7. VAE decode: latent → pixel image
8. Return image to user

Steps 3-4: use AR inference engine (KV cache, continuous batching)
Steps 5-6: use diffusion inference engine (compute-bound, step batching)
Step 7: separate decode pass (can be batched independently)
```

## Optimization Strategies

### Compute Isolation

```
Option A: Separate GPU pools
  GPU pool 1: AR inference (optimized for memory bandwidth)
  GPU pool 2: Diffusion inference (optimized for compute)
  + Simple, predictable
  - Wasted resources when load is imbalanced

Option B: Time-multiplexed sharing
  Alternate between AR batch and diffusion batch on same GPUs
  + Better utilization
  - Context switching overhead, harder to implement

Option C: Request-level routing
  Route text-only → AR pool, generation → diffusion pool
  + Natural fit for separated workloads
  - Unified models need both in one request
```

### Vision Encoder Optimization

| Technique | Description | Benefit |
|-----------|-------------|---------|
| Encoder batching | Batch ViT forward passes across requests | Higher GPU utilization |
| Async encoding | Encode images while LLM processes other requests | Better pipeline utilization |
| Cached encoding | Cache encoder outputs for repeated images | Avoid redundant compute |
| Token compression | Reduce visual tokens before LLM | Less LLM compute |

## Related

- [Inference Overview](00_Inference.md) — AR inference fundamentals
- [Frameworks](05_Frameworks.md) — vLLM, SGLang, TensorRT-LLM core architectures
- [Batching](02_Batching.md) — continuous batching for AR models
- [KV Cache](01_KV_Cache.md) — memory management for AR inference
- [Multimodal KB: Diffusion Models](../../Multimodal/diffusion/00_Diffusion.md) — diffusion model internals
- [Multimodal KB: Unified Models](../../Multimodal/vision_language/03_Unified_Models.md) — models requiring mixed inference
- [Multimodal KB: DiT](../../Multimodal/diffusion/04_DiT.md) — Diffusion Transformer architecture
