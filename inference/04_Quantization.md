# Quantization for LLM Inference

> Parent: [Inference Overview](inference.md)

## Overview

Quantization reduces model memory footprint and increases inference speed by using lower-precision number formats. For LLMs, this means storing weights (and sometimes activations) in INT8, INT4, or even lower bit-widths instead of FP16/BF16.

## Learning Objectives

- [ ] Why quantization works for LLMs
- [ ] Post-Training Quantization (PTQ) vs Quantization-Aware Training (QAT)
- [ ] Weight-only vs weight-activation quantization
- [ ] GPTQ, AWQ, and other methods
- [ ] Practical tradeoffs

## Resources

### Papers

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs](https://arxiv.org/abs/2211.10438)
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
- [QuIP: 2-Bit Quantization of Large Language Models](https://arxiv.org/abs/2307.13304)

### Blogs & Tutorials

- [HuggingFace Quantization Guide](https://huggingface.co/docs/transformers/quantization)
- [bitsandbytes Documentation](https://huggingface.co/docs/bitsandbytes)
- [NVIDIA TensorRT-LLM Quantization](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/precision.md)

### Code References

- [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ)
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF quantization

---

## Notes

### Why Quantization Works

LLMs are surprisingly robust to reduced precision because:
1. Weights follow approximately Gaussian distribution
2. Outliers are rare and can be handled specially
3. Most computation is in MatMul (tolerates noise)

```
FP16 weight distribution:      Quantized (INT4):
     ▲                              ▲
     │    ╭──╮                      │  │ │ │ │
     │   ╱    ╲                     │  │ │ │ │
     │  ╱      ╲                    │  │ │ │ │
     │ ╱        ╲                   │  │ │ │ │
     └──────────────▶               └──────────────▶
       -1    0    1                   16 discrete levels
```

### Precision Formats Comparison

| Format | Bits | Range | Memory | Speed | Quality |
|--------|------|-------|--------|-------|---------|
| FP32 | 32 | ±3.4e38 | 4 bytes | 1x | Baseline |
| FP16 | 16 | ±65504 | 2 bytes | 2x | ~Same |
| BF16 | 16 | ±3.4e38 | 2 bytes | 2x | ~Same |
| FP8 (E4M3) | 8 | ±448 | 1 byte | 4x | Slight drop |
| INT8 | 8 | -128~127 | 1 byte | 4x | Slight drop |
| INT4 | 4 | -8~7 | 0.5 bytes | 8x | Noticeable drop |

### Weight-Only vs Weight-Activation Quantization

**Weight-Only Quantization:**
```
Storage:  Weights in INT4/INT8
Compute:  Dequantize to FP16 → MatMul in FP16

┌──────────┐    ┌────────────┐    ┌─────────┐
│ INT4     │───▶│ Dequantize │───▶│  FP16   │───▶ MatMul
│ Weights  │    │ to FP16    │    │ Weights │
└──────────┘    └────────────┘    └─────────┘
```

Benefits:
- Memory reduction (4x for INT4)
- Works well for memory-bound decode
- Minimal quality loss

**Weight-Activation Quantization (W8A8):**
```
Both weights AND activations in INT8
Compute directly in INT8 (faster)

┌──────────┐         ┌──────────┐
│ INT8     │         │ INT8     │
│ Weights  │───┬────▶│ MatMul   │───▶ INT32 accumulate
└──────────┘   │     └──────────┘
               │
┌──────────┐   │
│ INT8     │───┘
│ Activat. │
└──────────┘
```

Challenges:
- Activation outliers cause accuracy loss
- Requires careful calibration

### The Outlier Problem

LLM activations have extreme outliers in certain channels:

```
Activation distribution:
      ▲
      │                              ○  ← Outliers (0.1% of values)
      │                              ○    but important for accuracy!
      │    ╭──╮
      │   ╱    ╲
      │  ╱      ╲
      └─────────────────────────────────▶

Naive INT8: Outliers clip → severe accuracy loss
```

### SmoothQuant Solution

Migrate quantization difficulty from activations to weights:

```
Original: Y = X @ W
          ↑ hard to quantize (outliers)

Smoothed: Y = (X @ diag(s)^-1) @ (diag(s) @ W)
              ↑ easier          ↑ slightly harder
              (scales down      (absorbs the
               outliers)        difficulty)
```

### GPTQ: Layer-wise Quantization

Quantize weights one layer at a time, minimizing squared error:

```python
# Simplified GPTQ algorithm
def gptq_quantize(W, H, bits=4):
    """
    W: weight matrix [out_features, in_features]
    H: Hessian approximation (X^T @ X from calibration)
    """
    Q = torch.zeros_like(W)

    for i in range(W.shape[1]):  # Column by column
        # Find optimal quantized value
        q = quantize(W[:, i], bits)

        # Compute quantization error
        error = W[:, i] - q

        # Update remaining weights to compensate
        # (This is the key insight - later columns adjust)
        W[:, i+1:] -= error.unsqueeze(1) @ H[i, i+1:].unsqueeze(0) / H[i, i]

        Q[:, i] = q

    return Q
```

Key insight: Errors in early columns are compensated by adjusting later columns.

### AWQ: Activation-Aware Quantization

Key observation: Not all weights are equally important.
Weights corresponding to large activations matter more.

```python
# AWQ scales important channels before quantization
def awq_quantize(W, X_calibration, bits=4):
    """
    X_calibration: calibration activations
    """
    # Find which channels have large activations
    importance = X_calibration.abs().mean(dim=0)

    # Scale important channels up (protect from quantization error)
    scales = importance.pow(0.5)  # sqrt heuristic
    W_scaled = W * scales.unsqueeze(0)

    # Quantize scaled weights
    Q_scaled = quantize(W_scaled, bits)

    # Store scales for inference (absorbed into next layer)
    return Q_scaled, scales
```

### Quantization Methods Comparison

| Method | Type | Bits | Calibration | Quality | Speed |
|--------|------|------|-------------|---------|-------|
| bitsandbytes | Weight-only | 8/4 | None | Good | Medium |
| GPTQ | Weight-only | 4/3/2 | Required | Very good | Fast |
| AWQ | Weight-only | 4 | Required | Best | Fast |
| SmoothQuant | W8A8 | 8 | Required | Good | Fastest |
| LLM.int8() | Mixed | 8 | None | Best | Slower |

### LLM.int8(): Mixed Precision

Handle outliers by keeping them in FP16:

```
┌────────────────────────────────────────────┐
│ Activation tensor                          │
├────────────────────────────────────────────┤
│ 99.9% normal values │ 0.1% outliers        │
│       ↓             │       ↓              │
│    INT8 MatMul      │    FP16 MatMul       │
│       ↓             │       ↓              │
│       └─────────────┴─────────┘            │
│              Add results                    │
└────────────────────────────────────────────┘
```

### Practical Quantization Guide

**When to use what:**

| Scenario | Recommended | Why |
|----------|-------------|-----|
| 7B model, consumer GPU | AWQ 4-bit | Fits in VRAM |
| 70B model, single A100 | AWQ 4-bit | Memory constrained |
| Production serving | W8A8 (SmoothQuant) | Best throughput |
| Quality-sensitive | GPTQ 4-bit | Better perplexity |
| Quick experimentation | bitsandbytes | No calibration |

### Memory Savings

```
Model Size in Memory:

70B model:
FP32: 280 GB  ████████████████████████████
FP16: 140 GB  ██████████████
INT8:  70 GB  ███████
INT4:  35 GB  ████

7B model:
FP32: 28 GB   ████████████████████████████
FP16: 14 GB   ██████████████
INT8:  7 GB   ███████
INT4:  3.5 GB ████
```

### Quantization-Aware Training (QAT)

Train with simulated quantization for better quality:

```python
# Simplified QAT forward pass
def qat_forward(x, W_fp32, bits=4):
    # Quantize then dequantize (simulate quantization error)
    W_fake_quant = dequantize(quantize(W_fp32, bits))

    # Forward with fake-quantized weights
    # Gradients flow through (straight-through estimator)
    return x @ W_fake_quant.T
```

Trade-off: Better quality but requires training compute.

### GGUF Format (llama.cpp)

Popular format for CPU inference with many quantization options:

| Format | Bits/Weight | Quality | Notes |
|--------|-------------|---------|-------|
| Q8_0 | 8.5 | Best | Baseline |
| Q6_K | 6.6 | Very good | Recommended |
| Q5_K_M | 5.7 | Good | Balanced |
| Q4_K_M | 4.8 | OK | Memory efficient |
| Q3_K_M | 3.9 | Degraded | Aggressive |
| Q2_K | 3.4 | Poor | Extreme |

### Code Example: Using bitsandbytes

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # Quantize the quantization constants
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
```

### Code Example: Using AutoGPTQ

```python
from auto_gptq import AutoGPTQForCausalLM

# Load pre-quantized model
model = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-2-70B-GPTQ",
    device_map="auto",
    use_safetensors=True,
)

# Or quantize yourself
from auto_gptq import BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,  # Activation order (slower but better)
)

model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantize_config,
)

# Calibrate with sample data
model.quantize(calibration_data)
model.save_quantized("llama-2-70b-4bit-gptq")
```

### FAQ: Quantization in Production

**Q: Given scaling laws show more parameters = more intelligence, do frontier labs (OpenAI, Anthropic) use quantization to fit more parameters, or full precision with fewer parameters?**

A: This is a common misconception. The choice isn't "quantized large model vs full-precision small model" — frontier labs do both optimally:

```
Training vs Inference: Different Precision Strategies

Training Phase:
┌─────────────────────────────────────────────────────────────┐
│  ALWAYS full precision (BF16/FP16)                          │
│  - Gradient updates need precision                          │
│  - Training cost is fixed (one-time)                        │
│  - Scaling laws apply here → maximize parameters            │
└─────────────────────────────────────────────────────────────┘
                              ↓
                         Model weights
                              ↓
Inference Phase:
┌─────────────────────────────────────────────────────────────┐
│  May use quantization (FP8, INT8) for serving               │
│  - Reduces memory/cost per request                          │
│  - Acceptable quality loss for production                   │
│  - This is AFTER training the largest model possible        │
└─────────────────────────────────────────────────────────────┘
```

**Key insight**: Quantization is an *inference optimization*, not a training choice. Frontier labs:

1. **Train** the largest model they can afford in full precision (BF16)
2. **Serve** that model with mild quantization (FP8/INT8) if quality permits

```
What OpenAI/Anthropic likely do:

                    Training Budget
                          │
                          ▼
    ┌─────────────────────────────────────────┐
    │ Train largest model possible in BF16    │
    │ (e.g., GPT-4 ~1.8T params, Claude ~???) │
    └─────────────────────────────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────┐
    │ For serving, may use:                   │
    │ - FP8 for compute (2x speedup)          │
    │ - KV cache in FP8/INT8                  │
    │ - NOT aggressive INT4 (quality loss)    │
    └─────────────────────────────────────────┘
```

**Why not train a quantized model directly?**

| Approach | Parameters | Quality | Why not? |
|----------|------------|---------|----------|
| Train 70B in INT4 | 70B | Poor | Gradients don't work well in low precision |
| Train 70B in BF16 | 70B | Best | ✓ Standard approach |
| Train 280B in INT4 | 280B | Mediocre | Training instability, INT4 training not mature |
| Train 280B in BF16, quantize to INT8 for serving | 280B | Very good | ✓ Best of both worlds |

**The real trade-off frontier labs face:**

```
Given fixed inference cost budget:

Option A: Serve 70B model in FP16
          - High quality
          - Expensive per token

Option B: Serve 70B model in FP8/INT8  ← Likely choice for APIs
          - Slightly lower quality
          - 2x cheaper per token
          - Can serve 2x more users

Option C: Serve 70B model in INT4
          - Noticeable quality drop
          - 4x cheaper per token
          - Usually too much quality loss for frontier APIs
```

**Summary**: Scaling laws guide *training* decisions (maximize parameters). Quantization is applied *after training* for inference efficiency. Frontier labs train the biggest model possible in full precision, then apply conservative quantization (FP8/INT8, not INT4) for serving to balance quality and cost.
