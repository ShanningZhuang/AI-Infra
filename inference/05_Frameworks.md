# LLM Inference Frameworks

> Parent: [Inference Overview](inference.md)

## Overview

The LLM inference framework landscape is rapidly evolving. The three major frameworks—vLLM, TensorRT-LLM, and SGLang—each have different design philosophies and strengths. Understanding their architectures helps choose the right tool and understand modern inference optimization techniques.

## Learning Objectives

- [ ] vLLM architecture and PagedAttention
- [ ] TensorRT-LLM and NVIDIA optimization stack
- [ ] SGLang and RadixAttention
- [ ] Framework comparison and selection criteria
- [ ] HuggingFace TGI

## Resources

### Documentation

- [vLLM Documentation](https://docs.vllm.ai/)
- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [SGLang Documentation](https://sgl-project.github.io/)
- [HuggingFace TGI](https://huggingface.co/docs/text-generation-inference)

### Papers

- [vLLM: Efficient Memory Management with PagedAttention](https://arxiv.org/abs/2309.06180)
- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)

### Code

- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- [SGLang GitHub](https://github.com/sgl-project/sglang)

---

## Notes

### Framework Comparison Overview

| Feature | vLLM | TensorRT-LLM | SGLang | TGI |
|---------|------|--------------|--------|-----|
| Primary Focus | Memory efficiency | Raw speed | Structured generation | Production serving |
| PagedAttention | ✅ | ✅ | ✅ | ✅ |
| Prefix Caching | ✅ | ✅ | ✅ (RadixAttention) | ✅ |
| Speculative Decoding | ✅ | ✅ | ✅ | ✅ |
| Tensor Parallelism | ✅ | ✅ | ✅ | ✅ |
| Custom Kernels | CUDA | TensorRT | Triton/CUDA | CUDA |
| Ease of Use | High | Medium | High | High |
| Quantization | AWQ, GPTQ | FP8, INT8, INT4 | AWQ, GPTQ | bitsandbytes |

### vLLM Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       vLLM Server                           │
├─────────────────────────────────────────────────────────────┤
│  API Layer (OpenAI-compatible)                              │
├─────────────────────────────────────────────────────────────┤
│                      LLM Engine                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Scheduler  │──│   Block     │──│   Model Runner      │ │
│  │  (FCFS,     │  │   Manager   │  │   (execute model)   │ │
│  │   priority) │  │  (PagedAttn)│  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Model Layer (HuggingFace models + custom kernels)          │
│  ┌────────────────┐  ┌────────────────┐                    │
│  │ PagedAttention │  │ FlashAttention │                    │
│  │ CUDA Kernel    │  │ (optional)     │                    │
│  └────────────────┘  └────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

**Key innovations:**
1. **PagedAttention**: Virtual memory for KV cache
2. **Continuous batching**: Iteration-level scheduling
3. **Block manager**: Efficient memory allocation/deallocation

**vLLM usage:**
```python
from vllm import LLM, SamplingParams

# Initialize
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
)

# Generate
sampling_params = SamplingParams(temperature=0.8, max_tokens=256)
outputs = llm.generate(["Hello, my name is"], sampling_params)
```

### TensorRT-LLM Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TensorRT-LLM                             │
├─────────────────────────────────────────────────────────────┤
│  High-level API (Python)                                    │
├─────────────────────────────────────────────────────────────┤
│  Model Definition Layer                                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Model builder (graph construction)                   │  │
│  │  - Weight loading from HuggingFace/checkpoints       │  │
│  │  - Quantization calibration                          │  │
│  └──────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  TensorRT Optimization Engine                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  - Layer fusion                                       │  │
│  │  - Kernel auto-tuning                                 │  │
│  │  - Memory optimization                                │  │
│  │  - FP8/INT8/INT4 kernels                              │  │
│  └──────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  Runtime (C++ with Python bindings)                        │
│  ┌────────────────┐  ┌────────────────┐                    │
│  │ Inflight       │  │ NVIDIA custom  │                    │
│  │ Batching       │  │ CUDA kernels   │                    │
│  └────────────────┘  └────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

**Key innovations:**
1. **TensorRT integration**: Automatic kernel optimization
2. **FP8 support**: Best-in-class H100 performance
3. **Custom CUDA kernels**: Highly optimized for NVIDIA hardware

**TensorRT-LLM usage:**
```python
# Build engine (offline)
from tensorrt_llm import build

build(
    model_dir="meta-llama/Llama-2-7b-hf",
    output_dir="./llama-2-7b-engine",
    dtype="float16",
    tp_size=1,
)

# Run inference
from tensorrt_llm.runtime import ModelRunner

runner = ModelRunner.from_dir("./llama-2-7b-engine")
outputs = runner.generate(["Hello, my name is"], max_new_tokens=256)
```

### SGLang Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       SGLang                                │
├─────────────────────────────────────────────────────────────┤
│  Frontend (SGLang Language / Python)                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Structured generation primitives                     │  │
│  │  - gen(), select(), fork(), regex constraints        │  │
│  └──────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  RadixAttention Engine                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Radix Tree (Trie) for KV Cache Management           │  │
│  │  - Automatic prefix matching                          │  │
│  │  - LRU eviction                                       │  │
│  │  - Cross-request KV sharing                           │  │
│  └──────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  Backend Runtime                                            │
│  ┌────────────────┐  ┌────────────────┐                    │
│  │ Triton kernels │  │ FlashInfer     │                    │
│  │ (custom)       │  │ attention      │                    │
│  └────────────────┘  └────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

**Key innovations:**
1. **RadixAttention**: Trie-based KV cache for prefix reuse
2. **Structured generation**: Native support for constrained decoding
3. **Compiled execution**: Optimizes multi-turn conversations

**SGLang usage:**
```python
import sglang as sgl

@sgl.function
def multi_turn_qa(s, questions):
    s += "You are a helpful assistant.\n"
    for q in questions:
        s += sgl.user(q)
        s += sgl.assistant(sgl.gen("answer", max_tokens=256))

# Run with automatic prefix caching
runtime = sgl.Runtime(model_path="meta-llama/Llama-2-7b-hf")
state = multi_turn_qa.run(questions=["What is 2+2?", "Why?"])
```

### RadixAttention Deep Dive

SGLang's key innovation for prefix caching:

```
Radix Tree (Trie) Structure:

                         [root]
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        "System:"    "Translate"   "Code:"
              │            │            │
              ▼            ▼            ▼
        "You are"    "to French:"  "def foo"
              │                         │
         ┌────┴────┐                    ▼
         ▼         ▼              "(x, y):"
   "a helpful" "an expert"

Each node stores:
- Token sequence
- Pointer to KV cache blocks
- Reference count (for LRU eviction)
```

Benefits over simple prefix matching:
- O(L) lookup for L-length prefix (vs O(N×L) for linear search)
- Automatic handling of partial prefix matches
- Efficient LRU eviction at any granularity

### HuggingFace TGI

Production-focused inference server:

```
┌─────────────────────────────────────────────────────────────┐
│                 Text Generation Inference                   │
├─────────────────────────────────────────────────────────────┤
│  Router (Rust)                                              │
│  - Load balancing across shards                             │
│  - Health checks                                            │
│  - Request validation                                        │
├─────────────────────────────────────────────────────────────┤
│  Model Server (Python + Rust)                               │
│  ┌────────────────┐  ┌────────────────┐                    │
│  │ Continuous     │  │ FlashAttention │                    │
│  │ Batching       │  │ + PagedAttn    │                    │
│  └────────────────┘  └────────────────┘                    │
├─────────────────────────────────────────────────────────────┤
│  Inference Engine (HuggingFace Transformers)               │
└─────────────────────────────────────────────────────────────┘
```

**TGI usage:**
```bash
# Docker deployment
docker run --gpus all -p 8080:80 \
    -v ~/.cache/huggingface:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-7b-hf \
    --num-shard 1

# Client
curl http://localhost:8080/generate \
    -d '{"inputs": "Hello", "parameters": {"max_new_tokens": 100}}'
```

### When to Use Which Framework

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Quick prototyping | vLLM | Easy setup, good defaults |
| Maximum throughput (NVIDIA) | TensorRT-LLM | Best raw performance |
| Multi-turn / agent workloads | SGLang | RadixAttention excels |
| Production deployment | TGI or vLLM | Battle-tested |
| Custom model architecture | vLLM | Most flexible |
| Constrained generation | SGLang | Native support |
| Edge / consumer GPU | vLLM + AWQ | Memory efficient |

### Performance Comparison (Rough)

```
Throughput comparison (normalized to vLLM baseline):

Short prompts, short outputs:
vLLM:        ████████████████████ 1.0x
TRT-LLM:     ██████████████████████████ 1.3x
SGLang:      █████████████████████ 1.05x

Long prompts (prefix-heavy):
vLLM:        ████████████████████ 1.0x
TRT-LLM:     ██████████████████████ 1.1x
SGLang:      ███████████████████████████████ 1.5x (prefix caching)

Multi-turn conversation:
vLLM:        ████████████████████ 1.0x
TRT-LLM:     ██████████████████████ 1.1x
SGLang:      ██████████████████████████████████████ 2.0x (RadixAttention)
```

*Note: Actual performance varies greatly based on model, hardware, and workload.*

### Code Example: vLLM Server

```python
# server.py
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from fastapi import FastAPI
import uvicorn

app = FastAPI()

engine_args = AsyncEngineArgs(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    max_model_len=4096,
)
engine = AsyncLLMEngine.from_engine_args(engine_args)

@app.post("/generate")
async def generate(prompt: str, max_tokens: int = 256):
    sampling_params = SamplingParams(max_tokens=max_tokens)
    results = []

    async for output in engine.generate(prompt, sampling_params, request_id="1"):
        results.append(output)

    return {"text": results[-1].outputs[0].text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### OpenAI-Compatible API

All major frameworks support OpenAI-compatible APIs:

```python
# Works with vLLM, TGI, SGLang (with --api-mode openai)
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

### Key Configuration Parameters

| Parameter | vLLM | TensorRT-LLM | SGLang |
|-----------|------|--------------|--------|
| GPU memory | `gpu_memory_utilization` | Compile-time | `mem_fraction_static` |
| Tensor parallel | `tensor_parallel_size` | `tp_size` | `tp_size` |
| Max sequence | `max_model_len` | `max_input_len` | `context_length` |
| Quantization | `quantization` | Build-time | `quantization` |
