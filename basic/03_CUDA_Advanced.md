# CUDA Advanced Topics

## Overview

Advanced CUDA concepts: PTX/SASS assembly, warp-level primitives, and performance profiling with Nsight Compute.

## Learning Path

### Phase 1: Master GEMM in CUDA
- [x] Simon Boehm's GEMM tutorial - Implement from scratch
  - [x] Naive kernel implementation
  - [x] Global memory coalescing
  - [x] Shared memory tiling
  - [ ] 1D blocktiling
  - [ ] 2D blocktiling
  - [ ] Vectorized memory access (float4)
  - [ ] Register tiling / thread-level parallelism
  - [ ] Warptiling
  - [ ] Benchmark against cuBLAS (~90% performance)

### Phase 2: FlashAttention
- [x] FlashAttention in Triton
- [ ] (Optional) Reimplement FlashAttention in raw CUDA
- [x] Compare with Dao-AILab/flash-attention source

### Phase 3: Study Production CUDA Code
- [x] Read CUTLASS GEMM kernel
- [x] Read flash-attention CUDA source
- [x] Explore FlashInfer kernels

### Phase 4: Advanced & Profiling
- [x] Learn Nsight Compute profiling
- [x] Learn to read PTX/SASS assembly
- [x] Understand warp-level primitives (shuffle, vote)
- [x] Memory coalescing and bank conflicts deep dive

---

## PTX and SASS Assembly

CUDA compilation has two intermediate representation levels: **PTX** and **SASS**.

### Compilation Pipeline

```text
CUDA C/C++ (.cu)
      │
      ▼ (nvcc frontend)
    PTX (virtual ISA)
      │
      ▼ (ptxas assembler)
   SASS (native machine code)
      │
      ▼
   GPU execution
```

### PTX (Parallel Thread Execution)

**What it is**: A virtual/intermediate assembly language, similar to LLVM IR.

**Key characteristics**:
- **Virtual ISA** - not tied to specific GPU architecture
- **Forward compatible** - PTX code can run on future GPUs (JIT compiled at runtime)
- **Human-readable** - relatively easy to understand
- **Register infinite** - uses virtual registers, compiler maps to physical registers

**Common PTX instructions**:

```asm
ld.global.f32  %f1, [%rd1];      // Load float from global memory
st.shared.f32  [%rd2], %f2;      // Store to shared memory
add.f32        %f3, %f1, %f2;    // f3 = f1 + f2
fma.rn.f32     %f5, %f1, %f2, %f3;  // fused multiply-add
setp.lt.f32    %p1, %f1, %f2;    // set predicate if f1 < f2
bar.sync       0;                 // __syncthreads()
```

**Generate PTX**: `nvcc -ptx kernel.cu -o kernel.ptx`

### SASS (Shader Assembly)

**What it is**: The actual native machine code executed by GPU.

**Key characteristics**:
- **Architecture-specific** - different for each GPU generation (sm_80, sm_89, sm_90...)
- **Not forward compatible** - must recompile for new architectures
- **Lowest level** - what actually runs on GPU hardware
- **Physical registers** - uses real hardware registers (R0, R1, ...)

**Common SASS instructions** (varies by arch):

```asm
LDG.E.128      R4, [R2.64];       // Load 128 bits from global
STS.128        [R0], R4;          // Store 128 bits to shared
FFMA           R8, R4, R5, R8;    // Fused float multiply-add
LDGSTS.E.128   [R0], [R2.64];     // Async copy (global to shared)
BAR.SYNC       0x0;               // Barrier sync
```

**View SASS**: `cuobjdump -sass kernel.cubin` or use Nsight Compute

### Why Read Assembly?

| Goal | Use |
|------|-----|
| Debug performance issues | SASS shows actual instructions, stalls, register usage |
| Verify compiler optimizations | Check if FMA fusion, loop unrolling happened |
| Understand instruction latency | SASS reveals memory access patterns |
| Architecture comparison | See how same PTX maps to different SASS |

### Quick Comparison

| Aspect | PTX | SASS |
|--------|-----|------|
| Level | Virtual (IR) | Native machine code |
| Portability | Cross-architecture | Architecture-specific |
| Registers | Virtual (unlimited) | Physical (limited, e.g., 255/thread) |
| Use case | Distribution, JIT | Performance analysis |
| Readability | Higher | Lower |

### Tools for Analysis

```bash
nvcc -ptx -arch=sm_80 kernel.cu           # Generate PTX
nvcc -cubin -arch=sm_80 kernel.cu         # Generate cubin
cuobjdump -sass kernel.cubin              # View SASS
cuobjdump -ptx -sass kernel               # Both from binary
ncu --set full ./your_program             # Nsight Compute profiling
```

### Example: FMA Optimization

CUDA code:
```cuda
float a = x * y + z;  // Hoping for FMA
```

PTX (may or may not fuse):
```asm
mul.f32 %f3, %f1, %f2;
add.f32 %f4, %f3, %f0;
// OR (if fused)
fma.rn.f32 %f3, %f1, %f2, %f0;
```

SASS (Ampere) - shows what actually executes:
```asm
FFMA R3, R1, R2, R0;  // Definitely fused - good!
```

---

## Warp-Level Primitives

A **warp** is 32 threads that execute in lockstep (SIMT). Warp-level primitives allow direct communication between threads within a warp **without shared memory**.

### Why Use Warp Primitives?

| Approach | Latency | Synchronization |
|----------|---------|-----------------|
| Shared memory | ~20-30 cycles | Requires `__syncthreads()` |
| Warp shuffle | ~2-4 cycles | Implicit (lockstep) |

### Shuffle Operations (`__shfl_*`)

Exchange data directly between threads in a warp via registers.

```cuda
// All threads must participate (or use __shfl_sync with mask)
int val = __shfl_sync(0xffffffff, src_val, src_lane);  // Get value from specific lane
int val = __shfl_up_sync(0xffffffff, src_val, delta);  // Get from lane - delta
int val = __shfl_down_sync(0xffffffff, src_val, delta); // Get from lane + delta
int val = __shfl_xor_sync(0xffffffff, src_val, mask);  // Get from lane ^ mask
```

**Key parameters**:
- `mask`: Bitmask of participating threads (0xffffffff = all 32)
- `src_val`: Value to share from this thread
- `src_lane`/`delta`/`mask`: Target lane calculation

**Example: Warp-level reduction (sum)**

```cuda
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;  // Only lane 0 has final result
}
```

```text
Initial:     [a0, a1, a2, a3, ... a31]
offset=16:   [a0+a16, a1+a17, ..., a15+a31, ...]
offset=8:    [a0+a8+a16+a24, ...]
offset=4:    ...
offset=2:    ...
offset=1:    [sum of all 32 values in lane 0]
```

### Vote Operations (`__ballot_*`, `__all_*`, `__any_*`)

Collective boolean operations across warp.

```cuda
// Ballot: each thread contributes 1 bit based on predicate
unsigned mask = __ballot_sync(0xffffffff, predicate);
// Returns 32-bit mask where bit i = predicate of thread i

// All: true if ALL threads have predicate=true
int result = __all_sync(0xffffffff, predicate);

// Any: true if ANY thread has predicate=true
int result = __any_sync(0xffffffff, predicate);
```

**Example: Count active elements**

```cuda
int count = __popc(__ballot_sync(0xffffffff, data[tid] > threshold));
// __popc = population count (count set bits)
```

### Match Operations (`__match_*`)

Find threads with matching values (Volta+).

```cuda
// Returns mask of threads with same value
unsigned mask = __match_any_sync(0xffffffff, value);

// Returns mask only if ALL threads match, else 0
unsigned mask = __match_all_sync(0xffffffff, value, &pred);
```

### Activemask

Get mask of currently active threads (useful with divergence).

```cuda
unsigned active = __activemask();  // Which threads are executing this line
```

### Common Patterns

**Pattern 1: Broadcast from lane 0**
```cuda
float shared_val = __shfl_sync(0xffffffff, my_val, 0);  // All get lane 0's value
```

**Pattern 2: Parallel prefix sum (scan)**
```cuda
for (int d = 1; d < 32; d *= 2) {
    float n = __shfl_up_sync(0xffffffff, val, d);
    if (lane_id >= d) val += n;
}
```

**Pattern 3: Warp-level reduction then broadcast**
```cuda
float sum = warpReduceSum(val);
sum = __shfl_sync(0xffffffff, sum, 0);  // Broadcast result to all lanes
```

**Pattern 4: Butterfly reduction (using XOR)**
```cuda
for (int mask = 16; mask > 0; mask /= 2)
    val += __shfl_xor_sync(0xffffffff, val, mask);
// All threads get the final sum
```

### Warp Intrinsics Summary

| Function | Purpose |
|----------|---------|
| `__shfl_sync` | Read from specific lane |
| `__shfl_up_sync` | Read from lower lane (lane - delta) |
| `__shfl_down_sync` | Read from higher lane (lane + delta) |
| `__shfl_xor_sync` | Read from XOR'd lane (lane ^ mask) |
| `__ballot_sync` | Collect predicates into bitmask |
| `__all_sync` | AND of all predicates |
| `__any_sync` | OR of all predicates |
| `__match_any_sync` | Find threads with same value |
| `__activemask` | Get mask of active threads |
| `__popc` | Count set bits (population count) |

### Performance Tips

1. **Prefer shuffles over shared memory** for warp-local communication
2. **Use full warp participation** (mask = 0xffffffff) when possible for best performance
3. **Combine with `__syncwarp()`** if threads might diverge before shuffle
4. **Lane ID**: `int lane = threadIdx.x % 32;` or `threadIdx.x & 31`

---

## Nsight Compute Metrics Guide

When profiling CUDA kernels, focus on these metrics in order of importance.

### 1. The Roofline Model (Start Here)

The **roofline** tells you if your kernel is **compute-bound** or **memory-bound**.

**Key metric**: `Arithmetic Intensity = FLOPs / Bytes transferred`

| AI Value | Kernel Type | Optimization Focus |
|----------|-------------|-------------------|
| Low (<10) | Memory-bound | Reduce memory traffic, improve coalescing |
| High (>10) | Compute-bound | Increase ILP, use Tensor Cores |

### 2. Top-Level Metrics (GPU Speed of Light)

**SOL (Speed of Light)** shows how close you are to hardware limits.

| Metric | What it measures | Target |
|--------|------------------|--------|
| `SM Throughput (%)` | Compute utilization | >80% for compute-bound |
| `Memory Throughput (%)` | Memory bandwidth utilization | >80% for memory-bound |
| `Achieved Occupancy` | Active warps / max warps | Higher is usually better |

**Interpretation**:
- High SM%, Low Memory% → Compute-bound (good for GEMM)
- Low SM%, High Memory% → Memory-bound (typical for elementwise ops)
- Low both → Something wrong (stalls, low occupancy)

### 3. Memory Metrics (Most Common Bottleneck)

**Global Memory Access**:

| Metric | Meaning | Target |
|--------|---------|--------|
| `Global Load/Store Efficiency` | Useful bytes / total bytes transferred | >80% (ideally 100%) |
| `L2 Hit Rate` | Cache effectiveness | Higher = fewer HBM accesses |
| `DRAM Throughput` | HBM bandwidth used | Compare to peak (e.g., 2TB/s on H100) |

**Coalescing check**:
```text
Efficiency = (requested bytes) / (actual bytes transferred)

Perfect coalescing:  32 threads × 4 bytes = 128 bytes requested, 128 bytes transferred → 100%
Bad access pattern:  32 threads × 4 bytes = 128 bytes requested, 1024 bytes transferred → 12.5%
```

**Shared Memory**:

| Metric | Meaning | Target |
|--------|---------|--------|
| `Shared Memory Efficiency` | Useful bytes / bytes moved | >80% |
| `Bank Conflicts/Request` | Serialization due to bank conflicts | 0 (ideal), <2 (acceptable) |

### 4. Compute Metrics

| Metric | Meaning | What to check |
|--------|---------|---------------|
| `Executed IPC` | Instructions per cycle | Higher = better utilization |
| `Warp Execution Efficiency` | Active threads / 32 | Low = warp divergence |
| `Tensor Core Utilization` | % of Tensor Core peak | Should be high for GEMM/attention |

### 5. Occupancy Analysis

```text
Occupancy = Active Warps per SM / Max Warps per SM
```

**Limiters** (check which one is limiting):

| Limiter | Meaning | Fix |
|---------|---------|-----|
| Registers | Too many registers/thread | Reduce register usage, use `__launch_bounds__` |
| Shared Memory | Block uses too much smem | Reduce tile size, multi-stage pipeline |
| Block Size | Block too small/large | Tune block dimensions |
| Max Blocks/SM | Hardware limit hit | Usually fine |

**Note**: Higher occupancy ≠ always better. Sometimes lower occupancy with more registers/smem per thread wins.

### 6. Stall Reasons (Why Warps Wait)

| Stall Reason | Cause | Fix |
|--------------|-------|-----|
| `Long Scoreboard` | Waiting for memory | Prefetch, increase ILP, hide latency |
| `Wait` | Barrier sync (`__syncthreads`) | Reduce sync frequency |
| `Not Selected` | Other warps scheduled | Usually fine |
| `Short Scoreboard` | Waiting for math result | Increase ILP |
| `MIO Throttle` | Memory instruction queue full | Reduce memory pressure |

### 7. Practical Analysis Workflow

```text
1. Check Roofline
   └─→ Memory-bound? Focus on memory metrics
   └─→ Compute-bound? Focus on compute metrics

2. Check SOL percentages
   └─→ Both low? Check occupancy and stalls

3. For memory-bound kernels:
   ├─ Global Load Efficiency < 80%? → Fix coalescing
   ├─ L2 Hit Rate low? → Improve locality, tiling
   ├─ Bank conflicts > 0? → Pad shared memory
   └─ DRAM throughput < 70% peak? → More parallelism

4. For compute-bound kernels:
   ├─ Tensor Core util low? → Use mma instructions
   ├─ IPC low? → Increase ILP, unroll loops
   └─ Warp efficiency low? → Fix divergence
```

### 8. Quick Reference Commands

```bash
# Full analysis
ncu --set full -o profile ./kernel

# Roofline only
ncu --set roofline ./kernel

# Specific metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed ./kernel

# Compare two kernels
ncu --set full -o baseline ./kernel_v1
ncu --set full -o optimized ./kernel_v2
# Then compare in GUI
```

### 9. Key Metrics Cheatsheet

| Goal | Primary Metrics |
|------|-----------------|
| Am I memory or compute bound? | Roofline, SM vs Memory SOL |
| Is my memory access efficient? | Global Load/Store Efficiency, L2 Hit Rate |
| Am I using hardware well? | Achieved Occupancy, Tensor Core Utilization |
| Why are warps stalling? | Warp Stall Reasons breakdown |
| Bank conflicts? | Shared Memory Bank Conflicts |

### 10. Common Issues & Indicators

| Symptom | Likely Cause |
|---------|--------------|
| Low Global Load Efficiency (~25%) | Strided or random access, not coalesced |
| High "Long Scoreboard" stalls | Memory latency not hidden, need more ILP |
| Low occupancy (limited by registers) | Too many registers, consider `__launch_bounds__` |
| Bank conflicts > 0 | Shared memory access pattern causes serialization |
| Low SM throughput, low Memory throughput | Launch config issue, not enough parallelism |

---

## Resources

### Blogs & Tutorials

- **[How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)** by Simon Boehm
  - GitHub: [siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA)
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

### Code References (Production CUDA)

- [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) - CUDA C++ template library for high-performance GEMM
- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) - Original FlashAttention implementation
- [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) - CUDA kernels for LLM serving
- [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - NVIDIA's LLM inference library

### Related Notes

- [GPU Fundamentals](01_GPU_Fundamentals.md) - Thread/memory hierarchy basics
- [Triton Programming](04_Triton_Programming.md) - Higher-level kernel programming
