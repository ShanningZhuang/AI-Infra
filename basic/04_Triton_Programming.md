# Triton Programming

## Overview

Triton is a Python-based language for writing GPU kernels that's easier than CUDA but still gives you low-level control. It enables kernel fusion and custom GPU operations.

## Learning Objectives

- [x] Why kernel fusion matters (memory bandwidth)
- [x] Triton programming basics (grid, program_id, constexpr)
- [x] Memory access with make_block_ptr and strides
- [x] How PyTorch 2.0 `torch.compile` uses Triton

---

## Why Kernel Fusion?

The main bottleneck in GPU computing is often **memory bandwidth**, not compute.

### Without Fusion

```
GPU Memory → Kernel 1 → GPU Memory → Kernel 2 → GPU Memory
            (compute)   (write/read)  (compute)
```

Each kernel:
1. Reads inputs from global memory (slow)
2. Computes (fast)
3. Writes outputs to global memory (slow)

### With Fusion

```
GPU Memory → Fused Kernel → GPU Memory
             (compute)
```

One kernel:
1. Reads inputs once
2. Computes multiple operations (keeping intermediates in registers/shared memory)
3. Writes final output once

**Saves memory bandwidth by 2-10x!**

---

## Triton Basics

### The Programming Model

Think of a GPU as having thousands of small processors that can run the same code simultaneously on different data.

```python
# HOST CODE (Python/PyTorch)
grid = (num_blocks,)  # How many parallel programs to launch
kernel[grid](args)    # Launch kernel with grid configuration

# DEVICE CODE (Triton kernel on GPU)
@triton.jit
def kernel(...):
    pid = tl.program_id(0)  # Each program knows its ID
    # Process different data based on pid
```

### Grid and Program Instances

The **grid** defines how many **parallel program instances** to launch.

```python
grid = (triton.cdiv(N, BLOCK_SIZE),)
# Example: If N=1024, BLOCK_SIZE=64
# grid = (16,) = 16 program instances running in parallel
```

Each instance processes a **different chunk** of data:
- Program 0: elements 0-63
- Program 1: elements 64-127
- Program 2: elements 128-191
- ...

### `tl.program_id()` - Which Worker Am I?

```python
pid = tl.program_id(0)  # My position in dimension 0
```

Each parallel instance needs to know **which chunk of data it's responsible for**.

### `tl.constexpr` - Compile-Time Constants

```python
BLOCK_SIZE: tl.constexpr  # Known at compile time
```

Triton can **optimize the code** knowing these values ahead of time:
- Unroll loops
- Allocate exact memory sizes
- Generate specialized code

---

## Example 1: Vector Addition

The simplest Triton kernel.

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # 1. Which block am I?
    pid = tl.program_id(0)

    # 2. Calculate my element indices
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 3. Create mask for bounds checking
    mask = offsets < n_elements

    # 4. Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 5. Compute
    output = x + y

    # 6. Store result
    tl.store(output_ptr + offsets, output, mask=mask)

# Launch kernel
def add(x, y):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

---

## Memory Access: Strides and Block Pointers

### Understanding Strides

Memory is 1D, but tensors are multidimensional. **Strides** tell you how to navigate.

```python
Q.shape = [8, 1024, 64]  # [batch, seq, dim]
Q.stride() = (65536, 64, 1)
```

| Dimension | Stride | Meaning |
|-----------|--------|---------|
| 0 (batch) | 65536 | Skip 65,536 elements to next batch |
| 1 (seq) | 64 | Skip 64 elements to next position |
| 2 (dim) | 1 | Skip 1 element to next feature |

**Formula:** To access `Q[b, s, d]`:
```python
memory_index = b * stride[0] + s * stride[1] + d * stride[2]
```

### `tl.make_block_ptr()` - Smart Data Access

Creates a pointer to a 2D block of data with automatic bounds checking.

```python
Q_block_ptr = tl.make_block_ptr(
    Q_ptr + batch_index * stride_qb,  # Base address (offset to correct batch)
    shape=(N_QUERIES, D),              # Total tensor shape
    strides=(stride_qq, stride_qd),    # How to move in memory
    offsets=(tile_index * TILE_SIZE, 0),  # Where this block starts
    block_shape=(TILE_SIZE, D),        # Size of block to load
    order=(1, 0),                       # Memory layout order
)

# Load the block
Q_tile = tl.load(Q_block_ptr)
```

**Visual Example:**
```
Q tensor for one batch: [1024 queries, 64 dimensions]

┌─────────────────────────────────────┐
│ Tile 0 (rows 0-63)    ← Program 0   │
├─────────────────────────────────────┤
│ Tile 1 (rows 64-127)  ← Program 1   │
├─────────────────────────────────────┤
│ Tile 2 (rows 128-191) ← Program 2   │
├─────────────────────────────────────┤
│ ...                                  │
└─────────────────────────────────────┘

Each program loads its tile using make_block_ptr
```

---

## Example 2: Matrix Multiplication

A more complex example showing tiled computation.

### Algorithm Overview

```
Matrix C (M × N) - divided into blocks
Each program computes ONE block of output

     N (columns)
   ┌─────────────────────────────────┐
   │ B0  │ B1  │ B2  │ B3  │ B4  │...│
   ├─────┼─────┼─────┼─────┼─────┼───┤
M  │ B5  │ B6  │ B7  │ B8  │ B9  │...│
   ├─────┼─────┼─────┼─────┼─────┼───┤
   │ B10 │ B11 │ B12 │ B13 │ B14 │...│
   └─────────────────────────────────┘

Program 0 → Block 0
Program 1 → Block 1
...
All programs run in PARALLEL
```

### The K-dimension Loop

```
For Block C[i,j]:
C[i,j] = A[i,0]×B[0,j] + A[i,1]×B[1,j] + A[i,2]×B[2,j] + ...

Iteration k=0:
┌─────────┐     ┌─────────┐     ┌─────────┐
│ A[i,0]  │  ×  │ B[0,j]  │  →  │  acc    │
│(M×K_blk)│     │(K_blk×N)│     │ (M×N)   │
└─────────┘     └─────────┘     └─────────┘
                                 acc += A×B

... continue until all K blocks processed ...
```

### Simplified Matmul Kernel

```python
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 1. Determine which output block this program computes
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 2. Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 3. Create block pointers
    A_block_ptr = tl.make_block_ptr(
        A_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    B_block_ptr = tl.make_block_ptr(
        B_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    # 4. Loop over K dimension
    for k in range(0, K, BLOCK_K):
        A_tile = tl.load(A_block_ptr)
        B_tile = tl.load(B_block_ptr)
        acc += tl.dot(A_tile, B_tile)

        # Advance pointers
        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))

    # 5. Store result
    C_block_ptr = tl.make_block_ptr(
        C_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(C_block_ptr, acc)
```

---

## Grid Layout Best Practices

### Primary Dimension First

Put the dimension with **more parallelism** in dimension 0.

```python
# Good: query tiles (many) in dim 0
grid = (num_query_tiles, batch_size)  # e.g., (64, 8)

# Less good: batch (few) in dim 0
grid = (batch_size, num_query_tiles)  # e.g., (8, 64)
```

**Why?** GPUs schedule dimension 0 more efficiently.

### Grouped Ordering for Cache Efficiency

Process blocks in an order that maximizes cache reuse:

```
Standard ordering:          Grouped ordering:
┌───┬───┬───┬───┐          ┌───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │          │ 0 │ 4 │ 8 │12 │
├───┼───┼───┼───┤          ├───┼───┼───┼───┤
│ 4 │ 5 │ 6 │ 7 │          │ 1 │ 5 │ 9 │13 │
├───┼───┼───┼───┤          ├───┼───┼───┼───┤
│ 8 │ 9 │10 │11 │          │ 2 │ 6 │10 │14 │
└───┴───┴───┴───┘          └───┴───┴───┴───┘

Grouped ordering processes same-column blocks together
→ Better L2 cache reuse for B matrix!
```

---

## torch.compile Integration

PyTorch 2.0's `torch.compile` uses Triton under the hood.

### How It Works

```
Python/PyTorch code
        ↓
   TorchDynamo (traces code)
        ↓
   TorchInductor (generates Triton)
        ↓
   Compiled GPU kernels
```

### When It Helps

| Scenario | Speedup | Why |
|----------|---------|-----|
| Many small ops | 20-50%+ | Kernel fusion reduces overhead |
| Custom functions | 10-30% | Eliminates Python overhead |
| Large matmuls | 5-10% | Already optimized (cuBLAS) |

### Usage

```python
model = MyModel()
compiled_model = torch.compile(model)  # One line!

# Or with options
compiled_model = torch.compile(model, mode="reduce-overhead")
compiled_model = torch.compile(model, mode="max-autotune")
```

### Why Modest Speedups for Transformers

- Core operations (matmul, attention) are **already highly optimized**
- `torch.compile` shines on **fusible ops** (activations, layernorms)
- Large matmuls dominate and are often memory-bound

---

## Resources

### Papers

- "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations" (Tillet et al.)

### Blogs & Tutorials

- [Triton Documentation](https://triton-lang.org/)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [PyTorch 2.0 torch.compile tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)

### Related Notes

- [GPU Fundamentals](01_GPU_Fundamentals.md) - Thread/memory hierarchy
- [CUDA Advanced](03_CUDA_Advanced.md) - Lower-level GPU programming
- [Matrix Multiplication](05_Matrix_Multiplication.md) - Cache optimization theory
- [FlashAttention](06_FlashAttention_CS336.md) - Memory-efficient attention algorithm
