# Matrix Multiplication Optimization

## Overview

Matrix multiplication is the most critical operation in deep learning. Understanding how to optimize it is fundamental to AI infrastructure.

> For Triton implementation, see [Triton Programming](04_Triton_Programming.md).

---

## Mathematical Definition

Given two matrices:
- **A** ∈ ℝ^(M×K)
- **B** ∈ ℝ^(K×N)

The matrix multiplication **C = AB** produces **C** ∈ ℝ^(M×N) where:

$$C_{ij} = \sum_{p=1}^{K} A_{ip} \cdot B_{pj}$$

Each element C[i,j] is the dot product of the i-th row of A and the j-th column of B.

---

## Tiled Computation

### Why Tiling?

Loading entire matrices from global memory is slow. Instead, we divide matrices into **tiles** that fit in fast shared memory.

```
Matrix C (M × N) - divided into blocks
Each block is BLOCK_SIZE_M × BLOCK_SIZE_N

     N (columns)
   ┌─────────────────────────────────┐
   │ B0  │ B1  │ B2  │ B3  │ B4  │...│
   ├─────┼─────┼─────┼─────┼─────┼───┤
M  │ B5  │ B6  │ B7  │ B8  │ B9  │...│  Each block is
   ├─────┼─────┼─────┼─────┼─────┼───┤  computed by one
   │ B10 │ B11 │ B12 │ B13 │ B14 │...│  parallel program
   └─────────────────────────────────┘
```

### The K-dimension Loop

To compute one output block C[i,j], we accumulate partial products:

```
C[i,j] = A[i,0]×B[0,j] + A[i,1]×B[1,j] + A[i,2]×B[2,j] + ...

Iteration k=0:
A block           B block              Accumulator
┌─────────┐      ┌─────────┐          ┌─────────┐
│ A[i,0]  │  ×   │ B[0,j]  │    →     │  acc    │
│(M×K_blk)│      │(K_blk×N)│          │ (M×N)   │
└─────────┘      └─────────┘          └─────────┘
                                       acc += A×B

Iteration k=1:
┌─────────┐      ┌─────────┐          ┌─────────┐
│ A[i,1]  │  ×   │ B[1,j]  │    →     │  acc    │
└─────────┘      └─────────┘          └─────────┘
                                       acc += A×B

... continue until all K blocks processed ...
```

---

## Cache Optimization: Grouped Ordering

### The Problem

L2 cache is **limited** and cannot hold all data simultaneously. The order in which we process blocks matters for cache efficiency.

```
Typical scenario:
- Matrix size: 1024×1024, divided into 8×8 = 64 blocks
- Block size: 128×128 elements
- Each block: 128×128 × 2 bytes (FP16) = 32KB
- L2 cache: Can hold ~2-4 blocks at a time

Key insight: We CANNOT keep all B columns in cache!
```

### Standard Ordering (Row-by-Row)

```
Execution order: P0, P1, P2, P3, P4, P5, P6, P7, P8, ...

┌───┬───┬───┬───┬───┬───┬───┬───┐
│P0 │P1 │P2 │P3 │P4 │P5 │P6 │P7 │ ← Row 0 (executed first)
├───┼───┼───┼───┼───┼───┼───┼───┤
│P8 │P9 │P10│P11│P12│P13│P14│P15│ ← Row 1 (executed second)
└───┴───┴───┴───┴───┴───┴───┴───┘
```

**Cache behavior:**
```
Time | Program | Needs         | Cache State
-----|---------|---------------|------------------
  0  |   P0    | A[r0], B[c0]  | A[r0], B[c0]
  1  |   P1    | A[r0], B[c1]  | A[r0], B[c1]  ← B[c0] evicted
  2  |   P2    | A[r0], B[c2]  | A[r0], B[c2]  ← B[c1] evicted
  ...
  8  |   P8    | A[r1], B[c0]  | Must RELOAD B[c0]! ❌
```

**Problem:** B[c0] was evicted before we needed it again for row 1.

### Grouped Ordering (Column-First within Groups)

```
Execution order: P0, P8, P16, P24, P1, P9, P17, P25, ...

┌───┬───┬───┬───┐
│ 0 │ 4 │ 8 │12 │  Process column 0 first (rows 0,1,2,3)
├───┼───┼───┼───┤  Then column 1 (rows 0,1,2,3)
│ 1 │ 5 │ 9 │13 │  ...
├───┼───┼───┼───┤
│ 2 │ 6 │10 │14 │
├───┼───┼───┼───┤
│ 3 │ 7 │11 │15 │
└───┴───┴───┴───┘
```

**Cache behavior:**
```
Time | Program | Needs         | Cache State
-----|---------|---------------|------------------
  0  |   P0    | A[r0], B[c0]  | A[r0], B[c0]
  1  |   P8    | A[r1], B[c0]  | A[r1], B[c0]  ← B[c0] REUSED! ✓
  2  |   P16   | A[r2], B[c0]  | A[r2], B[c0]  ← B[c0] REUSED! ✓
  3  |   P24   | A[r3], B[c0]  | A[r3], B[c0]  ← B[c0] REUSED! ✓
  4  |   P1    | A[r0], B[c1]  | A[r0], B[c1]  ← B[c0] done, load B[c1]
```

### Comparison

```
WITHOUT Grouping:
- A rows: Excellent reuse (used 8 times consecutively) ✓
- B columns: TERRIBLE reuse (evicted before reuse) ❌
- Total B loads: 8 columns × 4 rows = 32 loads

WITH Grouping:
- B columns: Excellent reuse (used 4 times immediately) ✓
- A rows: Moderate reuse (some reloading) ⚠️
- Total B loads: 8 columns × 1 load each = 8 loads

Net Result: ~20-30% reduction in memory traffic!
```

### Why Grouping Wins

The key insight is **temporal locality** - keeping data "hot" in cache by using it multiple times in quick succession before moving on.

```
Without grouping: Use B[c0] → wait many cycles → need B[c0] again → MISS
With grouping:    Use B[c0] → use B[c0] → use B[c0] → done with B[c0]
```

### Choosing GROUP_SIZE

```python
if M >> N:  # Tall matrix (many rows, few columns)
    Use GROUP_SIZE_M (group by columns)
    → Optimize for B column reuse

elif N >> M:  # Wide matrix (few rows, many columns)
    Use GROUP_SIZE_N (group by rows)
    → Optimize for A row reuse

else:  # Square matrix
    Use 2D grouping
    → Balance A and B reuse
```

**Typical values:** GROUP_SIZE_M = 4-8

---

## Performance Impact

In practice with typical matrix sizes:
- **Without grouping**: ~30-40% cache miss rate on B matrix
- **With grouping**: ~10-15% cache miss rate on B matrix

This translates to **20-30% performance improvement**!

---

## Summary

Optimizing matrix multiplication requires:

1. **Tiled computation** - Divide matrices into blocks that fit in shared memory
2. **Grouped ordering** - Process blocks in an order that maximizes cache reuse
3. **Memory access patterns** - Ensure coalesced global memory access
4. **Accumulator management** - Compute in FP32, write in FP16

---

## Resources

### Blogs & Tutorials

- [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) by Simon Boehm
- [Tiled Matrix Multiplication](https://penny-xu.github.io/blog/tiled-matrix-multiplication)

### Related Notes

- [Triton Programming](04_Triton_Programming.md) - Implementation with Triton
- [GPU Fundamentals](01_GPU_Fundamentals.md) - Memory hierarchy basics
- [CUDA Advanced](03_CUDA_Advanced.md) - Low-level optimization techniques
