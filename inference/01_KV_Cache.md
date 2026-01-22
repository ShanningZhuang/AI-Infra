# KV Cache & Memory Management

> Parent: [Inference Overview](inference.md)

## Overview

The KV (Key-Value) Cache is the most fundamental optimization in LLM inference. It stores computed key and value tensors from the attention mechanism, avoiding redundant computation during autoregressive generation.

## Learning Objectives

- [x] KV Cache mechanism and why it's needed
- [ ] Memory requirements calculation
- [ ] PagedAttention (vLLM's innovation)
- [ ] Prefix caching and KV cache reuse
- [ ] Memory management strategies

## Resources

### Papers

- [vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) - RadixAttention
- [ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition](https://arxiv.org/abs/2402.15220)

### Blogs & Tutorials

- [vLLM PagedAttention Blog](https://blog.vllm.ai/2023/06/20/vllm.html)
- [Efficient Memory Management for LLMs](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)

### Code References

- [vLLM Block Manager](https://github.com/vllm-project/vllm/blob/main/vllm/core/block_manager.py)
- [HuggingFace KV Cache Implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py)

---

## Notes

### Why KV Cache?

In standard attention, for each token we compute:
```
Attention(Q, K, V) = softmax(QK^T / √d) * V
```

Without cache, generating token `t` requires:
- Recomputing K and V for all previous tokens [0, 1, ..., t-1]
- O(t²) total compute for sequence of length t

With cache:
- Store K, V from all previous tokens
- Only compute K, V for new token
- O(t) total compute

### KV Cache Memory Formula

For a single sequence:
```
KV_memory = 2 × num_layers × num_kv_heads × head_dim × seq_len × bytes_per_element
```

Example for Llama-2-70B (FP16):
```
num_layers = 80
num_kv_heads = 8 (GQA)
head_dim = 128
seq_len = 4096
bytes = 2 (FP16)

KV_memory = 2 × 80 × 8 × 128 × 4096 × 2 = 1.34 GB per sequence!
```

### The Memory Fragmentation Problem

Traditional KV cache allocation:

```
┌────────────────────────────────────────────────────────┐
│ Request 1: ████████░░░░░░░░░░░░  (allocated max_len)   │
│ Request 2: ██████████████░░░░░░  (allocated max_len)   │
│ Request 3: ██░░░░░░░░░░░░░░░░░░  (allocated max_len)   │
│ Request 4: CANNOT FIT (fragmented memory)              │
└────────────────────────────────────────────────────────┘
                              ↑
                    Wasted memory (internal fragmentation)
```

Problems:
1. **Over-allocation**: Must reserve max possible sequence length
2. **Fragmentation**: Can't use gaps between allocations
3. **Low utilization**: Often < 50% of allocated memory is used

### PagedAttention Solution

Inspired by OS virtual memory, PagedAttention uses fixed-size blocks:

```
Physical KV Memory (GPU):
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ B0  │ B1  │ B2  │ B3  │ B4  │ B5  │ B6  │ B7  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

Logical View (per request):
Request 1: [B0] → [B3] → [B5]         (3 blocks, non-contiguous)
Request 2: [B1] → [B2] → [B4] → [B7]  (4 blocks, non-contiguous)
Request 3: [B6]                        (1 block)
```

Key insight: Use a **block table** (like page table) to map logical → physical

```python
# Block table maps sequence position to physical block
block_table = {
    "request_1": [0, 3, 5],      # block indices
    "request_2": [1, 2, 4, 7],
    "request_3": [6],
}
```

### PagedAttention Benefits

| Aspect | Traditional | PagedAttention |
|--------|-------------|----------------|
| Memory waste | ~60-80% | < 4% |
| Max batch size | Limited | 2-4x higher |
| Memory sharing | Not possible | Supported |
| Dynamic growth | Expensive | Free |

### Prefix Caching (KV Cache Reuse)

Many requests share common prefixes (system prompts, few-shot examples):

```
Request 1: "You are a helpful assistant. User: What is 2+2?"
Request 2: "You are a helpful assistant. User: What is the capital of France?"
                ↑ Same prefix ↑

With prefix caching:
┌────────────────────────────────────┐
│ Shared KV blocks for system prompt │ ← Computed once
├────────────────────────────────────┤
│ Request 1 unique KV               │
├────────────────────────────────────┤
│ Request 2 unique KV               │
└────────────────────────────────────┘
```

### KV Cache Reuse Performance Impact

The performance gain from KV cache reuse is dramatic:

```
Without Cache Reuse:
- Process 1000 token prompt: ~2 seconds
- Process 1005 token prompt (5 new tokens): ~2 seconds
- Total: 4 seconds

With Cache Reuse:
- Process 1000 token prompt: ~2 seconds
- Process 1005 token prompt (reuse 1000, compute 5): ~0.05 seconds
- Total: 2.05 seconds (2x faster!)
```

**Real-World Use Cases:**
1. **Multi-turn Conversations**: Follow-up messages share conversation history
2. **Batch Processing**: Multiple prompts with common system prompts
3. **RAG Systems**: Queries sharing retrieval context
4. **Code Completion**: Iterative edits share file prefixes
5. **Agent Systems**: Multiple tool calls sharing system context

### RadixAttention (SGLang)

SGLang uses a **radix tree (trie)** to track all cached prefixes. A radix tree is a space-optimized trie where nodes with single children are merged.

**Regular Trie** (uncompressed):
```
    root
     |
     1
     |
     2
    / \
   3   6
```

**Radix Tree** (compressed):
```
    root
     |
   [1,2]  ← Compressed common prefix
    / \
  [3] [6]
```

**How RadixAttention manages KV cache:**

```
Request 1: [1, 2, 3, 4, 5]     # "Explain Python"
Request 2: [1, 2, 3, 6, 7]     # "Explain Python classes"
Request 3: [1, 2, 8, 9, 10]    # "Explain JavaScript"

RadixAttention builds this cache tree:
        root
         |
      [1, 2] ← Shared KV cache
       /   \
    [3]   [8, 9, 10]
    / \
 [4,5] [6,7]
   ↑     ↑
Req1   Req2   Req3
```

**Memory Savings Example:**

Without RadixAttention (separate caches):
```
Request 1: Cache 1000 tokens
Request 2: Cache 1000 + 200 tokens (duplication!)
Request 3: Cache 1000 + 150 tokens (more duplication!)
Total: 3,350 tokens × KV size
```

With RadixAttention (shared cache nodes):
```
Request 1: Cache 1000 tokens
Request 2: Reuse 1000, cache only +200 tokens
Request 3: Reuse 1000, cache only +150 tokens
Total: 1,350 tokens × KV size (60% savings!)
```

**Core Features:**
- **Automatic Prefix Detection**: No manual tracking needed
- **Cross-Request Sharing**: Different users share cache if prompts overlap
- **LRU Eviction**: When memory full, least recently used branches evicted
- **Tree Pruning**: Unused branches cleaned up automatically

### Trie-Based Prefix Matching

For efficient prefix lookup with O(k) complexity:

```python
class TrieNode:
    def __init__(self):
        self.children = {}       # token -> TrieNode
        self.worker_ids = []     # Workers with this prefix cached

class PrefixSchedulerTrie:
    def __init__(self, workers):
        self.root = TrieNode()
        self._build_trie(workers)

    def _build_trie(self, workers):
        """Build prefix tree from all worker caches. Time: O(T)"""
        for worker in workers:
            node = self.root
            for token in worker.cache_tokens:
                if token not in node.children:
                    node.children[token] = TrieNode()
                node = node.children[token]
                node.worker_ids.append(worker.worker_id)

    def dispatch(self, prompt_tokens):
        """Find worker with longest prefix match. Time: O(k)"""
        node = self.root
        last_worker = None

        for token in prompt_tokens:
            if token not in node.children:
                break
            node = node.children[token]
            if node.worker_ids:
                last_worker = node.worker_ids[-1]

        return last_worker
```

**Complexity Comparison:**

| Operation | Naive O(n×m) | Trie O(k) |
|-----------|--------------|-----------|
| 100 workers, 1000 tokens | 100,000 ops | 1,000 ops |
| Dispatch at 1000 req/s | Bottleneck | Easy |

### Memory Management Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| Preallocation | Reserve max length upfront | Simple, predictable |
| Dynamic | Grow as needed | Memory efficient |
| Paged | Fixed-size blocks | Best utilization |
| Prefix sharing | Reuse common prefixes | Multi-turn, batch |

### Practical Considerations

**Block size selection:**
- Too small: More overhead from block table lookups
- Too large: More internal fragmentation
- Typical: 16-256 tokens per block

**Swap to CPU:**
- When GPU memory full, swap blocks to CPU
- Resume when space available (preemption)
- vLLM supports this automatically

### Code Example: Simple KV Cache

```python
class KVCache:
    def __init__(self, num_layers, max_batch, max_seq_len, num_heads, head_dim):
        # Pre-allocate cache tensors
        self.k_cache = torch.zeros(
            num_layers, max_batch, max_seq_len, num_heads, head_dim
        )
        self.v_cache = torch.zeros(
            num_layers, max_batch, max_seq_len, num_heads, head_dim
        )
        self.seq_lens = torch.zeros(max_batch, dtype=torch.int32)

    def update(self, layer_idx, batch_idx, k, v):
        pos = self.seq_lens[batch_idx]
        self.k_cache[layer_idx, batch_idx, pos] = k
        self.v_cache[layer_idx, batch_idx, pos] = v
        self.seq_lens[batch_idx] += 1

    def get(self, layer_idx, batch_idx):
        seq_len = self.seq_lens[batch_idx]
        return (
            self.k_cache[layer_idx, batch_idx, :seq_len],
            self.v_cache[layer_idx, batch_idx, :seq_len]
        )
```

### KV Cache Compression Techniques

| Technique | Description | Trade-off |
|-----------|-------------|-----------|
| Quantized KV | Store KV in INT8/FP8 | 2-4x memory, slight quality loss |
| Sliding window | Only keep recent K tokens | Fixed memory, loses long context |
| Sparse attention | Only cache important tokens | Reduced memory, selection overhead |
| MQA/GQA | Fewer KV heads | Built into model architecture |

### Multi-Query Attention (MQA) vs Grouped-Query Attention (GQA)

```
MHA (Multi-Head Attention):
Q: [num_heads × head_dim]
K: [num_heads × head_dim]  ← Full KV per head
V: [num_heads × head_dim]

MQA (Multi-Query Attention):
Q: [num_heads × head_dim]
K: [1 × head_dim]          ← Single KV shared across all heads
V: [1 × head_dim]

GQA (Grouped-Query Attention):
Q: [num_heads × head_dim]
K: [num_kv_heads × head_dim]  ← num_kv_heads < num_heads
V: [num_kv_heads × head_dim]
```

GQA is used in Llama-2/3, Mistral, and most modern models.

## My notes

