# Speculative Decoding

> Parent: [Inference Overview](inference.md)

## Overview

Speculative decoding is a technique to speed up autoregressive generation without changing the model's output distribution. The key insight: use a small, fast "draft" model to propose multiple tokens, then verify them in parallel with the large "target" model.

## Learning Objectives

- [ ] Why speculative decoding works
- [ ] Draft-verify paradigm
- [ ] Acceptance/rejection sampling
- [ ] Draft model choices
- [ ] When speculative decoding helps

## Resources

### Papers

- [Fast Inference from Transformers via Speculative Decoding (2022)](https://arxiv.org/abs/2211.17192) - Original paper
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) - DeepMind's version
- [Medusa: Simple LLM Inference Acceleration with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
- [SpecInfer: Accelerating Generative LLM Serving](https://arxiv.org/abs/2305.09781) - Tree-based speculation

### Blogs & Tutorials

- [HuggingFace: Assisted Generation](https://huggingface.co/blog/assisted-generation)
- [Jay Alammar: Speculative Decoding Visual Guide](https://jalammar.github.io/illustrated-speculative-decoding/)

### Code References

- [HuggingFace Assisted Generation](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py)
- [vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/models/spec_decode.html)

---

## Notes

### The Core Problem

Autoregressive decoding is slow because:
1. Each token requires a full forward pass
2. Forward passes are memory-bound (low GPU utilization)
3. Can only generate ONE token per pass

```
Standard decoding (memory-bound, ~30% GPU utilization):
┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
│ Fwd │───│ Fwd │───│ Fwd │───│ Fwd │──▶ 4 tokens, 4 passes
└─────┘   └─────┘   └─────┘   └─────┘
  "The"    "cat"     "sat"     "on"
```

### Speculative Decoding Insight

What if we could verify multiple tokens in ONE forward pass?

```
Speculative decoding:

Draft model (fast):           Target model (verify):
┌─────────────────────┐      ┌────────────────────────────┐
│ Generate 4 tokens   │──────│ Verify all 4 in ONE pass   │
│ "The cat sat on"    │      │ Accept: "The cat sat" ✓    │
│ (4 cheap passes)    │      │ Reject: "on" → sample "by" │
└─────────────────────┘      └────────────────────────────┘

Result: 3 tokens verified + 1 new = 4 tokens from 1 target pass!
```

### Why It's Lossless

The magic is in the **rejection sampling** algorithm:
- If draft token matches what target would sample → accept
- If not → reject and sample from adjusted distribution
- Mathematically identical to sampling from target model alone!

### Target Model Verification: How It Works

**Key Question**: What does the target model output during verification?

**Answer**: The exact same thing as normal inference - logits over vocabulary at each position. **No special classifier or additional training needed.**

```
Normal autoregressive inference (1 token at a time):
┌─────────────────────────────────────────────────────────────┐
│ Input: [prompt]                                             │
│ Output: logits at position len(prompt)                      │
│         → sample token_0                                    │
├─────────────────────────────────────────────────────────────┤
│ Input: [prompt, token_0]                                    │
│ Output: logits at position len(prompt)+1                    │
│         → sample token_1                                    │
├─────────────────────────────────────────────────────────────┤
│ Input: [prompt, token_0, token_1]                           │
│ Output: logits at position len(prompt)+2                    │
│         → sample token_2                                    │
└─────────────────────────────────────────────────────────────┘
Total: 3 forward passes for 3 tokens
```

**Speculative verification (all tokens at once):**
```
┌─────────────────────────────────────────────────────────────┐
│ Input: [prompt, draft_0, draft_1, draft_2]  ← All at once!  │
│                                                             │
│ Output: logits at EVERY position (due to causal attention)  │
│         logits[len(prompt)]   → verify draft_0              │
│         logits[len(prompt)+1] → verify draft_1              │
│         logits[len(prompt)+2] → verify draft_2              │
│         logits[len(prompt)+3] → sample NEW token if all pass│
└─────────────────────────────────────────────────────────────┘
Total: 1 forward pass, get logits for ALL positions!
```

**Why does this work?** Causal (autoregressive) attention means:
- Position i only attends to positions [0, 1, ..., i-1]
- So logits[i] is computed as if tokens [i+1, i+2, ...] don't exist
- Each position's output is independent of future tokens

```python
# Target model forward pass - standard transformer
def target_forward(input_ids):
    """
    input_ids: [batch, seq_len] - prompt + all draft tokens
    returns: logits [batch, seq_len, vocab_size]

    Due to causal mask, logits[t] only depends on input_ids[:t+1]
    """
    hidden = self.embed(input_ids)

    for layer in self.layers:
        # Causal attention: position t only sees [0..t]
        hidden = layer(hidden, causal_mask=True)

    logits = self.lm_head(hidden)  # [batch, seq_len, vocab_size]
    return logits
```

### The Comparison Process: Step by Step

Given draft tokens `[d0, d1, d2, d3]` and their probabilities from draft model:

```python
# Step 1: Run target model on full sequence
input_seq = prompt + [d0, d1, d2, d3]
target_logits = target_model(input_seq)  # [seq_len, vocab_size]

# Step 2: Extract probabilities at each draft position
# Position after prompt = where we verify d0
# Position after d0 = where we verify d1, etc.

p_target = []
for i in range(4):  # 4 draft tokens
    pos = len(prompt) + i
    probs = softmax(target_logits[pos])  # Distribution over vocab
    p_target.append(probs)

# Step 3: Compare with draft probabilities
for i, (draft_token, p_draft, p_tgt) in enumerate(zip(drafts, draft_probs, p_target)):
    # p_draft[draft_token] = probability draft model assigned to this token
    # p_tgt[draft_token] = probability target model assigns to this token

    acceptance_prob = min(1, p_tgt[draft_token] / p_draft[draft_token])

    if random.random() < acceptance_prob:
        accept(draft_token)
    else:
        # Rejection: sample from corrected distribution
        corrected = max(0, p_tgt - p_draft)
        corrected = corrected / corrected.sum()
        new_token = sample(corrected)
        accept(new_token)
        break  # Stop here, don't verify remaining drafts
```

### Why Rejection Sampling Preserves Distribution

The acceptance criterion `min(1, p_target/p_draft)` is carefully designed:

```
Case 1: p_target[token] >= p_draft[token]
        → acceptance_prob = 1
        → Always accept (target likes it at least as much)

Case 2: p_target[token] < p_draft[token]
        → acceptance_prob = p_target/p_draft < 1
        → Sometimes reject (draft was overconfident)

On rejection, we sample from:
        corrected = (p_target - p_draft).clamp(min=0).normalize()

This ensures the FINAL distribution equals p_target exactly!
```

**Mathematical proof sketch:**
```
P(output = x) = P(accept x from draft) + P(reject, then sample x)
              = p_draft[x] × min(1, p_target[x]/p_draft[x])
                + (1 - Σ_y p_draft[y] × min(1, p_target[y]/p_draft[y])) × corrected[x]
              = p_target[x]  ✓
```

### No Training Required for Basic Speculative Decoding

| Component | Training Needed? | Notes |
|-----------|------------------|-------|
| Target model | No | Use as-is, standard forward pass |
| Draft model (separate) | No | Just needs compatible tokenizer |
| Medusa heads | **Yes** | Train extra heads on target's hidden states |
| EAGLE | **Yes** | Train autoregressive head on features |

For basic speculative decoding with a separate draft model:
- Both models must share the same tokenizer (or be compatible)
- Draft model should be trained on similar data for high acceptance rate
- No architectural changes to either model

### Algorithm

```python
def speculative_decode(target_model, draft_model, prompt, K=4):
    """
    K: number of tokens to speculate
    """
    tokens = prompt

    while not done:
        # 1. Draft: generate K tokens with small model
        draft_tokens = []
        draft_probs = []
        for _ in range(K):
            p_draft = draft_model(tokens + draft_tokens)
            token = sample(p_draft)
            draft_tokens.append(token)
            draft_probs.append(p_draft[token])

        # 2. Verify: run target model on all K+1 positions in parallel
        # Input: [prompt, draft_0, draft_1, ..., draft_{K-1}]
        target_probs = target_model(tokens + draft_tokens)  # One forward pass!

        # 3. Accept/Reject each draft token
        accepted = []
        for i, (draft_tok, p_d, p_t) in enumerate(
            zip(draft_tokens, draft_probs, target_probs)
        ):
            # Acceptance probability
            r = random.random()
            if r < min(1, p_t[draft_tok] / p_d[draft_tok]):
                accepted.append(draft_tok)
            else:
                # Reject: sample from adjusted distribution
                adjusted = max(0, p_t - p_d)
                adjusted /= adjusted.sum()
                new_token = sample(adjusted)
                accepted.append(new_token)
                break  # Stop at first rejection

        # 4. If all K accepted, sample one more from target
        if len(accepted) == K:
            accepted.append(sample(target_probs[K]))

        tokens.extend(accepted)

    return tokens
```

### Speedup Analysis

Let:
- `α` = acceptance rate (probability draft matches target)
- `c` = cost ratio (draft_time / target_time), typically 0.05-0.1
- `K` = speculation length

Expected tokens per target forward pass:
```
E[tokens] = Σ(i=1 to K) α^(i-1) + α^K = (1 - α^(K+1)) / (1 - α)
```

Speedup factor:
```
Speedup ≈ E[tokens] / (1 + K × c)
```

Example: α=0.8, K=4, c=0.1
```
E[tokens] = (1 - 0.8^5) / (1 - 0.8) = 3.36 tokens/pass
Speedup = 3.36 / (1 + 4×0.1) = 2.4x
```

### Draft Model Choices

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| Smaller model | Same family, fewer params | Easy setup | Need separate model |
| Quantized | INT4/INT8 of target | No extra model | Limited speedup |
| Early exit | Exit after N layers | Same weights | Architecture change |
| Medusa heads | Extra prediction heads | Single model | Training required |
| N-gram/retrieval | Statistical prediction | Zero compute | Limited accuracy |

### Medusa: Multi-Head Speculation

Instead of separate draft model, add extra "heads" to predict future tokens:

```
                    ┌──────────────┐
Hidden state ──────▶│ Original head │──▶ token[t]
    │               └──────────────┘
    ├──────────────▶│ Medusa head 1 │──▶ token[t+1] (predicted)
    │               └──────────────┘
    ├──────────────▶│ Medusa head 2 │──▶ token[t+2] (predicted)
    │               └──────────────┘
    └──────────────▶│ Medusa head 3 │──▶ token[t+3] (predicted)
                    └──────────────┘
```

Advantages:
- Single model (no separate draft model)
- Heads are lightweight (few parameters)
- Can be trained efficiently

### Tree-Based Speculation (SpecInfer)

Generate a tree of candidates instead of single sequence:

```
            "The"
           /     \
        "cat"   "dog"
        /   \      \
     "sat" "ran"  "barked"
```

Verify entire tree in one pass using tree attention mask.

### When Speculative Decoding Helps

**Good scenarios:**
- Large target model (high forward pass cost)
- High acceptance rate (draft matches target)
- Low latency requirements
- Single request (not batch-limited)

**Poor scenarios:**
- Small target model (draft overhead dominates)
- Low acceptance rate (many rejections)
- High batch sizes (already efficient)
- Very different draft/target distributions

### Acceptance Rate Factors

High acceptance when:
- Draft model is similar to target (same training data)
- Text is predictable (code, structured text)
- Greedy/low temperature sampling

Low acceptance when:
- Draft model quality is poor
- Text is creative/unpredictable
- High temperature sampling

### Practical Considerations

**Memory overhead:**
```
Without speculation: target_model memory
With speculation: target_model + draft_model memory

For Medusa: target_model + small_heads memory (~1-5% extra)
```

**Latency vs Throughput tradeoff:**
- Speculative decoding optimizes **latency** for single requests
- May hurt **throughput** when batch size is limited by draft overhead
- Best for latency-sensitive applications

### Code Example: Simple Implementation

```python
class SpeculativeDecoder:
    def __init__(self, target_model, draft_model, K=4):
        self.target = target_model
        self.draft = draft_model
        self.K = K

    def generate(self, prompt_ids, max_tokens):
        generated = []

        while len(generated) < max_tokens:
            # Draft phase
            draft_ids, draft_probs = self._draft(
                prompt_ids + generated, self.K
            )

            # Verify phase (single target forward pass)
            target_probs = self._verify(
                prompt_ids + generated + draft_ids
            )

            # Accept/reject
            accepted = self._accept_reject(
                draft_ids, draft_probs, target_probs
            )

            generated.extend(accepted)

        return generated

    def _draft(self, context, k):
        """Generate k tokens with draft model."""
        ids, probs = [], []
        for _ in range(k):
            logits = self.draft(context + ids)
            p = F.softmax(logits[-1], dim=-1)
            token = torch.multinomial(p, 1).item()
            ids.append(token)
            probs.append(p[token].item())
        return ids, probs

    def _verify(self, full_sequence):
        """Run target model on full sequence."""
        logits = self.target(full_sequence)
        return F.softmax(logits[-self.K-1:], dim=-1)

    def _accept_reject(self, draft_ids, draft_probs, target_probs):
        """Acceptance sampling."""
        accepted = []
        for i, (tok, p_d) in enumerate(zip(draft_ids, draft_probs)):
            p_t = target_probs[i, tok].item()
            if random.random() < min(1, p_t / p_d):
                accepted.append(tok)
            else:
                # Sample from adjusted distribution
                adjusted = torch.clamp(target_probs[i] - draft_probs[i], min=0)
                adjusted /= adjusted.sum()
                accepted.append(torch.multinomial(adjusted, 1).item())
                break
        return accepted
```
