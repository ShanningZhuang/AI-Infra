# RL Alignment Notes

> These notes are integrated into [01_Algorithms.md](01_Algorithms.md)

## Reward Model

### Not ELO - It's Bradley-Terry!

Reward models do **NOT** use ELO ranking. ELO is designed for sequential games (like chess) with clear winners. Instead, reward models use the **Bradley-Terry model** for pairwise preference comparisons:

```
Bradley-Terry Formula:
P(A > B | prompt) = σ(r(A) - r(B)) = 1 / (1 + exp(-(r_A - r_B)))

Where σ is the sigmoid function: σ(x) = 1 / (1 + e^(-x))
- Maps any real number to a probability between 0 and 1
- When r_A = r_B: σ(0) = 0.5 (equal preference)
- When r_A >> r_B: σ → 1 (strongly prefer A)
- When r_A << r_B: σ → 0 (strongly prefer B)
```

### How Human Feedback Becomes Reward

1. **Generate** multiple responses for a prompt
2. **Humans rank** responses (e.g., C > A > D > B)
3. **Convert to pairs**: (C, A), (C, D), (C, B), (A, D), (A, B), (D, B)
4. **Train reward model**: minimize `-log(σ(r_chosen - r_rejected))`

### Reward Model Architecture

```
Input: [prompt + response]
    ↓
LLM Backbone (e.g., LLaMA, frozen or fine-tuned)
    ↓
Last hidden state of [EOS] token
    ↓
Linear(hidden_dim → 1)
    ↓
Scalar reward (e.g., 0.73)
```

---

## DPO

DPO (Direct Preference Optimization) **eliminates the reward model** entirely by showing that:

```
Optimal RLHF policy: π*(y|x) ∝ π_ref(y|x) · exp(r(x,y) / β)

Rearranging: r(x,y) = β · log(π*(y|x) / π_ref(y|x)) + const
```

Since reward differences cancel the constant:

```python
# DPO Loss (no reward model needed!)
logits = β * ((log π_θ(chosen) - log π_ref(chosen))
            - (log π_θ(rejected) - log π_ref(rejected)))
loss = -log_sigmoid(logits)
```

---

## RL Algorithms by Company

| Company | Model | Method | Key Innovation |
|---------|-------|--------|----------------|
| **OpenAI** | GPT-4, ChatGPT | **PPO (RLHF)** | Pioneered RLHF with InstructGPT (2022) |
| **Anthropic** | Claude 3/3.5 | **Constitutional AI (RLAIF)** | AI judges responses using "constitution" principles, no human labelers needed |
| **DeepSeek** | DeepSeek-R1, Math | **GRPO** | No value model, rule-based rewards (correctness), reasoning emerges from RL |
| **Qwen** | Qwen 2.5, Qwen 3 | **GRPO / DPO** | MS-SWIFT supports multiple methods, GSPO for Qwen3 |
| **Meta** | LLaMA 3 | **DPO + PPO** | Hybrid approach |
| **Google** | Gemini | **PPO (RLHF)** | Similar to OpenAI |

### OpenAI: PPO-based RLHF

```
SFT → Train Reward Model (human labels) → PPO optimization
```
- 4 models in memory (policy, reference, reward, value)
- Human annotators rank responses
- Most compute-intensive but proven at scale

### Anthropic: Constitutional AI (RLAIF)

```
SFT → AI Critique + Revision → RLAIF (AI labels) → PPO
```
- AI (Claude) judges which responses are better
- Guided by "constitution" (set of principles)
- Scales without human labelers
- More transparent (can inspect principles)

### DeepSeek: GRPO

```
Pre-train → GRPO with rule-based rewards (skip SFT!)
```
- No value model needed (saves memory)
- Uses group normalization for advantages
- For math/code: reward = correctness (binary)
- Chain-of-thought emerges naturally from RL

### Qwen: Flexible approach

```
MS-SWIFT framework supports:
- DPO for general alignment
- GRPO for reasoning tasks
- Both offline and online training
```
- Qwen3 uses GSPO (Group Sequence Policy Optimization)

---

## Summary: Algorithm Evolution

```
2022: PPO/RLHF (OpenAI)
  ↓   - Powerful but complex (4 models)
  ↓   - Requires reward model + value model

2023: DPO (Stanford)
  ↓   - No reward model (2 models only)
  ↓   - Simple supervised learning
  ↓   - But offline only

2024: GRPO (DeepSeek)
  ↓   - No value model
  ↓   - Works with rule-based rewards
  ↓   - Great for math/code

2025: GDPO (ICLR)
      - Handles diverse preferences
      - Pluralistic alignment
```

---

## Appendix: RLHF Deployment Stages

RLHF happens in **two distinct stages**:

### Pre-deployment RLHF (Post-training)

The initial training phase **before** the model is released:
- Uses a curated dataset of prompts
- Professional annotators rank responses
- Trains the reward model and policy offline
- This is how GPT-4, Claude, etc. were initially aligned

### Online/Continuous RLHF (After Deployment)

Data collection from real users (e.g., ChatGPT showing two answers):

**Why unique prompts still matter:**
- Reveals **preference patterns** (e.g., users prefer concise vs. verbose)
- Captures **edge cases** the original training missed
- Handles **distribution shift** (real queries differ from training data)

**How companies use it:**
- Aggregate preferences across many users
- Even unique prompts share underlying patterns (tone, format, accuracy)
- Periodically retrain or fine-tune the reward model
- Some systems do **online learning** where feedback affects future responses

**The math still works:**
- Don't need the exact same prompt twice
- 1000 users picking "concise" over "verbose" on different prompts = strong signal
- Reward model learns **generalizable preferences**, not prompt-specific ones

| Stage | When | Data Source |
|-------|------|-------------|
| Post-training RLHF | Before release | Paid annotators, curated prompts |
| Online RLHF | After release | Real users, real queries |

---

## Appendix: Reward Model Architecture Deep Dive

### Why Use the Last Hidden State (EOS Token)?

```
Sequence: [prompt tokens] [response tokens] [EOS]
                                              ↑
                              Extract this hidden state
```

**Key insight**: In autoregressive transformers, each token's hidden state contains information about **all previous tokens** (due to causal attention). The EOS token is special:

1. **Information aggregation**: By the time we reach EOS, the model has "seen" the entire prompt + response. The EOS hidden state is a compressed summary of everything.

2. **Causal attention means forward-only flow**:
   ```
   Token:    [The] [cat] [sat] [EOS]
   Attends:   -    [The] [The] [The]
                   [cat] [cat]
                         [sat]
   ```
   EOS attends to ALL previous tokens, making it the most informed position.

3. **Why not use all hidden states?**
   - **Redundancy**: Earlier tokens don't know about later tokens
   - **Variable length**: Different responses have different lengths → need pooling anyway
   - **Compute cost**: Processing all states is expensive
   - **What matters is the final judgment**: We want "how good is this complete response?" not per-token scores

### The Information Bottleneck Problem

Using only the EOS hidden state creates a **compression bottleneck**:

```
500-token response → compressed into → 4096-dim vector → 1 scalar reward
     ~2M bits of info                    ~16KB              1 number
```

**What gets lost:**
- **Fine-grained quality signals**: A response might be 90% excellent but have one bad paragraph
- **Positional nuances**: Was the error at the beginning (more forgivable) or the conclusion (critical)?
- **Multi-aspect evaluation**: Helpfulness, accuracy, safety, style—all collapsed into one score

**Why it still works (mostly):**
1. **Transformers are good compressors**: The model learns to encode task-relevant features
2. **Pairwise comparison is forgiving**: We only need to rank A vs B, not assign absolute scores
3. **Training signal is coarse anyway**: Human preferences are noisy; fine-grained signals might not help

**When it breaks down:**
- Very long responses (>1000 tokens)
- Complex multi-part questions
- Responses with mixed quality (good intro, bad conclusion)

**Mitigations:**
| Approach | How it helps |
|----------|--------------|
| **Larger hidden dim** | More capacity to encode info (but diminishing returns) |
| **Process Reward Models** | Score each step, not just final output |
| **Multi-head rewards** | Separate heads for helpfulness, safety, accuracy |
| **Chunked evaluation** | Split response, score chunks, aggregate |

### Alternative Architectures (Less Common)

| Method | Description | Trade-off |
|--------|-------------|-----------|
| **EOS only** (standard) | Single hidden state → Linear → scalar | Simple, effective |
| **Mean pooling** | Average all hidden states | Loses position info, dilutes signal |
| **Attention pooling** | Learned weighted average | More params, marginal gains |
| **Per-token rewards** | Reward at each step | Useful for process supervision (PRM) |

### Process Reward Models (PRM) vs Outcome Reward Models (ORM)

```
ORM (standard): Score the final answer only
  "2+2=5" → reward = 0.1 (wrong answer)

PRM: Score each reasoning step
  "2+2" → step1_reward = 1.0 (correct setup)
  "=5"  → step2_reward = 0.0 (wrong calculation)
```

PRMs use **all hidden states** because they need to evaluate intermediate steps. This is useful for math/reasoning where the process matters, not just the outcome.

### Code Example

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model  # e.g., LLaMA
        self.reward_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, input_ids, attention_mask):
        # Get all hidden states
        outputs = self.backbone(input_ids, attention_mask, output_hidden_states=True)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        
        # Find EOS position (last non-padding token)
        eos_indices = attention_mask.sum(dim=1) - 1  # [batch]
        
        # Extract EOS hidden state
        batch_indices = torch.arange(hidden_states.size(0))
        eos_hidden = hidden_states[batch_indices, eos_indices]  # [batch, hidden_dim]
        
        # Project to scalar reward
        reward = self.reward_head(eos_hidden).squeeze(-1)  # [batch]
        return reward
```
