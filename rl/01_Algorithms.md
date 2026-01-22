# RL Alignment Algorithms

> Parent: [RL & Alignment Overview](rl.md)

## Overview

This document covers the evolution of RL-based alignment algorithms from PPO to the latest methods like GDPO. All algorithms are presented together for easy comparison.

## Algorithm Timeline

```
2017        2022           2023         2024           2025
 │           │              │            │              │
 ▼           ▼              ▼            ▼              ▼
PPO ──▶ InstructGPT ──▶   DPO    ──▶  GRPO   ──▶    GDPO
        (RLHF)         (reward-free)  (DeepSeek)  (distributional)
         │                 │            │              │
         │                 ▼            │              │
         │            IPO, KTO,        │              │
         │            ORPO, SimPO      │              │
         │                             │              │
         └─────────────────────────────┴──────────────┘
                     │
              Process Reward Models
              DeepSeek-R1 (pure RL)
```

## Resources

### Papers (Chronological)

| Year | Paper | Method |
|------|-------|--------|
| 2017 | [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) | PPO |
| 2022 | [Training LMs to follow instructions (InstructGPT)](https://arxiv.org/abs/2203.02155) | RLHF |
| 2023 | [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) | DPO |
| 2023 | [A General Theoretical Paradigm (IPO)](https://arxiv.org/abs/2310.12036) | IPO |
| 2024 | [KTO: Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) | KTO |
| 2024 | [ORPO: Without Reference Model](https://arxiv.org/abs/2403.07691) | ORPO |
| 2024 | [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300) | GRPO |
| 2024 | [Back to Basics: REINFORCE Style](https://arxiv.org/abs/2402.14740) | RLOO |
| 2025 | [DeepSeek-R1: Incentivizing Reasoning](https://arxiv.org/abs/2501.12948) | GRPO+R1 |
| 2025 | [No Preference Left Behind (GDPO)](https://arxiv.org/abs/2412.20299) | GDPO |

### Tutorials

- [HuggingFace RLHF Blog](https://huggingface.co/blog/rlhf)
- [HuggingFace DPO Tutorial](https://huggingface.co/blog/dpo-trl)
- [How to align LLMs in 2025](https://www.philschmid.de/rl-with-llms-in-2025-dpo)
- [GRPO Explained](https://cameronrwolfe.substack.com/p/grpo)

---

## Algorithm Comparison Summary

| Method | Year | Reward Model | Value Model | Reference Model | Online | Key Innovation |
|--------|------|--------------|-------------|-----------------|--------|----------------|
| **PPO** | 2017 | Yes | Yes | Yes | Yes | Clipped objective |
| **DPO** | 2023 | No | No | Yes | No | Direct preference |
| **IPO** | 2023 | No | No | Yes | No | Robust to noise |
| **KTO** | 2024 | No | No | Yes | No | Unpaired data |
| **ORPO** | 2024 | No | No | No | No | No reference |
| **GRPO** | 2024 | Yes* | No | Yes | Yes | Group normalization |
| **RLOO** | 2024 | Yes* | No | Yes | Yes | Leave-one-out baseline |
| **GDPO** | 2025 | No | No | Yes | No | Distributional preferences |

*GRPO/RLOO can use learned RM or rule-based reward (e.g., correctness)

### Memory Footprint

```
PPO:     4 models in memory (Policy + Reference + Reward + Value)
DPO:     2 models in memory (Policy + Reference)
ORPO:    1 model in memory  (Policy only)
GRPO:    2-3 models        (Policy + Reference + optional RM)
```

---

## Industry Usage: Who Uses What?

| Company | Model | Primary Method | Notes |
|---------|-------|----------------|-------|
| **OpenAI** | GPT-4, ChatGPT | PPO (RLHF) | Pioneered RLHF with InstructGPT |
| **Anthropic** | Claude 3/3.5 | Constitutional AI (RLAIF) | PPO + AI feedback instead of human |
| **DeepSeek** | DeepSeek-R1, Math | GRPO | No value model, rule-based rewards |
| **Qwen** | Qwen 2.5, Qwen 3 | GRPO / DPO | MS-SWIFT supports both |
| **Meta** | LLaMA 3 | DPO + PPO | Hybrid approach |
| **Google** | Gemini | PPO (RLHF) | Similar to OpenAI approach |

### OpenAI Approach

```
InstructGPT / GPT-4 Pipeline:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  SFT on     │───▶│Train Reward │───▶│  PPO with   │
│  demos      │    │   Model     │    │ 4 models    │
└─────────────┘    └─────────────┘    └─────────────┘
                         │
                  Human labelers
                  rank responses
```

### Anthropic (Claude) Approach

```
Constitutional AI:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  SFT on     │───▶│ AI Critique │───▶│ RLAIF with  │
│  demos      │    │ + Revision  │    │    PPO      │
└─────────────┘    └─────────────┘    └─────────────┘
                         │
                   AI judges responses
                   using "constitution"
                   (no human labelers!)
```

Key innovation: Use AI (Claude itself) to judge which responses are better, guided by a set of principles (the "constitution"). This is **RLAIF** (RL from AI Feedback) instead of RLHF.

### DeepSeek Approach

```
DeepSeek-R1 / DeepSeek-Math:
┌─────────────┐    ┌─────────────────────────────────┐
│ Pre-train   │───▶│  GRPO with rule-based rewards   │
│ (no SFT!)   │    │  (correctness verification)     │
└─────────────┘    └─────────────────────────────────┘
                              │
                   No reward model needed!
                   Just check if answer is correct
```

Key innovation: Skip SFT entirely, use GRPO with verifiable rewards (math/code correctness). Chain-of-thought reasoning **emerges** from RL.

### Qwen Approach

```
Qwen uses MS-SWIFT framework supporting multiple methods:
- DPO for general alignment
- GRPO for reasoning tasks (math, code)
- Supports both offline and online training
```

---

## Reward Models: How Human Feedback Becomes Reward

### Not ELO - It's Bradley-Terry!

A common misconception is that reward models use ELO ranking (like chess). They actually use the **Bradley-Terry model** for pairwise comparisons:

```
ELO (Chess):
- Designed for sequential games with clear winners
- Updates ratings after each game
- Not suitable for one-shot comparisons

Bradley-Terry (Reward Models):
- Designed for pairwise preference data
- P(A beats B) = σ(score_A - score_B)
- Directly maps to reward model training
```

### Bradley-Terry Model

The probability that response A is preferred over response B:

```
P(A > B | x) = σ(r(x, A) - r(x, B)) = 1 / (1 + exp(-(r_A - r_B)))

where:
- σ is the sigmoid function
- r(x, y) is the reward model score for response y given prompt x
```

### Reward Model Training

```python
def reward_model_loss(model, prompt, chosen, rejected):
    """
    Train reward model on human preferences

    Data format:
    - prompt: "What is the capital of France?"
    - chosen: "The capital of France is Paris." (preferred by human)
    - rejected: "France is a European country." (not preferred)
    """
    # Get reward scores
    r_chosen = model(prompt, chosen)      # scalar score
    r_rejected = model(prompt, rejected)  # scalar score

    # Bradley-Terry loss (negative log likelihood)
    # We want P(chosen > rejected) to be high
    loss = -torch.log(torch.sigmoid(r_chosen - r_rejected))

    return loss.mean()
```

### Reward Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Reward Model                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: [prompt] + [response]                               │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────────────────────────────┐                │
│  │      LLM Backbone (e.g., LLaMA)         │                │
│  │      (can be frozen or fine-tuned)      │                │
│  └─────────────────────────────────────────┘                │
│           │                                                  │
│           ▼                                                  │
│  Last hidden state of [EOS] token                           │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────────────────────────────┐                │
│  │   Linear layer: hidden_dim → 1          │                │
│  └─────────────────────────────────────────┘                │
│           │                                                  │
│           ▼                                                  │
│      Scalar reward (e.g., 0.73)                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Human Feedback Collection Process

```
Step 1: Generate multiple responses
┌─────────────────────────────────────────────────────────────┐
│ Prompt: "Explain quantum computing"                          │
│                                                              │
│ Response A: "Quantum computing uses qubits..."              │
│ Response B: "It's a type of fast computer..."               │
│ Response C: "Quantum computers leverage superposition..."   │
│ Response D: "Computing with quantum mechanics..."           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 2: Human annotators rank responses
┌─────────────────────────────────────────────────────────────┐
│ Annotator ranking: C > A > D > B                            │
│                                                              │
│ This generates pairwise preferences:                        │
│   C > A, C > D, C > B                                       │
│   A > D, A > B                                              │
│   D > B                                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 3: Train reward model on pairs
┌─────────────────────────────────────────────────────────────┐
│ For each pair (chosen, rejected):                           │
│   loss = -log(σ(r_chosen - r_rejected))                    │
│                                                              │
│ After training:                                             │
│   r(C) > r(A) > r(D) > r(B)                                │
└─────────────────────────────────────────────────────────────┘
```

### Reward Model Quality Matters!

The reward model is the bottleneck of RLHF:

| RM Quality | Result |
|------------|--------|
| Good RM | Policy learns desired behavior |
| Noisy RM | Policy learns inconsistent behavior |
| Biased RM | Policy learns biased behavior |
| Hackable RM | Policy exploits RM weaknesses |

**Best practices:**
- Use diverse annotators (avoid single-annotator bias)
- Include attention checks in labeling
- Train RM on diverse prompts
- Use RM ensembles to reduce variance
- Validate RM on held-out preferences

---

## 1. PPO (Proximal Policy Optimization)

> Used in: InstructGPT, ChatGPT, GPT-4, Claude (with RLAIF)

### The Three Stages of RLHF

```
Stage 1: SFT                Stage 2: Reward Model       Stage 3: PPO
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│ Supervised       │       │ Train reward     │       │ Optimize policy  │
│ fine-tuning on   │ ───▶  │ model on human   │ ───▶  │ using PPO with   │
│ demonstrations   │       │ preferences      │       │ reward signal    │
└──────────────────┘       └──────────────────┘       └──────────────────┘
```

### PPO Objective

```
max_θ E_{x~D, y~π_θ}[r(x, y)] - β · KL(π_θ || π_ref)
```

The KL penalty prevents reward hacking and catastrophic forgetting.

### Reward Model Training

Bradley-Terry model for preference:

```
P(y₁ > y₂ | x) = σ(r(x, y₁) - r(x, y₂))
```

```python
def reward_model_loss(r_chosen, r_rejected):
    return -torch.log(torch.sigmoid(r_chosen - r_rejected)).mean()
```

### PPO Algorithm

```python
def ppo_step(policy, ref_policy, reward_model, value_model, prompts):
    # 1. Generate responses
    responses = policy.generate(prompts)

    # 2. Compute rewards
    rewards = reward_model(prompts, responses)

    # 3. KL penalty
    log_probs = policy.log_prob(responses | prompts)
    ref_log_probs = ref_policy.log_prob(responses | prompts)
    kl_penalty = log_probs - ref_log_probs
    modified_rewards = rewards - beta * kl_penalty

    # 4. Compute advantages (GAE)
    values = value_model(prompts, responses)
    advantages = compute_gae(modified_rewards, values)

    # 5. PPO clipped update
    for _ in range(ppo_epochs):
        new_log_probs = policy.log_prob(responses | prompts)
        ratio = torch.exp(new_log_probs - log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-ε, 1+ε) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        new_values = value_model(prompts, responses)
        value_loss = (new_values - returns).pow(2).mean()

        loss = policy_loss + value_coef * value_loss
        loss.backward()
        optimizer.step()
```

### PPO Pros & Cons

| Pros | Cons |
|------|------|
| Powerful, proven at scale | 4 models in memory |
| Online learning | Complex to implement |
| Works with any reward signal | Hyperparameter sensitive |
| Continuous improvement | Training instability |

---

## 2. DPO (Direct Preference Optimization)

> The breakthrough: eliminate reward model and RL loop

### Key Insight

The optimal RL policy has a closed-form solution:

```
π*(y|x) = π_ref(y|x) · exp(r(x,y) / β) / Z(x)
```

Rearranging, reward can be expressed as:

```
r(x, y) = β · log(π*(y|x) / π_ref(y|x)) + β · log Z(x)
```

Since Z(x) cancels in preference comparisons → train directly on preferences!

### DPO Loss

```python
def dpo_loss(policy_chosen_logp, policy_rejected_logp,
             ref_chosen_logp, ref_rejected_logp, beta=0.1):
    """
    The elegant DPO loss function
    """
    # Log ratios (implicit rewards)
    chosen_logratios = policy_chosen_logp - ref_chosen_logp
    rejected_logratios = policy_rejected_logp - ref_rejected_logp

    # DPO loss
    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -F.logsigmoid(logits).mean()

    return loss
```

### DPO vs PPO

```
PPO Pipeline:
┌─────────────────────────────────────────────────────────────┐
│ Preferences → Train RM → Generate → Score → PPO Update     │
│               (expensive)  (slow)   (RM call) (complex)     │
└─────────────────────────────────────────────────────────────┘

DPO Pipeline:
┌─────────────────────────────────────────────────────────────┐
│ Preferences → Direct Supervised Loss → Update               │
│                     (simple, stable)                         │
└─────────────────────────────────────────────────────────────┘
```

### The β Parameter

```
β = 0.01 (small): Strong updates, may overfit
β = 0.1 (typical): Good balance
β = 1.0 (large): Conservative, may underfit
```

### DPO Pros & Cons

| Pros | Cons |
|------|------|
| Simple supervised training | Offline only (fixed dataset) |
| Only 2 models in memory | Can't improve beyond data |
| Stable training | Sensitive to data quality |
| Few hyperparameters | Length bias issues |

---

## 3. DPO Variants

### IPO (Identity Preference Optimization)

More robust to label noise:

```python
def ipo_loss(chosen_logratios, rejected_logratios, beta=0.1):
    """Squared hinge loss instead of log-sigmoid"""
    logits = chosen_logratios - rejected_logratios
    loss = ((logits - 1 / (2 * beta)) ** 2).mean()
    return loss
```

### KTO (Kahneman-Tversky Optimization)

Works with **unpaired** data (just good/bad labels, no comparisons):

```python
def kto_loss(policy_logps, ref_logps, is_chosen, beta=0.1):
    """
    Doesn't require paired preferences!
    Just: "this response is good" or "this response is bad"
    """
    logratios = policy_logps - ref_logps

    # Asymmetric loss (prospect theory)
    chosen_loss = 1 - F.sigmoid(beta * logratios[is_chosen])
    rejected_loss = 1 - F.sigmoid(-beta * logratios[~is_chosen])

    # Loss aversion: humans weight losses more than gains
    lambda_chosen = 1.0
    lambda_rejected = 1.33

    loss = lambda_chosen * chosen_loss.mean() + lambda_rejected * rejected_loss.mean()
    return loss
```

### ORPO (Odds Ratio Preference Optimization)

**No reference model needed** - saves memory:

```python
def orpo_loss(policy_logps_chosen, policy_logps_rejected,
              policy_avg_logp_chosen):
    """Combines SFT + preference in one loss"""
    # SFT loss on chosen responses
    sft_loss = -policy_avg_logp_chosen

    # Odds ratio preference
    log_odds = policy_logps_chosen - policy_logps_rejected
    pref_loss = -F.logsigmoid(log_odds).mean()

    return sft_loss + lambda_pref * pref_loss
```

### SimPO (Simple Preference Optimization)

Adds margin and length normalization:

```python
def simpo_loss(policy_chosen_logp, policy_rejected_logp,
               chosen_len, rejected_len, beta=2.0, gamma=0.5):
    """
    Length-normalized with margin
    """
    # Length normalize
    chosen_reward = policy_chosen_logp / chosen_len
    rejected_reward = policy_rejected_logp / rejected_len

    # Add margin
    logits = beta * (chosen_reward - rejected_reward) - gamma
    loss = -F.logsigmoid(logits).mean()
    return loss
```

### DPO Variants Comparison

| Method | Paired Data | Reference Model | Key Feature |
|--------|-------------|-----------------|-------------|
| DPO | Yes | Yes | Simple, effective |
| IPO | Yes | Yes | Robust to noise |
| KTO | **No** | Yes | Unpaired data |
| ORPO | Yes | **No** | Memory efficient |
| SimPO | Yes | **No** | Length normalized |

---

## 4. GRPO (Group Relative Policy Optimization)

> Used in: DeepSeek-Math, DeepSeek-R1

### Key Innovation

Replace value network with **group-based advantage normalization**:

```
Traditional PPO:  A(s,a) = R(s,a) - V(s)  ← Need to learn V
                                    ↑
GRPO:            A(x,y) = (r(x,y) - mean(r)) / std(r)  ← Compute from group
```

### GRPO Algorithm

```python
def grpo_step(policy, ref_policy, reward_fn, prompts, G=4):
    """
    G: number of responses per prompt (group size)
    """
    all_losses = []

    for prompt in prompts:
        # Generate G responses for this prompt
        responses = [policy.generate(prompt) for _ in range(G)]

        # Get rewards
        rewards = torch.tensor([reward_fn(prompt, r) for r in responses])

        # GROUP NORMALIZATION (the key insight!)
        # This replaces the value function entirely
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Policy gradient with advantages
        for response, advantage in zip(responses, advantages):
            log_prob = policy.log_prob(response | prompt)
            ref_log_prob = ref_policy.log_prob(response | prompt)

            # KL penalty
            kl = log_prob - ref_log_prob

            # Loss
            loss = -advantage * log_prob + beta * kl
            all_losses.append(loss)

    return torch.stack(all_losses).mean()
```

### Why GRPO Works

```
For prompt "What is 2+2?":

Response 1: "4"           → reward = 1.0
Response 2: "The answer is 4" → reward = 1.0
Response 3: "2+2=5"       → reward = 0.0
Response 4: "Four"        → reward = 1.0

Group stats: mean=0.75, std=0.43

Normalized advantages:
Response 1: (1.0-0.75)/0.43 = +0.58  ← reinforce
Response 2: (1.0-0.75)/0.43 = +0.58  ← reinforce
Response 3: (0.0-0.75)/0.43 = -1.74  ← suppress
Response 4: (1.0-0.75)/0.43 = +0.58  ← reinforce
```

### GRPO vs PPO

| Aspect | PPO | GRPO |
|--------|-----|------|
| Value network | Required | **Not needed** |
| Models in memory | 4 | 2-3 |
| Advantage estimation | Learned V(s) | Group statistics |
| Variance reduction | GAE | Group normalization |
| Implementation | Complex | Simpler |

---

## 5. RLOO (REINFORCE Leave-One-Out)

Similar to GRPO but uses leave-one-out baseline:

```python
def rloo_advantage(rewards):
    """
    For each sample, baseline is mean of OTHER samples
    """
    n = len(rewards)
    advantages = []

    for i in range(n):
        # Leave-one-out mean
        baseline = (rewards.sum() - rewards[i]) / (n - 1)
        advantages.append(rewards[i] - baseline)

    return advantages
```

### RLOO vs GRPO

| Aspect | GRPO | RLOO |
|--------|------|------|
| Baseline | Group mean | Leave-one-out mean |
| Normalization | By std | None (or optional) |
| Bias | Slightly biased | Unbiased |
| Variance | Lower | Higher |
| Practice | Often more stable | Theoretically cleaner |

---

## 6. GDPO (Group Distributional Preference Optimization)

> ICLR 2025 - Handles **diverse preferences** within a group

### The Problem GDPO Solves

Standard DPO assumes everyone agrees on preferences. But in reality:

```
Prompt: "Is social media good or bad?"

Annotator 1: Prefers response supporting social media
Annotator 2: Prefers response criticizing social media
Annotator 3: Prefers balanced response

DPO: Averages these → loses diversity, favors majority
GDPO: Models the DISTRIBUTION of preferences
```

### GDPO Key Insight

Incorporate **beliefs** that shape preferences:

```
Standard DPO:  P(y₁ > y₂ | x)

GDPO:          P(y₁ > y₂ | x, belief)

               Where belief ∈ {belief₁, belief₂, ...}
               represents different viewpoints
```

### GDPO Framework

```python
def gdpo_loss(policy, ref_policy, prompt, responses,
              belief_distribution, beta=0.1):
    """
    Align with DISTRIBUTION of preferences, not single preference

    belief_distribution: estimated distribution over belief types
                        e.g., {conservative: 0.3, liberal: 0.4, moderate: 0.3}
    """
    total_loss = 0

    for belief, prob in belief_distribution.items():
        # Get preference under this belief
        chosen, rejected = get_preference_for_belief(responses, belief)

        # Standard DPO-style loss for this belief
        chosen_logratio = policy.log_prob(chosen) - ref_policy.log_prob(chosen)
        rejected_logratio = policy.log_prob(rejected) - ref_policy.log_prob(rejected)

        belief_loss = -F.logsigmoid(beta * (chosen_logratio - rejected_logratio))

        # Weight by belief probability
        total_loss += prob * belief_loss

    return total_loss
```

### GDPO vs DPO

```
DPO Training:
┌─────────────────────────────────────────────────┐
│ All annotators → Single preference → Train      │
│                      ↓                          │
│              Majority wins, minority lost       │
└─────────────────────────────────────────────────┘

GDPO Training:
┌─────────────────────────────────────────────────┐
│ Annotators → Belief estimation → Per-belief     │
│     │              │            preferences     │
│     │              ▼                  │         │
│     │     Distribution over      ◀───┘         │
│     │        belief types                       │
│     │              │                            │
│     └──────────────┼──────────────────────────▶│
│                    ▼                            │
│         Train to match distribution             │
│         (no preference left behind!)            │
└─────────────────────────────────────────────────┘
```

### When to Use GDPO

| Scenario | Use GDPO? |
|----------|-----------|
| Controversial topics | Yes |
| Subjective preferences (style, tone) | Yes |
| Factual correctness | No (use DPO) |
| Multi-cultural alignment | Yes |
| Single clear preference | No (use DPO) |

---

## 7. Process vs Outcome Rewards

### Outcome Reward Model (ORM)

```
Only score final answer
Reward = 1 if correct, 0 if wrong

Pros: Simple
Cons: Sparse signal, credit assignment problem
```

### Process Reward Model (PRM)

```
Score each reasoning step

Math: "What is 15 × 7?"
  Step 1: "15 × 7 = 15 × (5+2)" ✓ (+0.8)
  Step 2: "= 75 + 30"          ✓ (+0.9)
  Step 3: "= 105"              ✓ (+1.0)

Pros: Dense signal, better credit assignment
Cons: Needs step-level labels, expensive
```

### DeepSeek-R1: Pure RL for Reasoning

```
Traditional: Pre-train → SFT → RLHF
                          ↑
                   Need demonstrations

DeepSeek-R1: Pre-train → GRPO (pure RL)
                          ↑
               No SFT needed!
               Chain-of-thought emerges naturally
```

Key findings:
- RL alone can teach reasoning
- Long CoT develops to solve hard problems
- Self-reflection emerges spontaneously

---

## 8. Practical Guide: Choosing an Algorithm

### Decision Tree

```
Do you have preference pairs (chosen vs rejected)?
├── No → KTO (works with unpaired good/bad labels)
│
└── Yes → Is memory a concern?
          ├── Yes → ORPO or SimPO (no reference model)
          │
          └── No → Do you need online learning?
                   ├── No → DPO (simple, effective)
                   │
                   └── Yes → Do you have a reward model?
                            ├── Yes → Is it rule-based (code/math correctness)?
                            │         ├── Yes → GRPO
                            │         └── No → PPO (learned RM)
                            │
                            └── No → Online DPO / Iterative DPO
```

### Recommendations by Use Case

| Use Case | Recommended | Why |
|----------|-------------|-----|
| First alignment attempt | DPO | Simple, stable |
| Limited GPU memory | ORPO | No reference model |
| Thumbs up/down data | KTO | Works with unpaired |
| Math/Code tasks | GRPO | Rule-based rewards |
| Continuous improvement | Online DPO or GRPO | Iterative |
| Diverse user preferences | GDPO | Distributional |
| Maximum performance | PPO | Most flexible |

### Hyperparameter Defaults

| Algorithm | Key Param | Typical Value |
|-----------|-----------|---------------|
| PPO | KL coef (β) | 0.01 - 0.2 |
| PPO | Clip ratio (ε) | 0.1 - 0.2 |
| DPO | β | 0.1 |
| GRPO | Group size (G) | 4 - 8 |
| GRPO | β | 0.01 - 0.1 |
| KTO | λ_rejected | 1.33 |

---

## 9. Common Issues & Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Reward hacking | Exploits RM weaknesses | Increase KL penalty, RM ensemble |
| Length bias | Prefers longer/shorter | Length normalization (SimPO) |
| Training collapse | Gibberish output | Lower LR, higher β |
| KL explosion | Policy diverges | Reduce LR, increase β |
| Weak alignment | Unchanged behavior | Decrease β, more epochs |
| Forgetting | Loses capabilities | Increase β, early stopping |
| Majority bias | Ignores minority prefs | Use GDPO |

---

## 10. Code Examples

### DPO with TRL

```python
from trl import DPOTrainer, DPOConfig

config = DPOConfig(
    beta=0.1,
    learning_rate=5e-7,
    per_device_train_batch_size=4,
    num_train_epochs=1,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=config,
    train_dataset=preference_data,
    tokenizer=tokenizer,
)
trainer.train()
```

### GRPO Implementation

```python
def train_grpo(policy, ref_policy, reward_fn, prompts,
               G=4, beta=0.04, lr=1e-6):
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    for prompt_batch in dataloader:
        optimizer.zero_grad()
        batch_loss = 0

        for prompt in prompt_batch:
            # Generate G responses
            responses = [policy.generate(prompt) for _ in range(G)]
            rewards = torch.tensor([reward_fn(prompt, r) for r in responses])

            # Group normalize
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # Compute loss
            for resp, adv in zip(responses, advantages):
                log_p = policy.log_prob(resp | prompt)
                ref_log_p = ref_policy.log_prob(resp | prompt)
                kl = log_p - ref_log_p

                batch_loss += -adv * log_p + beta * kl

        batch_loss.backward()
        optimizer.step()
```

### PPO with TRL

```python
from trl import PPOTrainer, PPOConfig

config = PPOConfig(
    learning_rate=1e-5,
    batch_size=64,
    ppo_epochs=4,
    init_kl_coef=0.2,
)

trainer = PPOTrainer(
    config=config,
    model=policy,
    ref_model=ref_model,
    tokenizer=tokenizer,
    reward_model=reward_model,
)

for batch in dataloader:
    responses = trainer.generate(batch["input_ids"])
    rewards = reward_model(batch["input_ids"], responses)
    stats = trainer.step(batch["input_ids"], responses, rewards)
```
