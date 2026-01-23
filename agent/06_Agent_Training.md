# Agent Training & RL

> Parent: [Agent Infrastructure](00_Agent.md)

## Overview

This document covers the emerging field of **training AI agents** using reinforcement learning and other methods. While traditional RLHF/RLVR focuses on single-turn responses, Agent RL extends these techniques to **multi-turn, agentic environments** where success is verified by task completion.

## The Agent Training Paradigm Shift

```
Traditional RLVR (2024-2025):
┌─────────────────────────────────────────────────────────────┐
│  Prompt → Single Response → Verify (math/code correct?)     │
│                                                              │
│  Example: "What is 15 × 7?" → "105" → ✓ Correct!            │
│                                                              │
│  Reward: Binary (0/1) per response                          │
└─────────────────────────────────────────────────────────────┘

Agent RL (2025+):
┌─────────────────────────────────────────────────────────────┐
│  Task → Action₁ → Obs₁ → Action₂ → ... → Verify (task done?)│
│                                                              │
│  Example: "Fix bug #123" → read file → edit → run tests → ✓ │
│                                                              │
│  Reward: Sparse (only at end of multi-step trajectory)      │
│  Challenge: Credit assignment across many actions            │
└─────────────────────────────────────────────────────────────┘
```

## Key Challenges in Agent RL

| Challenge | Description | Solutions |
|-----------|-------------|-----------|
| **Sparse Rewards** | Only get reward at task completion | Agent guidance, curriculum learning, ORM |
| **Long Horizons** | Many steps before reward signal | Trajectory compression, checkpointing |
| **High Environment Cost** | Running real environments is slow | Async rollouts, sandboxing |
| **Exploration** | Complex action spaces | Self-evolving tasks, error feedback |
| **Credit Assignment** | Which action caused success/failure? | Process rewards, step-level feedback |

---

## Agent RL Papers & Methods

### 1. Agent-RLVR (Scale AI, June 2025)

> Paper: [arXiv:2506.11425](https://arxiv.org/abs/2506.11425)

**Problem**: RLVR fails in agentic settings due to sparse rewards (high failure rates).

**Key Innovation**: **Agent Guidance** - a mechanism inspired by human pedagogy that steers agents toward successful trajectories using diverse informational cues (plans, error feedback, environmental hints).

```
Agent-RLVR Training Loop:
┌─────────────────────────────────────────────────────────────┐
│  1. Agent attempts task → Initial trajectories              │
│                    ↓                                        │
│  2. Validate with unit tests (sparse reward)                │
│                    ↓                                        │
│  3. Generate agent guidance for failed attempts             │
│     • High-level strategic plans                            │
│     • Dynamic feedback on errors                            │
│     • Environmental interaction hints                       │
│                    ↓                                        │
│  4. Agent reattempts WITH guidance                          │
│                    ↓                                        │
│  5. Update policy via RLVR on guided trajectories           │
└─────────────────────────────────────────────────────────────┘
```

**Results**:
- Qwen-2.5-72B-Instruct: 9.4% → **22.4%** on SWE-Bench Verified
- With test-time reward model: **27.8%**
- Only 817 training environments needed

**Key Insight**: The guidance data is also useful for training test-time reward models.

---

### 2. WebAgent-R1 (EMNLP 2025)

> Paper: [arXiv:2505.16421](https://arxiv.org/abs/2505.16421) | [GitHub](https://github.com/weizhepei/WebAgent-R1)

**Problem**: Training web agents to interact with real websites.

**Key Innovation**: **M-GRPO** (Multi-turn GRPO) - extension of GRPO to multi-turn settings with:
- Dynamic context compression (reduce memory overhead)
- Asynchronous trajectory rollout (improve sampling efficiency)
- Binary task success rewards (no reward model needed)

```
WebAgent-R1 Framework:
┌─────────────────────────────────────────────────────────────┐
│  Warm-up Stage (Behavior Cloning):                          │
│  └── Train on demonstrations to initialize policy           │
│                                                              │
│  RL Stage (M-GRPO):                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  For each task:                                       │   │
│  │  1. Generate multiple trajectories (async rollout)    │   │
│  │  2. Get binary rewards (task success/failure)         │   │
│  │  3. Compute group-relative advantages                 │   │
│  │  4. Update policy with GRPO objective                 │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Results** (WebArena-Lite):
| Model | Before RL | After RL |
|-------|-----------|----------|
| Qwen-2.5-3B | 6.1% | **33.9%** |
| LLaMA-3.1-8B | 8.5% | **44.8%** |

Surpasses OpenAI o3 on this benchmark.

**Variants**:
- **WebAgent-R1-Zero**: No warm-up, pure RL from scratch
- **WebAgent-R1-CoT**: Incorporates long chain-of-thought reasoning

---

### 3. WebRL (ICLR 2025)

> Paper: [arXiv:2411.02337](https://arxiv.org/abs/2411.02337) | [GitHub](https://github.com/THUDM/WebRL)

**Problem**: Scarcity of training tasks and sparse feedback signals for web agents.

**Key Innovations**:
1. **Self-evolving curriculum**: Generate new tasks from failed attempts
2. **Outcome-supervised Reward Model (ORM)**: Learn to predict task success
3. **Adaptive RL strategies**: Adjust training based on agent performance

```
WebRL Self-Evolving Curriculum:
┌─────────────────────────────────────────────────────────────┐
│  Task Pool                                                   │
│      ↓                                                       │
│  Agent attempts tasks                                        │
│      ↓                                                       │
│  ┌──────────────┬──────────────┐                            │
│  │   Success    │   Failure    │                            │
│  │   (remove)   │   (analyze)  │                            │
│  └──────────────┴──────┬───────┘                            │
│                        ↓                                     │
│         Generate new harder tasks                            │
│                        ↓                                     │
│                Add to task pool                              │
└─────────────────────────────────────────────────────────────┘
```

**Results**:
- LLaMA-3.1-8B: 4.8% → **42.4%** success rate
- LLaMA-3.1-70B: **47.3%** (surpasses GPT-4-Turbo's 17.6% by 160%+)

---

### 4. UI-R1: GUI Agent RL (AAAI 2026)

> Paper: [arXiv:2503.21620](https://arxiv.org/abs/2503.21620) | [GitHub](https://github.com/lll6gg/UI-R1)

**Problem**: Training agents to interact with mobile/desktop GUIs.

**Key Innovation**: First rule-based RL framework for GUI action prediction with a novel **composite reward**:

```
UI-R1 Reward Function:
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  Reward = α × ActionTypeReward + (1-α) × GroundingReward    │
│                                                              │
│  ActionTypeReward:                                           │
│  └── Did agent select correct action type? (click, scroll)  │
│                                                              │
│  GroundingReward:                                            │
│  └── Are the coordinates correct? (IoU with target element) │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Results** (with only **136 training samples**!):
| Model | ScreenSpot | AndroidControl |
|-------|------------|----------------|
| Base (Qwen2.5-VL-3B) | Baseline | Baseline |
| UI-R1-3B | **+22.1%** | **+12.7%** |

Outperforms models trained on 76K samples via SFT.

**Key Insight**: Rule-based rewards are extremely data-efficient for well-defined tasks.

---

### 5. DeepSWE (Together AI, July 2025)

> Blog: [together.ai/blog/deepswe](https://www.together.ai/blog/deepswe) | Framework: [rLLM](https://github.com/agentica-project/rllm)

**Problem**: Train open-source coding agents that can match proprietary models.

**Key Innovation**: Pure RL training using **rLLM** framework on real-world SWE tasks.

```
DeepSWE Training:
┌─────────────────────────────────────────────────────────────┐
│  Base: Qwen3-32B                                             │
│                                                              │
│  Training:                                                   │
│  • 4,500 real-world SWE tasks from R2E-Gym                  │
│  • 6 days on 64 H100 GPUs                                   │
│  • Pure RL (no supervised fine-tuning)                      │
│                                                              │
│  Result: 23% → 42% Pass@1 with just 200 RL steps            │
│          Final: 59% on SWE-Bench Verified                   │
└─────────────────────────────────────────────────────────────┘
```

**rLLM Framework**:
- Modular RL framework for language agents
- Build custom agents and environments
- Train with reinforcement learning
- Deploy for real-world workloads

---

### 6. SWE-RL: Self-Play for SWE Agents

> Paper: [arXiv:2512.18552](https://arxiv.org/abs/2512.18552)

**Problem**: Current SWE agents depend heavily on human-labeled data.

**Key Innovation**: **Self-Play SWE-RL (SSR)** - agents learn by:
1. Injecting bugs into code
2. Learning to repair those bugs
3. Progressively increasing bug complexity

```
Self-Play SWE-RL Loop:
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  ┌──────────────┐      ┌──────────────┐                     │
│  │  Bug Injector │ ←── │ Same Agent   │                     │
│  │  (generate   │      │ (learns to   │                     │
│  │   harder     │      │  inject and  │                     │
│  │   bugs)      │      │  repair)     │                     │
│  └──────────────┘      └──────────────┘                     │
│         │                     ↑                              │
│         └────── competes ─────┘                              │
│                                                              │
│  Minimal data assumptions:                                   │
│  • Only need: code repos + dependencies                      │
│  • No human-labeled issues or tests                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Vision**: Path toward training **superintelligent** software agents.

---

## Industry Approaches

### OpenAI: Codex Agent Training

**GPT-5-Codex / GPT-5.2-Codex** (2025):

```
OpenAI Codex Training:
┌─────────────────────────────────────────────────────────────┐
│  Training Tasks:                                             │
│  • Building full projects from scratch                       │
│  • Adding features and tests                                 │
│  • Debugging                                                 │
│  • Large-scale refactors                                     │
│  • Code reviews                                              │
│                                                              │
│  Key Properties:                                             │
│  • Trained to align with human coding preferences            │
│  • Works in isolated sandbox (no external internet)          │
│  • Can work 24+ hours autonomously on complex tasks          │
│  • Ask for permission before potentially dangerous actions   │
│                                                              │
│  Agent Skills:                                               │
│  • Extend Codex with task-specific workflows                 │
│  • Package instructions + resources + scripts                │
│  • Share across teams and community                          │
└─────────────────────────────────────────────────────────────┘
```

**Results**: 2M+ public pull requests merged on GitHub using Codex.

---

### Anthropic: Computer Use & Agent Infrastructure

**Computer Use** (Oct 2024 - ongoing):

```
Claude Computer Use:
┌─────────────────────────────────────────────────────────────┐
│  Training:                                                   │
│  • Trained to perceive screens visually                      │
│  • Translates visual data to coordinate grid                 │
│  • "Counts pixels" to locate UI elements                     │
│                                                              │
│  Capabilities:                                               │
│  • Look at screen, move cursor, click, type                  │
│  • Take frequent screenshots to maintain context             │
│  • OSWorld benchmark: 61.4% (Sonnet 4.5)                    │
│                                                              │
│  2025 Progress:                                              │
│  • Success rates: ~80%+ for standard office tasks           │
│  • Available via Claude 4 API                                │
└─────────────────────────────────────────────────────────────┘
```

**Claude Agent SDK** (Sept 2025):
- Build agents on top of Claude Code
- Used internally for: deep research, video creation, note-taking
- Powers most major agent loops at Anthropic

**Agent Skills** (Oct 2025):
- Open standard for reusable agent workflows
- Create, deploy, share, and discover skills
- Extends Claude's capabilities with domain-specific knowledge

---

## Open Standards & Ecosystem

### Model Context Protocol (MCP) - Anthropic

> "USB-C for AI" - standardized way to connect LLMs to external tools

```
Before MCP:                    After MCP:
┌─────────┐  ┌─────────┐      ┌─────────┐
│ LLM A   │──│ Tool 1  │      │         │
└─────────┘  └─────────┘      │         │
┌─────────┐  ┌─────────┐      │   MCP   │──── Any Tool
│ LLM B   │──│ Tool 1  │      │ Protocol│
└─────────┘  └─────────┘      │         │
                              │         │
Custom integration            │         │
for each LLM × Tool           └─────────┘
                              ↑
                              One integration
                              works everywhere
```

**Adoption**: Becoming the industry standard for tool connections.

---

### AGENTS.md - OpenAI

> Specification for how agents should behave in codebases

- Adopted by **60,000+ open-source projects**
- Used by: Cursor, Devin, GitHub Copilot, Gemini CLI, VS Code, etc.
- Defines agent capabilities, permissions, and behaviors

---

### Agentic AI Foundation (AAIF)

> Linux Foundation initiative for agent interoperability

**Founded by**: OpenAI, Anthropic, Block

**Supported by**: Google, Microsoft, AWS, Bloomberg, Cloudflare

**Goals**:
- Neutral stewardship for open, interoperable agent infrastructure
- Standards for agentic AI in production
- Includes MCP and AGENTS.md donations

---

## Agent Training Comparison

| Method | Environment | Reward | Data Efficiency | Scalability |
|--------|-------------|--------|-----------------|-------------|
| **Agent-RLVR** | SWE/Code | Unit tests + guidance | High (817 tasks) | Good |
| **WebAgent-R1** | Web/Browser | Binary success | Medium | Good |
| **WebRL** | Web/Browser | ORM | Medium | Good |
| **UI-R1** | Mobile GUI | Rule-based composite | Very High (136 tasks) | Good |
| **DeepSWE** | SWE/Code | Unit tests | Medium (4500 tasks) | Excellent |
| **SWE-RL** | SWE/Code | Self-play | Minimal human data | Excellent |

---

## The Common Pattern

All successful Agent RL approaches share:

```
1. Verifiable Reward
   └── Task completion can be objectively verified
       (tests pass, task completes, correct UI element clicked)

2. GRPO-family Algorithms
   └── Not PPO (too complex)
   └── Group-relative advantages, no value network

3. Warm-start with Behavior Cloning
   └── Initialize with demonstrations/SFT
   └── Then refine with RL

4. Guidance Mechanisms
   └── To overcome sparse rewards
   └── Plans, hints, error feedback, curriculum
```

---

## Practical Considerations

### When to Use Agent RL

| Use Case | Recommended Approach |
|----------|---------------------|
| **SWE agents** | Agent-RLVR, DeepSWE |
| **Web automation** | WebAgent-R1, WebRL |
| **GUI/mobile** | UI-R1, MobileGUI-RL |
| **General agents** | Start with SFT, then RL |

### Compute Requirements

| Method | GPUs | Training Time | Data |
|--------|------|---------------|------|
| Agent-RLVR | ~8-16 H100 | Days | 817 tasks |
| WebAgent-R1 | ~8 H100 | Days | WebArena |
| UI-R1 | ~4 H100 | Hours | 136 samples |
| DeepSWE | 64 H100 | 6 days | 4500 tasks |

### Key Frameworks

| Framework | Use Case | Link |
|-----------|----------|------|
| **rLLM** | RL for language agents | [GitHub](https://github.com/agentica-project/rllm) |
| **veRL** | High-throughput RLHF | ByteDance |
| **TRL** | General RL fine-tuning | HuggingFace |
| **OpenRLHF** | Distributed RLHF | Open source |

---

## Future Directions (2026+)

```
2025: Agent RL proves effective for specific domains
      • SWE, web, GUI agents trained with RL

2026: Agent RL scales and generalizes
      • Cross-domain agent training
      • Self-play and curriculum learning
      • Reduced human data requirements
      • Online/continual learning agents

Beyond: Toward agentic superintelligence
      • Agents that learn from experience
      • Self-improving through interaction
      • Minimal human oversight needed
```

**Key prediction** from research: RLVR will expand beyond math/code into structured compliance, medical guidelines, and general agentic tasks where verifiers can be built.

---

## Resources

### Papers

| Paper | Year | Focus |
|-------|------|-------|
| [Agent-RLVR](https://arxiv.org/abs/2506.11425) | 2025 | SWE agents + guidance |
| [WebAgent-R1](https://arxiv.org/abs/2505.16421) | 2025 | Multi-turn web agents |
| [WebRL](https://arxiv.org/abs/2411.02337) | 2024 | Self-evolving curriculum |
| [UI-R1](https://arxiv.org/abs/2503.21620) | 2025 | GUI agents |
| [SWE-RL](https://arxiv.org/abs/2512.18552) | 2025 | Self-play SWE agents |
| [DeepSWE](https://www.together.ai/blog/deepswe) | 2025 | Open-source coding agent |

### Industry Resources

| Resource | Organization |
|----------|--------------|
| [Model Context Protocol](https://modelcontextprotocol.io/) | Anthropic |
| [AGENTS.md Specification](https://github.com/anthropics/agents-md) | OpenAI |
| [Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) | Anthropic |
| [Codex Agent Skills](https://developers.openai.com/codex/skills/) | OpenAI |
| [Agentic AI Foundation](https://openai.com/index/agentic-ai-foundation/) | Linux Foundation |

### Tutorials & Blogs

- [Andrej Karpathy's 2025 LLM Review](https://karpathy.bearblog.dev/year-in-review-2025/)
- [State of RL in 2025](https://www.turingpost.com/p/stateofrl2025)
- [How AI is Transforming Work at Anthropic](https://www.anthropic.com/research/how-ai-is-transforming-work-at-anthropic)
