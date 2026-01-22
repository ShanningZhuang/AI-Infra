# RL Frameworks for LLM Alignment

> Parent: [RL & Alignment Overview](rl.md)

## Overview

Several frameworks have emerged to simplify RLHF and alignment training. Each has different strengths in terms of ease of use, scalability, and feature support.

## Learning Objectives

- [ ] TRL (Transformer Reinforcement Learning)
- [ ] OpenRLHF
- [ ] DeepSpeed-Chat
- [ ] veRL
- [ ] Framework selection criteria

## Resources

### Documentation

- [TRL Documentation](https://huggingface.co/docs/trl)
- [OpenRLHF GitHub](https://github.com/OpenRLHF/OpenRLHF)
- [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)
- [veRL Documentation](https://github.com/volcengine/verl)

### Papers

- [OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework](https://arxiv.org/abs/2405.11143)
- [DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training](https://arxiv.org/abs/2308.01320)

---

## Notes

### Framework Comparison

| Feature | TRL | OpenRLHF | DeepSpeed-Chat | veRL |
|---------|-----|----------|----------------|------|
| Ease of use | ★★★★★ | ★★★★☆ | ★★★☆☆ | ★★★☆☆ |
| Scalability | ★★★☆☆ | ★★★★★ | ★★★★★ | ★★★★★ |
| DPO | ✅ | ✅ | ❌ | ✅ |
| PPO | ✅ | ✅ | ✅ | ✅ |
| GRPO | ❌ | ✅ | ❌ | ✅ |
| Multi-node | Limited | ✅ | ✅ | ✅ |
| vLLM integration | ❌ | ✅ | ❌ | ✅ |
| HuggingFace integration | ✅ | ✅ | ❌ | Partial |

### TRL (Transformer Reinforcement Learning)

**Best for:** Quick experiments, single-node training, HuggingFace ecosystem

```python
# DPO Training with TRL
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Config
config = DPOConfig(
    beta=0.1,
    learning_rate=5e-7,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    output_dir="./dpo-llama",
)

# Train
trainer = DPOTrainer(
    model=model,
    ref_model=None,  # Will create copy automatically
    args=config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

```python
# PPO Training with TRL
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

# Model with value head
model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")

# Config
config = PPOConfig(
    learning_rate=1e-5,
    batch_size=64,
    mini_batch_size=16,
    ppo_epochs=4,
)

# Trainer
trainer = PPOTrainer(
    config=config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# Training loop
for batch in dataloader:
    responses = trainer.generate(batch["input_ids"])
    rewards = reward_model(batch["input_ids"], responses)
    stats = trainer.step(batch["input_ids"], responses, rewards)
```

**TRL Pros:**
- Simple API, great documentation
- Tight HuggingFace integration
- Active development
- SFTTrainer, RewardTrainer included

**TRL Cons:**
- Limited multi-node support
- No vLLM integration (slower generation)
- Memory hungry for large models

### OpenRLHF

**Best for:** Large-scale training, production RLHF, multi-node

```python
# OpenRLHF PPO training
from openrlhf.trainer import PPOTrainer

trainer = PPOTrainer(
    pretrain="meta-llama/Llama-2-7b-hf",
    reward_pretrain="reward-model-path",
    save_path="./rlhf-output",

    # Distributed settings
    actor_num_nodes=1,
    actor_num_gpus_per_node=4,
    critic_num_nodes=1,
    critic_num_gpus_per_node=4,

    # Training settings
    micro_train_batch_size=4,
    train_batch_size=128,
    max_epochs=1,
    prompt_max_len=1024,
    generate_max_len=512,

    # PPO settings
    init_kl_coef=0.01,
    ppo_epochs=1,
)

trainer.fit()
```

**OpenRLHF Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                      OpenRLHF                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌───────────────┐    ┌───────────────┐                     │
│  │ Actor (Policy)│    │ Critic (Value)│                     │
│  │ + Reference   │    │               │                     │
│  │ DeepSpeed ZeRO│    │ DeepSpeed ZeRO│                     │
│  └───────────────┘    └───────────────┘                     │
│          │                    │                              │
│          └────────────────────┘                              │
│                    │                                         │
│          ┌─────────▼─────────┐                              │
│          │  vLLM Generation  │ ← Fast rollouts               │
│          │  (optional)       │                              │
│          └───────────────────┘                              │
│                    │                                         │
│          ┌─────────▼─────────┐                              │
│          │   Reward Model    │                              │
│          └───────────────────┘                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**OpenRLHF Features:**
- vLLM integration for fast generation
- Ray-based distributed training
- Support for 70B+ models
- GRPO, DPO, PPO algorithms
- Reward model training included

```bash
# Example: Multi-node PPO training
ray start --head --node-ip-address=$HEAD_IP

deepspeed --hostfile hostfile \
    examples/train_ppo.py \
    --pretrain meta-llama/Llama-2-7b-hf \
    --reward_pretrain reward_model \
    --save_path ./output \
    --micro_train_batch_size 2 \
    --train_batch_size 128
```

### DeepSpeed-Chat

**Best for:** Microsoft ecosystem, DeepSpeed users, three-stage training

```python
# DeepSpeed-Chat uses a step-by-step approach
# Step 1: SFT
deepspeed main.py \
    --data_path ./data \
    --model_name_or_path facebook/opt-1.3b \
    --per_device_train_batch_size 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 3

# Step 2: Reward Model
deepspeed main.py \
    --data_path ./comparison_data \
    --model_name_or_path facebook/opt-1.3b \
    --per_device_train_batch_size 8

# Step 3: RLHF
deepspeed main.py \
    --actor_model_name_or_path step1_model \
    --critic_model_name_or_path step2_model \
    --per_device_train_batch_size 8
```

**DeepSpeed-Chat Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                    DeepSpeed-Chat                            │
├─────────────────────────────────────────────────────────────┤
│  Hybrid Engine (HE)                                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Training Mode                Inference Mode         │    │
│  │  (ZeRO-3)         ←────→      (Tensor Parallel)     │    │
│  │                                                      │    │
│  │  Automatic switching based on forward/generate      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Memory Optimizations:                                       │
│  - ZeRO-Offload (CPU offloading)                            │
│  - ZeRO-Infinity (NVMe offloading)                          │
│  - Activation checkpointing                                  │
└─────────────────────────────────────────────────────────────┘
```

**DeepSpeed-Chat Pros:**
- Hybrid Engine for efficient training + inference
- Battle-tested ZeRO optimizations
- Good for very large models

**DeepSpeed-Chat Cons:**
- More complex setup
- Less active development than TRL/OpenRLHF
- Limited algorithm support (PPO only)

### veRL

**Best for:** ByteDance ecosystem, flexible architecture, production

```python
# veRL configuration
from verl import RLHFConfig, RLHFTrainer

config = RLHFConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    algorithm="grpo",  # or "ppo", "dpo"

    # Generation
    rollout_engine="vllm",
    rollout_batch_size=512,

    # Training
    trainer_engine="fsdp",
    train_batch_size=64,
    learning_rate=1e-6,

    # Resources
    n_gpus_per_node=8,
    n_nodes=4,
)

trainer = RLHFTrainer(config)
trainer.train()
```

**veRL Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                         veRL                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Resource Pool Manager                    │    │
│  │  Dynamically allocates GPUs between:                 │    │
│  │  - Generation (vLLM)                                 │    │
│  │  - Training (FSDP)                                   │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                    │
│         ┌───────────────┴───────────────┐                   │
│         ▼                               ▼                   │
│  ┌─────────────────┐           ┌─────────────────┐         │
│  │  Generation     │           │   Training      │         │
│  │  Phase          │    ───▶   │   Phase         │         │
│  │  (all GPUs)     │           │   (all GPUs)    │         │
│  └─────────────────┘           └─────────────────┘         │
│         │                               │                   │
│         └───────────────────────────────┘                   │
│                         │                                    │
│                  (repeat)                                    │
└─────────────────────────────────────────────────────────────┘

Key insight: Don't partition GPUs, time-share them!
```

**veRL Features:**
- Colocated generation + training
- FSDP + vLLM integration
- GRPO, PPO, DPO support
- Good throughput optimization

### Quick Start Comparison

**For a 7B model on 8 A100s:**

| Framework | Setup Time | Lines of Code | Training Speed |
|-----------|------------|---------------|----------------|
| TRL | 30 min | ~50 | 1x (baseline) |
| OpenRLHF | 1 hour | ~100 | 2-3x |
| DeepSpeed-Chat | 2 hours | ~200 | 2x |
| veRL | 1 hour | ~80 | 2-3x |

### When to Use Which

| Scenario | Recommended Framework |
|----------|----------------------|
| Quick prototyping | TRL |
| Single GPU | TRL |
| Multi-node PPO | OpenRLHF |
| 70B+ model | OpenRLHF or veRL |
| DeepSpeed ecosystem | DeepSpeed-Chat |
| GRPO training | OpenRLHF or veRL |
| DPO only | TRL |
| Maximum throughput | veRL |
| HuggingFace Hub | TRL |

### Common Configurations

**TRL DPO (7B, 8 A100):**
```python
DPOConfig(
    beta=0.1,
    learning_rate=5e-7,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    bf16=True,
    gradient_checkpointing=True,
)
```

**OpenRLHF PPO (7B, 8 A100):**
```python
{
    "micro_train_batch_size": 2,
    "train_batch_size": 128,
    "rollout_batch_size": 64,
    "actor_learning_rate": 1e-6,
    "critic_learning_rate": 5e-6,
    "init_kl_coef": 0.01,
    "use_vllm": True,
}
```

**veRL GRPO (7B, 32 A100 across 4 nodes):**
```python
RLHFConfig(
    algorithm="grpo",
    rollout_engine="vllm",
    trainer_engine="fsdp",
    rollout_batch_size=1024,
    train_batch_size=256,
    grpo_group_size=4,
)
```

### Migration Between Frameworks

**TRL → OpenRLHF:**
```python
# TRL model
model = AutoModelForCausalLM.from_pretrained("trained-dpo-model")
model.save_pretrained("./model-for-openrlhf")

# Load in OpenRLHF
trainer = PPOTrainer(pretrain="./model-for-openrlhf", ...)
```

**Checkpoints are generally compatible** since all frameworks use HuggingFace model format.

### Debugging Tips

**TRL:**
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check reward distribution
print(f"Rewards: mean={rewards.mean():.2f}, std={rewards.std():.2f}")
```

**OpenRLHF:**
```bash
# Use Ray dashboard
ray dashboard

# Check actor/critic logs separately
tail -f logs/actor.log
tail -f logs/critic.log
```

**veRL:**
```python
# Monitor phase transitions
config.log_phase_transitions = True
config.profile_memory = True
```

### Cost Comparison (Rough)

For aligning a 7B model (1 epoch on 10K samples):

| Framework | Hardware | Time | Cloud Cost |
|-----------|----------|------|------------|
| TRL (DPO) | 1× A100 | 4h | ~$8 |
| TRL (PPO) | 8× A100 | 8h | ~$128 |
| OpenRLHF (PPO) | 8× A100 | 4h | ~$64 |
| veRL (GRPO) | 8× A100 | 3h | ~$48 |

*Costs are approximate based on cloud GPU pricing.*
