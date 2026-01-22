# RL Infrastructure for LLMs

> Parent: [RL & Alignment Overview](rl.md)

## Overview

Training LLMs with RL at scale requires specialized infrastructure. This covers distributed training architectures, experience buffer management, and the systems challenges unique to RL for language models.

## Learning Objectives

- [ ] Distributed PPO/GRPO training
- [ ] Actor-critic parallelism
- [ ] Experience buffer management
- [ ] Rollout generation at scale
- [ ] Memory optimization for RL

## Resources

### Papers

- [OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework](https://arxiv.org/abs/2405.11143)
- [DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training](https://arxiv.org/abs/2308.01320)
- [Sample Factory: Egocentric 3D Control from 2D Observations](https://arxiv.org/abs/2006.11751) - High-throughput RL

### Code References

- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [veRL](https://github.com/volcengine/verl)
- [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)
- [TRL](https://github.com/huggingface/trl)

---

## Notes

### The RL Training Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                    RLHF Training Loop                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Generate   │───▶│    Score     │───▶│   Compute    │       │
│  │  Responses   │    │  (Reward)    │    │  Advantages  │       │
│  │  (Rollout)   │    │              │    │              │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         ↑                                        │               │
│         │            ┌──────────────┐            │               │
│         └────────────│    Policy    │◀───────────┘               │
│                      │    Update    │                            │
│                      └──────────────┘                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Challenge: 4 Models

Standard RLHF requires 4 models in memory:

```
Model         Size (70B)    Purpose
─────────────────────────────────────────────────
Policy        140 GB        Being optimized
Reference     140 GB        KL penalty anchor
Reward        140 GB        Scores responses
Value         140 GB        Estimates returns
─────────────────────────────────────────────────
Total         560 GB        Needs 8× A100-80GB!
```

**Optimizations:**
1. Share backbone between policy/reference (LoRA)
2. Use smaller reward/value models
3. Offload to CPU when not in use
4. Use model parallelism

### Architecture Patterns

**Pattern 1: Colocated (Simple)**

All models on same GPUs:

```
┌─────────────────────────────────────────────────────────────┐
│  GPU 0-7 (8× A100)                                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Policy + Reference + Reward + Value                  │   │
│  │ (shared via model parallelism)                       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

Pros: Simple, low latency
Cons: Memory constrained, underutilized during generation
```

**Pattern 2: Disaggregated (Scalable)**

Separate generation and training:

```
┌───────────────────────────┐    ┌───────────────────────────┐
│     Generation Cluster     │    │     Training Cluster      │
│  ┌─────────────────────┐  │    │  ┌─────────────────────┐  │
│  │  Inference Engine   │  │    │  │    Training Loop    │  │
│  │  (vLLM, TGI)        │  │────│  │  (Policy update)    │  │
│  │                     │  │    │  │                     │  │
│  │  Policy + Reference │  │    │  │  Policy + Value     │  │
│  └─────────────────────┘  │    │  └─────────────────────┘  │
└───────────────────────────┘    └───────────────────────────┘
          │                                  │
          │         Experience Buffer        │
          └──────────────▼───────────────────┘
                   (Ray, Redis)
```

**Pattern 3: Hybrid (veRL style)**

Dynamically switch between generation and training:

```
Phase 1: Generation (inference mode)
┌─────────────────────────────────────────┐
│ All GPUs → Inference engine (vLLM)      │
│ Generate responses, store to buffer     │
└─────────────────────────────────────────┘
                    ↓
Phase 2: Training (training mode)
┌─────────────────────────────────────────┐
│ All GPUs → Training (FSDP/DeepSpeed)    │
│ Update policy on buffered experiences   │
└─────────────────────────────────────────┘
                    ↓
           (repeat)
```

### Experience Buffer Management

```python
class ExperienceBuffer:
    """
    Stores (prompt, response, reward, log_prob) tuples
    for PPO training
    """
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size

    def add(self, prompt, response, reward, log_prob, ref_log_prob):
        experience = {
            "prompt_ids": prompt,
            "response_ids": response,
            "reward": reward,
            "old_log_prob": log_prob,
            "ref_log_prob": ref_log_prob,
        }
        self.buffer.append(experience)

        # Remove old experiences if over capacity
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        return [self.buffer[i] for i in indices]

    def clear(self):
        self.buffer = []
```

**On-policy vs Off-policy:**
- PPO is on-policy: use fresh experiences, discard after update
- Can reuse for 1-2 epochs before too stale
- Off-policy (rare for LLMs): can reuse older experiences

### Rollout Generation

**Synchronous rollouts:**
```python
def synchronous_rollout(policy, prompts, batch_size):
    """
    Generate responses synchronously
    Simple but slow (GPU idle during generation)
    """
    responses = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        batch_responses = policy.generate(batch)
        responses.extend(batch_responses)
    return responses
```

**Asynchronous rollouts:**
```python
async def async_rollout(inference_engine, prompts):
    """
    Generate responses asynchronously using inference engine
    Maximizes GPU utilization
    """
    async with inference_engine.session() as session:
        tasks = [session.generate(p) for p in prompts]
        responses = await asyncio.gather(*tasks)
    return responses
```

### Distributed Training with Ray

```python
import ray
from ray import train

@ray.remote(num_gpus=1)
class RolloutWorker:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def generate(self, prompts):
        return self.model.generate(prompts)

@ray.remote(num_gpus=8)
class TrainingWorker:
    def __init__(self, model_path):
        self.model = load_model_fsdp(model_path)

    def update(self, experiences):
        return self.ppo_step(experiences)

# Main loop
rollout_workers = [RolloutWorker.remote(path) for _ in range(4)]
training_worker = TrainingWorker.remote(path)

for epoch in range(num_epochs):
    # Parallel rollouts
    futures = [w.generate.remote(batch) for w, batch in zip(rollout_workers, batches)]
    experiences = ray.get(futures)

    # Update
    stats = ray.get(training_worker.update.remote(experiences))
```

### Weight Synchronization

After policy update, sync weights to rollout workers:

```python
def sync_weights(training_worker, rollout_workers):
    """
    Broadcast updated weights from training to rollout workers
    """
    # Get state dict from training worker
    state_dict = ray.get(training_worker.get_state_dict.remote())

    # Broadcast to all rollout workers
    futures = [w.load_state_dict.remote(state_dict) for w in rollout_workers]
    ray.get(futures)
```

**Optimization: Incremental updates**
- Only sync changed parameters (LoRA weights)
- Use shared storage (NFS, S3) for large models
- Async broadcast while generating next batch

### Memory Optimization Techniques

**1. Gradient Checkpointing:**
```python
model.gradient_checkpointing_enable()
# Trades compute for memory
# Essential for large models
```

**2. Mixed Precision:**
```python
# FP16/BF16 for forward pass
# FP32 for loss and optimizer
scaler = torch.cuda.amp.GradScaler()
```

**3. LoRA for Reference:**
```python
# Instead of full reference model copy
# Share base, only compare LoRA weights
policy = base_model + LoRA_policy
reference = base_model  # Shared!
```

**4. Offloading:**
```python
# Offload reward model to CPU when not scoring
reward_model.to("cpu")
# ... generate responses ...
reward_model.to("cuda")
rewards = reward_model(responses)
```

### VRAM Budget Example

For 7B model with PPO on single A100-80GB:

```
Component           VRAM Usage
──────────────────────────────────
Policy (BF16)       14 GB
Reference (BF16)    14 GB
Reward (BF16)       14 GB
Value (BF16)        14 GB
Optimizer states    28 GB (AdamW, FP32)
Activations         8 GB (with checkpointing)
──────────────────────────────────
Total               92 GB → DOESN'T FIT

With optimizations:
- LoRA for policy      2 GB
- Shared ref backbone  0 GB (shared)
- Smaller reward (1B)  2 GB
- No value (GRPO)      0 GB
- Optimizer (8-bit)    4 GB
- Activations          8 GB
──────────────────────────────────
Total                 30 GB → FITS!
```

### Throughput Optimization

**Bottleneck analysis:**

```
Phase           Time    Bottleneck
─────────────────────────────────────
Generation      60%     Memory bandwidth (autoregressive)
Reward scoring  10%     Forward pass
Training        20%     Backward pass
Data transfer   10%     CPU-GPU, network

Optimizations:
- Continuous batching for generation
- Batch reward scoring
- Gradient accumulation for training
- Async data prefetch
```

**Tokens per second targets:**

| Scale | Tokens/sec | Hardware |
|-------|------------|----------|
| Small | 1K-10K | Single GPU |
| Medium | 10K-100K | 8 GPUs |
| Large | 100K-1M | Multi-node |

### Fault Tolerance

```python
class CheckpointManager:
    def __init__(self, save_dir, keep_last=3):
        self.save_dir = save_dir
        self.keep_last = keep_last

    def save(self, model, optimizer, step, metrics):
        path = f"{self.save_dir}/step_{step}"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "metrics": metrics,
        }, path)

        # Clean old checkpoints
        self._cleanup()

    def resume(self, model, optimizer):
        latest = self._find_latest()
        if latest:
            ckpt = torch.load(latest)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            return ckpt["step"]
        return 0
```

### Monitoring and Debugging

Key metrics to track:

```python
metrics = {
    # Reward
    "reward/mean": rewards.mean(),
    "reward/std": rewards.std(),
    "reward/max": rewards.max(),
    "reward/min": rewards.min(),

    # KL divergence
    "kl/mean": kl.mean(),
    "kl/max": kl.max(),

    # Policy
    "policy/entropy": entropy.mean(),
    "policy/clip_fraction": clip_fraction,

    # Value (if using)
    "value/loss": value_loss,
    "value/explained_var": explained_variance,

    # System
    "throughput/tokens_per_sec": tokens / time,
    "memory/allocated_gb": torch.cuda.memory_allocated() / 1e9,
}
```

### Common Infrastructure Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| OOM | CUDA out of memory | Reduce batch, enable checkpointing |
| Slow generation | Low throughput | Use inference engine (vLLM) |
| Weight sync lag | Stale rollouts | Reduce sync interval |
| Gradient explosion | NaN loss | Gradient clipping, lower LR |
| Reward collapse | All same reward | Check reward model, add diversity |
