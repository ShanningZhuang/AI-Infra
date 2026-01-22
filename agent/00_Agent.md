# Agent Infrastructure

> Parent: [AI Infrastructure](../00_AI_Infra.md)

## Overview

Agent infrastructure is the systems layer that enables LLMs to take actions in the world. This covers the patterns, frameworks, and infrastructure needed to build reliable, scalable AI agents that can use tools, maintain memory, and work together.

## What is an Agent?

```
┌─────────────────────────────────────────────────────────────────┐
│                         LLM Agent                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│   │   Observe   │────▶│   Reason    │────▶│    Act      │       │
│   │  (context)  │     │  (LLM call) │     │   (tools)   │       │
│   └─────────────┘     └─────────────┘     └─────────────┘       │
│         ↑                                        │               │
│         │                                        │               │
│         └────────────────────────────────────────┘               │
│                      (loop until done)                           │
│                                                                  │
│   Components:                                                    │
│   • LLM (reasoning engine)                                       │
│   • Tools (actions it can take)                                  │
│   • Memory (context across turns)                                │
│   • Orchestrator (manages the loop)                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Why Agent Infrastructure Matters

| Challenge | Infrastructure Solution |
|-----------|------------------------|
| Tool execution reliability | Retry strategies, error handling |
| Long conversations | Memory management, context compression |
| Multi-step tasks | State management, checkpointing |
| Parallel actions | Async execution, dependency graphs |
| Production deployment | Observability, rate limiting, cost control |
| Multi-agent coordination | Communication protocols, consensus |

## Learning Path

### Core Concepts

| Topic | File | Priority |
|-------|------|----------|
| Agent Fundamentals | [01_Fundamentals.md](01_Fundamentals.md) | ★★★★★ |
| Execution Infrastructure | [02_Execution.md](02_Execution.md) | ★★★★★ |
| Multi-Agent Systems | [03_Multi_Agent.md](03_Multi_Agent.md) | ★★★★☆ |

### Production & Operations

| Topic | File | Priority |
|-------|------|----------|
| Agent Serving | [04_Serving.md](04_Serving.md) | ★★★★☆ |
| Production Considerations | [05_Production.md](05_Production.md) | ★★★★☆ |

---

## The Agent Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                           │
│   User interfaces, APIs, integrations                            │
├─────────────────────────────────────────────────────────────────┤
│                    Orchestration Layer                           │
│   Agent frameworks (LangChain, LlamaIndex, custom)              │
├─────────────────────────────────────────────────────────────────┤
│                      Memory Layer                                │
│   Short-term (context), Long-term (vector DB), Episodic         │
├─────────────────────────────────────────────────────────────────┤
│                       Tool Layer                                 │
│   APIs, code execution, file systems, databases                 │
├─────────────────────────────────────────────────────────────────┤
│                     LLM Serving Layer                            │
│   vLLM, SGLang, TensorRT-LLM (see inference/)                   │
└─────────────────────────────────────────────────────────────────┘
```

## Key Papers

| Paper | Year | Key Contribution |
|-------|------|------------------|
| [ReAct](https://arxiv.org/abs/2210.03629) | 2022 | Reasoning + Acting paradigm |
| [Toolformer](https://arxiv.org/abs/2302.04761) | 2023 | Self-taught tool use |
| [Generative Agents](https://arxiv.org/abs/2304.03442) | 2023 | Memory and reflection |
| [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT) | 2023 | Autonomous agents |
| [Voyager](https://arxiv.org/abs/2305.16291) | 2023 | Skill library, curriculum |
| [Language Agent Tree Search](https://arxiv.org/abs/2310.04406) | 2023 | MCTS for agents |
| [AgentBench](https://arxiv.org/abs/2308.03688) | 2023 | Agent evaluation |

## Key Blogs & Resources

- [LangChain Documentation](https://python.langchain.com/docs/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use Guide](https://docs.anthropic.com/claude/docs/tool-use)
- [Lilian Weng: LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)

---

## Notes

### Agent Patterns Overview

| Pattern | Description | Use Case |
|---------|-------------|----------|
| ReAct | Reason then Act, iteratively | General tasks |
| Plan-and-Execute | Plan first, then execute steps | Complex multi-step |
| Reflexion | Self-reflect on failures | Learning from mistakes |
| Tree-of-Thoughts | Explore multiple paths | Complex reasoning |
| MCTS | Monte Carlo Tree Search | Game-like decisions |

### The ReAct Loop

```python
def react_agent(task, tools, max_steps=10):
    """
    Basic ReAct agent implementation
    """
    context = f"Task: {task}\n"

    for step in range(max_steps):
        # Reason: Ask LLM what to do
        response = llm.generate(
            f"{context}\n"
            f"Think step by step. What should I do next?\n"
            f"Available tools: {tools}\n"
            f"Respond with: Thought: <reasoning>\nAction: <tool_name>\nInput: <input>"
        )

        thought, action, action_input = parse_response(response)
        context += f"\nThought: {thought}\nAction: {action}\nInput: {action_input}\n"

        # Check if done
        if action == "finish":
            return action_input

        # Act: Execute the tool
        try:
            result = tools[action].execute(action_input)
            context += f"Observation: {result}\n"
        except Exception as e:
            context += f"Observation: Error - {e}\n"

    return "Max steps reached without completion"
```

### Memory Types

```
┌─────────────────────────────────────────────────────────────────┐
│                       Agent Memory                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Short-term Memory (Working Memory):                             │
│  └── Current conversation context                                │
│  └── Recent tool outputs                                         │
│  └── Limited by context window                                   │
│                                                                  │
│  Long-term Memory (Persistent):                                  │
│  └── Vector database for semantic search                         │
│  └── Key-value store for facts                                   │
│  └── Survives across sessions                                    │
│                                                                  │
│  Episodic Memory (Experiences):                                  │
│  └── Past task completions                                       │
│  └── Successful strategies                                       │
│  └── Mistakes to avoid                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Tool Calling Formats

**OpenAI Function Calling:**
```json
{
  "name": "get_weather",
  "description": "Get current weather for a location",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {"type": "string", "description": "City name"},
      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["location"]
  }
}
```

**Anthropic Tool Use:**
```json
{
  "name": "get_weather",
  "description": "Get current weather for a location",
  "input_schema": {
    "type": "object",
    "properties": {
      "location": {"type": "string"},
      "unit": {"type": "string"}
    },
    "required": ["location"]
  }
}
```

### Agent Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Task Success Rate | % of tasks completed correctly |
| Steps to Completion | Efficiency of agent |
| Tool Call Accuracy | Correct tool selection |
| Error Recovery Rate | Handling of failures |
| Cost per Task | API calls, compute |
| Latency | Time to complete |

### Common Failure Modes

| Failure | Cause | Mitigation |
|---------|-------|------------|
| Infinite loops | No termination condition | Max steps, loop detection |
| Wrong tool | Ambiguous descriptions | Better prompts, examples |
| Hallucinated tools | Model invents tools | Strict parsing, validation |
| Context overflow | Too much history | Summarization, pruning |
| Error cascades | Unhandled exceptions | Try-catch, fallbacks |
| Stuck states | No progress | Backtracking, restarts |

### Infrastructure Considerations

**Latency Budget:**
```
User request to response:
├── Agent reasoning:     500ms (LLM call)
├── Tool execution:      variable
│   ├── API call:        100-2000ms
│   ├── Code execution:  10-10000ms
│   └── DB query:        10-100ms
├── Response generation: 500ms (LLM call)
└── Total:              1-15+ seconds per step

For 5-step tasks: 5-75+ seconds total
```

**Cost Budget:**
```
Per agent step:
├── Input tokens:    ~1000 (context + prompt)
├── Output tokens:   ~200 (reasoning + action)
├── Tool costs:      variable
└── Total:          ~$0.01-0.05 per step

For 10-step task: $0.10-0.50 per task
```

### Qwen Agent Focus Areas

For Qwen's Agent Infrastructure team, key areas include:

1. **Tool Integration**: Efficient tool calling protocols
2. **Memory Systems**: Scalable long-term memory
3. **Multi-Agent**: Coordination for complex tasks
4. **Serving**: Low-latency agent inference
5. **Evaluation**: Benchmarking agent capabilities
6. **Safety**: Preventing harmful agent actions
