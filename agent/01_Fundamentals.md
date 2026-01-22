# Agent Fundamentals

> Parent: [Agent Infrastructure Overview](00_Agent.md)

## Overview

This covers the core concepts of LLM agents: the ReAct pattern, tool use, memory systems, and reasoning frameworks. These fundamentals form the foundation for all agent systems.

## Learning Objectives

- [ ] ReAct pattern and tool use
- [ ] Agent memory systems (short-term, long-term, episodic)
- [ ] Planning and reasoning frameworks (CoT, ToT, GoT)
- [ ] Prompt engineering for agents
- [ ] Tool design principles

## Resources

### Papers

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)
- [Tree of Thoughts: Deliberate Problem Solving with LLMs](https://arxiv.org/abs/2305.10601)
- [Chain-of-Thought Prompting Elicits Reasoning](https://arxiv.org/abs/2201.11903)
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)

### Blogs & Tutorials

- [Lilian Weng: LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)
- [LangChain Agents Concepts](https://python.langchain.com/docs/modules/agents/)
- [OpenAI Cookbook: Agent Examples](https://cookbook.openai.com/)

### Code References

- [LangChain Agents](https://github.com/langchain-ai/langchain)
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)
- [BabyAGI](https://github.com/yoheinakajima/babyagi)

---

## Notes

### The ReAct Pattern

ReAct (Reasoning + Acting) interleaves thinking with actions:

```
Task: "What is the population of the capital of France?"

Thought 1: I need to find the capital of France first.
Action 1: search("capital of France")
Observation 1: The capital of France is Paris.

Thought 2: Now I need to find the population of Paris.
Action 2: search("population of Paris")
Observation 2: Paris has a population of about 2.1 million.

Thought 3: I have the answer.
Action 3: finish("The population of Paris, the capital of France, is about 2.1 million.")
```

**Why ReAct Works:**
- Explicit reasoning traces for debugging
- Grounded in real observations (not hallucinations)
- Naturally handles multi-step tasks
- Easy to understand and extend

### ReAct Implementation

```python
class ReActAgent:
    def __init__(self, llm, tools: dict):
        self.llm = llm
        self.tools = tools
        self.system_prompt = """You are a helpful assistant that solves tasks step by step.

Available tools:
{tool_descriptions}

For each step, respond with:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <input to the tool>

When you have the final answer:
Thought: <your reasoning>
Action: finish
Action Input: <final answer>
"""

    def run(self, task: str, max_steps: int = 10) -> str:
        tool_desc = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])

        messages = [
            {"role": "system", "content": self.system_prompt.format(tool_descriptions=tool_desc)},
            {"role": "user", "content": f"Task: {task}"}
        ]

        for step in range(max_steps):
            # Get LLM response
            response = self.llm.chat(messages)
            messages.append({"role": "assistant", "content": response})

            # Parse action
            thought, action, action_input = self._parse_response(response)

            if action == "finish":
                return action_input

            # Execute tool
            if action not in self.tools:
                observation = f"Error: Tool '{action}' not found."
            else:
                try:
                    observation = self.tools[action].run(action_input)
                except Exception as e:
                    observation = f"Error: {str(e)}"

            messages.append({"role": "user", "content": f"Observation: {observation}"})

        return "Max steps reached without completing the task."

    def _parse_response(self, response: str):
        # Parse Thought/Action/Action Input from response
        lines = response.strip().split("\n")
        thought = action = action_input = ""

        for line in lines:
            if line.startswith("Thought:"):
                thought = line[8:].strip()
            elif line.startswith("Action:"):
                action = line[7:].strip()
            elif line.startswith("Action Input:"):
                action_input = line[13:].strip()

        return thought, action, action_input
```

### Chain-of-Thought (CoT) Prompting

CoT encourages step-by-step reasoning:

```
# Without CoT
Q: If there are 3 cars in the parking lot and 2 more arrive, how many are there?
A: 5

# With CoT
Q: If there are 3 cars in the parking lot and 2 more arrive, how many are there?
A: Let me think step by step.
   - Initially there are 3 cars
   - 2 more cars arrive
   - 3 + 2 = 5
   So there are 5 cars.
```

**CoT Variants:**

| Variant | Description |
|---------|-------------|
| Zero-shot CoT | Add "Let's think step by step" |
| Few-shot CoT | Provide examples with reasoning |
| Self-consistency | Generate multiple chains, vote |
| Auto-CoT | Automatically generate examples |

### Tree of Thoughts (ToT)

Explore multiple reasoning paths:

```
                    Problem
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Path A       Path B       Path C
          │            │            │
       ┌──┴──┐      ┌──┴──┐      ┌──┴──┐
       ▼     ▼      ▼     ▼      ▼     ▼
      A1    A2     B1    B2     C1    C2
       │     │      │     ✓      ✗     │
       ✗     │      │   (best)         │
             ✗    (pruned)           (pruned)
```

```python
def tree_of_thoughts(problem, breadth=3, depth=3):
    """
    Explore multiple solution paths using BFS/DFS
    """
    root = Node(problem)
    queue = [root]

    for d in range(depth):
        next_queue = []
        for node in queue:
            # Generate multiple next steps
            thoughts = llm.generate_thoughts(node.state, n=breadth)

            for thought in thoughts:
                # Evaluate each thought
                score = llm.evaluate_thought(thought)
                child = Node(thought, parent=node, score=score)
                node.children.append(child)
                next_queue.append(child)

        # Keep best candidates (beam search)
        next_queue.sort(key=lambda n: n.score, reverse=True)
        queue = next_queue[:breadth]

    # Return best path
    best_leaf = max(get_all_leaves(root), key=lambda n: n.score)
    return get_path(best_leaf)
```

### Memory Systems

**Short-term Memory (Working Memory):**

```python
class ShortTermMemory:
    """
    Maintains current conversation context
    Limited by model's context window
    """
    def __init__(self, max_tokens=4096):
        self.messages = []
        self.max_tokens = max_tokens

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._truncate_if_needed()

    def _truncate_if_needed(self):
        """Remove oldest messages if over limit"""
        while self._count_tokens() > self.max_tokens:
            # Keep system message, remove oldest user/assistant
            if len(self.messages) > 1:
                self.messages.pop(1)

    def get_context(self) -> list:
        return self.messages
```

**Long-term Memory (Vector Store):**

```python
class LongTermMemory:
    """
    Persistent memory using vector similarity search
    Survives across sessions
    """
    def __init__(self, embedding_model, vector_store):
        self.embedder = embedding_model
        self.store = vector_store

    def store(self, text: str, metadata: dict = None):
        embedding = self.embedder.encode(text)
        self.store.add(embedding, text, metadata)

    def retrieve(self, query: str, k: int = 5) -> list:
        query_embedding = self.embedder.encode(query)
        results = self.store.search(query_embedding, k=k)
        return results

    def forget(self, ids: list):
        """Remove outdated memories"""
        self.store.delete(ids)
```

**Episodic Memory (Experience Replay):**

```python
class EpisodicMemory:
    """
    Stores past experiences for learning
    """
    def __init__(self):
        self.episodes = []

    def record_episode(self, task: str, steps: list, outcome: str, success: bool):
        episode = {
            "task": task,
            "steps": steps,
            "outcome": outcome,
            "success": success,
            "timestamp": datetime.now()
        }
        self.episodes.append(episode)

    def get_similar_experiences(self, task: str, k: int = 3) -> list:
        """Find similar past tasks"""
        # Use semantic similarity to find relevant episodes
        similar = self._semantic_search(task, k)
        return similar

    def get_successful_strategies(self, task_type: str) -> list:
        """Get strategies that worked before"""
        return [e for e in self.episodes
                if e["success"] and self._matches_type(e, task_type)]
```

### Generative Agents Memory (Stanford)

The Generative Agents paper introduced a sophisticated memory architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Memory Stream                                 │
├─────────────────────────────────────────────────────────────────┤
│  All observations stored with:                                   │
│  - Timestamp                                                     │
│  - Importance score (1-10)                                       │
│  - Embedding for retrieval                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Retrieval = f(recency, importance, relevance)                  │
│                                                                  │
│  score = α × recency + β × importance + γ × relevance           │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                    Reflection                                    │
│  Periodically synthesize insights from memories:                │
│  "What are the 3 most salient high-level questions about       │
│   the statements above?"                                        │
│                                                                  │
│  Store reflections as new (higher importance) memories          │
└─────────────────────────────────────────────────────────────────┘
```

### Tool Design Principles

**1. Clear Descriptions:**
```python
# Bad
def search(q):
    """Search"""
    pass

# Good
def search(query: str) -> str:
    """
    Search the web for information.

    Args:
        query: The search query (e.g., "population of Tokyo")

    Returns:
        Relevant search results as text

    Example:
        search("current weather in New York")
        -> "New York: 72°F, partly cloudy"
    """
    pass
```

**2. Atomic Operations:**
```python
# Bad: Does too much
def process_and_save_and_email(data, path, recipient):
    # Hard for LLM to understand when to use
    pass

# Good: Single responsibility
def process_data(data) -> ProcessedData: pass
def save_to_file(data, path) -> str: pass
def send_email(recipient, content) -> bool: pass
```

**3. Structured Inputs/Outputs:**
```python
from pydantic import BaseModel

class WeatherInput(BaseModel):
    location: str
    unit: str = "celsius"

class WeatherOutput(BaseModel):
    temperature: float
    condition: str
    humidity: float

def get_weather(input: WeatherInput) -> WeatherOutput:
    # Type safety and validation
    pass
```

### Prompt Engineering for Agents

**System Prompt Template:**

```
You are {agent_role}.

Your goal is to {objective}.

You have access to the following tools:
{tool_list}

Guidelines:
- Think step by step before taking action
- If unsure, ask for clarification
- Report errors honestly
- Stop when the task is complete

Output format:
Thought: <your reasoning>
Action: <tool_name or "finish">
Action Input: <tool arguments as JSON>
```

**Few-shot Examples:**

```python
examples = """
Example 1:
Task: Find the CEO of Apple
Thought: I need to search for information about Apple's CEO.
Action: web_search
Action Input: {"query": "Apple CEO 2024"}
Observation: Tim Cook is the CEO of Apple Inc.
Thought: I found the answer.
Action: finish
Action Input: {"answer": "Tim Cook is the CEO of Apple."}

Example 2:
Task: Calculate 15% tip on $85.50
Thought: I need to calculate 15% of $85.50.
Action: calculator
Action Input: {"expression": "85.50 * 0.15"}
Observation: 12.825
Thought: I have the answer.
Action: finish
Action Input: {"answer": "A 15% tip on $85.50 is $12.83."}
"""
```

### Reflexion: Learning from Mistakes

```python
class ReflexionAgent:
    def __init__(self, llm, tools, memory):
        self.llm = llm
        self.tools = tools
        self.memory = memory

    def run_with_reflection(self, task: str, max_attempts: int = 3):
        for attempt in range(max_attempts):
            # Try to solve
            trajectory = self.run(task)

            # Evaluate outcome
            success, feedback = self.evaluate(task, trajectory)

            if success:
                # Store successful strategy
                self.memory.store_success(task, trajectory)
                return trajectory

            # Reflect on failure
            reflection = self.reflect(task, trajectory, feedback)
            self.memory.store_reflection(reflection)

            # Use reflection in next attempt
            # (automatically retrieved via memory)

        return None  # Failed after max attempts

    def reflect(self, task, trajectory, feedback) -> str:
        prompt = f"""
Task: {task}
Your attempt: {trajectory}
Feedback: {feedback}

Reflect on what went wrong and how to improve next time.
Be specific about the mistakes and the corrective actions.
"""
        return self.llm.generate(prompt)
```

### Planning Strategies

**1. Upfront Planning (Plan-then-Execute):**
```python
def plan_and_execute(task):
    # Phase 1: Create plan
    plan = llm.generate(f"Create a step-by-step plan for: {task}")
    steps = parse_plan(plan)

    # Phase 2: Execute each step
    results = []
    for step in steps:
        result = execute_step(step)
        results.append(result)

        # Optional: Re-plan if step failed
        if not result.success:
            plan = llm.generate(f"Revise plan. Step failed: {step}")
            steps = parse_plan(plan)

    return results
```

**2. Interleaved Planning (ReAct-style):**
```python
def interleaved_planning(task):
    # Plan and execute are mixed
    while not done:
        thought = llm.think(context)  # Includes planning
        action = llm.decide_action(thought)
        observation = execute(action)
        context += observation
```

**3. Hierarchical Planning:**
```
High-level plan:
1. Research the topic
2. Write outline
3. Write draft
4. Review and edit

Low-level plans (generated as needed):
Research the topic:
  1.1 Search for key concepts
  1.2 Find relevant papers
  1.3 Summarize findings
```

### Common Pitfalls

| Pitfall | Description | Solution |
|---------|-------------|----------|
| Over-planning | Spends too long planning | Limit planning steps |
| Under-planning | Acts without thinking | Require thought before action |
| Tool misuse | Uses wrong tool | Better descriptions, examples |
| Hallucinated tools | Invents non-existent tools | Strict validation |
| Infinite loops | Repeats same action | Loop detection |
| Context confusion | Loses track of state | Explicit state tracking |
