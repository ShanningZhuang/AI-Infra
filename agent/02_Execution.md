# Agent Execution Infrastructure

> Parent: [Agent Infrastructure Overview](00_Agent.md)

## Overview

Execution infrastructure handles how agents actually run: tool calling, orchestration, error handling, and state management. This layer bridges the LLM's decisions with real-world actions.

## Learning Objectives

- [ ] Orchestration frameworks (LangChain, LlamaIndex, custom)
- [ ] Tool calling protocols and execution
- [ ] Parallel tool execution
- [ ] Error handling and retry strategies
- [ ] State management and checkpointing

## Resources

### Documentation

- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [LlamaIndex Agents](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use](https://docs.anthropic.com/claude/docs/tool-use)

### Papers

- [Gorilla: Large Language Model Connected with Massive APIs](https://arxiv.org/abs/2305.15334)
- [ToolLLM: Facilitating LLMs to Master 16000+ Real-world APIs](https://arxiv.org/abs/2307.16789)
- [TaskWeaver: A Code-First Agent Framework](https://arxiv.org/abs/2311.17541)

### Code References

- [LangChain](https://github.com/langchain-ai/langchain)
- [LlamaIndex](https://github.com/run-llama/llama_index)
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel)

---

## Notes

### Orchestration Layer

The orchestration layer manages the agent loop:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Orchestrator                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│   │   Message   │────▶│    LLM      │────▶│   Parser    │       │
│   │  Formatter  │     │   Call      │     │  (actions)  │       │
│   └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                  │               │
│                                                  ▼               │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│   │   Memory    │◀────│   State     │◀────│    Tool     │       │
│   │   Update    │     │  Manager    │     │  Executor   │       │
│   └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Basic Orchestrator Implementation

```python
class AgentOrchestrator:
    def __init__(self, llm, tools: dict, memory, max_steps=10):
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.max_steps = max_steps
        self.state = AgentState()

    def run(self, task: str) -> str:
        self.state.reset()
        self.memory.add("user", task)

        for step in range(self.max_steps):
            # 1. Format messages for LLM
            messages = self._format_messages()

            # 2. Get LLM response
            response = self.llm.chat(messages)

            # 3. Parse actions from response
            actions = self._parse_actions(response)

            # 4. Check for completion
            if actions.is_final:
                return actions.final_answer

            # 5. Execute tools
            results = self._execute_tools(actions.tool_calls)

            # 6. Update state and memory
            self._update_state(actions, results)

        return "Max steps reached"

    def _execute_tools(self, tool_calls: list) -> list:
        results = []
        for call in tool_calls:
            try:
                tool = self.tools[call.name]
                result = tool.execute(call.args)
                results.append(ToolResult(success=True, output=result))
            except Exception as e:
                results.append(ToolResult(success=False, error=str(e)))
        return results
```

### Tool Calling Protocols

**OpenAI Function Calling Format:**

```python
# Tool definition
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g., 'San Francisco'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    }
}]

# API call
response = openai.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools,
    tool_choice="auto"  # or "required" or {"type": "function", "function": {"name": "..."}}
)

# Parse tool calls
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        # Execute and return result
```

**Anthropic Tool Use Format:**

```python
tools = [{
    "name": "get_weather",
    "description": "Get current weather for a location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location"]
    }
}]

response = anthropic.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    tools=tools,
    messages=messages
)

# Check for tool use
for block in response.content:
    if block.type == "tool_use":
        tool_name = block.name
        tool_input = block.input
        # Execute tool...
```

### Parallel Tool Execution

When tools are independent, execute them in parallel:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ParallelToolExecutor:
    def __init__(self, tools: dict, max_workers=4):
        self.tools = tools
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def execute_parallel(self, tool_calls: list) -> list:
        """
        Execute multiple tool calls in parallel
        """
        # Analyze dependencies
        independent, dependent = self._analyze_dependencies(tool_calls)

        # Execute independent tools in parallel
        tasks = [
            self._execute_async(call)
            for call in independent
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Execute dependent tools sequentially
        for call in dependent:
            result = await self._execute_async(call)
            results.append(result)

        return results

    async def _execute_async(self, tool_call):
        """Run tool in thread pool to avoid blocking"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.tools[tool_call.name].execute,
            tool_call.args
        )

    def _analyze_dependencies(self, tool_calls):
        """
        Determine which tool calls can run in parallel
        Simple heuristic: if output of one is input to another, dependent
        """
        # Implementation depends on tool semantics
        independent = []
        dependent = []

        used_outputs = set()
        for call in tool_calls:
            # Check if any argument references another tool's output
            if self._references_output(call.args, used_outputs):
                dependent.append(call)
            else:
                independent.append(call)
            used_outputs.add(call.id)

        return independent, dependent
```

**Parallel Execution Diagram:**

```
Sequential (slow):
┌──────┐   ┌──────┐   ┌──────┐
│Tool A│──▶│Tool B│──▶│Tool C│  Total: 3 seconds
└──────┘   └──────┘   └──────┘
   1s         1s         1s

Parallel (fast):
┌──────┐
│Tool A│──┐
└──────┘  │
┌──────┐  ├──▶ Results    Total: 1 second
│Tool B│──┤
└──────┘  │
┌──────┐  │
│Tool C│──┘
└──────┘
```

### Error Handling Strategies

```python
class RobustToolExecutor:
    def __init__(self, tools: dict, max_retries=3, timeout=30):
        self.tools = tools
        self.max_retries = max_retries
        self.timeout = timeout

    def execute_with_retry(self, tool_call) -> ToolResult:
        """
        Execute tool with retry logic and error handling
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # Timeout wrapper
                result = self._execute_with_timeout(tool_call)
                return ToolResult(success=True, output=result)

            except TimeoutError:
                last_error = f"Tool timed out after {self.timeout}s"
                # Don't retry timeouts
                break

            except ValidationError as e:
                # Invalid input - don't retry, fix the input
                return ToolResult(
                    success=False,
                    error=f"Invalid input: {e}",
                    retry=False
                )

            except RateLimitError:
                # Back off and retry
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                last_error = "Rate limited"

            except Exception as e:
                last_error = str(e)
                # Exponential backoff
                time.sleep(0.5 * (2 ** attempt))

        return ToolResult(success=False, error=last_error)

    def _execute_with_timeout(self, tool_call):
        """Execute with timeout using threading"""
        import threading

        result = [None]
        error = [None]

        def run():
            try:
                result[0] = self.tools[tool_call.name].execute(tool_call.args)
            except Exception as e:
                error[0] = e

        thread = threading.Thread(target=run)
        thread.start()
        thread.join(timeout=self.timeout)

        if thread.is_alive():
            raise TimeoutError()
        if error[0]:
            raise error[0]
        return result[0]
```

**Error Recovery Patterns:**

| Error Type | Strategy | Example |
|------------|----------|---------|
| Timeout | Retry with longer timeout | API slow response |
| Rate limit | Exponential backoff | Too many requests |
| Invalid input | Ask LLM to fix | Wrong parameter type |
| Tool not found | Suggest alternatives | Typo in tool name |
| Network error | Retry with backoff | Connection failed |
| Auth error | Fail fast, report | Invalid API key |

### State Management

```python
from dataclasses import dataclass, field
from typing import Any
import pickle

@dataclass
class AgentState:
    """
    Tracks agent execution state for recovery and debugging
    """
    task_id: str = ""
    current_step: int = 0
    messages: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)
    variables: dict = field(default_factory=dict)
    checkpoints: list = field(default_factory=list)

    def save_checkpoint(self):
        """Save current state for recovery"""
        checkpoint = {
            "step": self.current_step,
            "messages": self.messages.copy(),
            "tool_calls": self.tool_calls.copy(),
            "tool_results": self.tool_results.copy(),
            "variables": self.variables.copy(),
        }
        self.checkpoints.append(checkpoint)
        return len(self.checkpoints) - 1

    def restore_checkpoint(self, checkpoint_id: int):
        """Restore state from checkpoint"""
        checkpoint = self.checkpoints[checkpoint_id]
        self.current_step = checkpoint["step"]
        self.messages = checkpoint["messages"].copy()
        self.tool_calls = checkpoint["tool_calls"].copy()
        self.tool_results = checkpoint["tool_results"].copy()
        self.variables = checkpoint["variables"].copy()

    def serialize(self) -> bytes:
        """Serialize state for persistence"""
        return pickle.dumps(self)

    @classmethod
    def deserialize(cls, data: bytes) -> "AgentState":
        """Restore state from bytes"""
        return pickle.loads(data)
```

### Framework Comparison

| Feature | LangChain | LlamaIndex | Custom |
|---------|-----------|------------|--------|
| Ease of use | ★★★★★ | ★★★★☆ | ★★☆☆☆ |
| Flexibility | ★★★☆☆ | ★★★☆☆ | ★★★★★ |
| RAG support | ★★★★☆ | ★★★★★ | ★★☆☆☆ |
| Tool ecosystem | ★★★★★ | ★★★☆☆ | ★☆☆☆☆ |
| Performance | ★★★☆☆ | ★★★☆☆ | ★★★★★ |
| Debugging | ★★★☆☆ | ★★★☆☆ | ★★★★★ |

### LangChain Agent Example

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain import hub

# Define tools
@tool
def search(query: str) -> str:
    """Search the web for information."""
    return web_search(query)

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

tools = [search, calculator]

# Create agent
llm = ChatOpenAI(model="gpt-4")
prompt = hub.pull("hwchase17/openai-tools-agent")
agent = create_openai_tools_agent(llm, tools, prompt)

# Create executor with error handling
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    early_stopping_method="generate",
    handle_parsing_errors=True,
)

# Run
result = executor.invoke({"input": "What is 15% of the population of Tokyo?"})
```

### LlamaIndex Agent Example

```python
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool

# Define tools
def search(query: str) -> str:
    """Search for information"""
    return web_search(query)

def calculator(expression: str) -> str:
    """Calculate math expression"""
    return str(eval(expression))

tools = [
    FunctionTool.from_defaults(fn=search),
    FunctionTool.from_defaults(fn=calculator),
]

# Create agent
llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools(
    tools,
    llm=llm,
    verbose=True,
    max_iterations=10,
)

# Run
response = agent.chat("What is 15% of the population of Tokyo?")
```

### Structured Output Parsing

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class ToolCall(BaseModel):
    name: str = Field(description="Name of the tool to call")
    args: dict = Field(description="Arguments for the tool")

class AgentAction(BaseModel):
    thought: str = Field(description="Agent's reasoning")
    tool_calls: List[ToolCall] = Field(default_factory=list)
    final_answer: Optional[str] = Field(default=None)

class OutputParser:
    def __init__(self, schema: type):
        self.schema = schema

    def parse(self, response: str) -> AgentAction:
        """
        Parse LLM response into structured action
        """
        # Try JSON parsing first
        try:
            data = json.loads(response)
            return self.schema(**data)
        except:
            pass

        # Fall back to regex parsing
        return self._regex_parse(response)

    def _regex_parse(self, response: str) -> AgentAction:
        """Parse using regex patterns"""
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", response, re.DOTALL)
        action_match = re.search(r"Action:\s*(\w+)", response)
        input_match = re.search(r"Action Input:\s*(.+?)(?=Thought:|$)", response, re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else ""

        if action_match and action_match.group(1).lower() == "finish":
            return AgentAction(
                thought=thought,
                final_answer=input_match.group(1).strip() if input_match else ""
            )

        tool_calls = []
        if action_match and input_match:
            tool_calls.append(ToolCall(
                name=action_match.group(1),
                args={"input": input_match.group(1).strip()}
            ))

        return AgentAction(thought=thought, tool_calls=tool_calls)
```

### Execution Tracing

```python
import time
from dataclasses import dataclass
from typing import Any

@dataclass
class TraceEvent:
    timestamp: float
    event_type: str
    data: dict
    duration_ms: float = 0

class ExecutionTracer:
    def __init__(self):
        self.events: list[TraceEvent] = []
        self.start_time = None

    def start(self, event_type: str, data: dict = None):
        """Start timing an event"""
        self.start_time = time.time()
        return TraceContext(self, event_type, data or {})

    def record(self, event_type: str, data: dict, duration_ms: float = 0):
        """Record an event"""
        self.events.append(TraceEvent(
            timestamp=time.time(),
            event_type=event_type,
            data=data,
            duration_ms=duration_ms
        ))

    def get_summary(self) -> dict:
        """Get execution summary"""
        total_time = sum(e.duration_ms for e in self.events)
        by_type = {}
        for e in self.events:
            if e.event_type not in by_type:
                by_type[e.event_type] = {"count": 0, "total_ms": 0}
            by_type[e.event_type]["count"] += 1
            by_type[e.event_type]["total_ms"] += e.duration_ms

        return {
            "total_events": len(self.events),
            "total_time_ms": total_time,
            "by_type": by_type
        }

class TraceContext:
    def __init__(self, tracer: ExecutionTracer, event_type: str, data: dict):
        self.tracer = tracer
        self.event_type = event_type
        self.data = data
        self.start_time = time.time()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        duration_ms = (time.time() - self.start_time) * 1000
        self.tracer.record(self.event_type, self.data, duration_ms)

# Usage
tracer = ExecutionTracer()

with tracer.start("llm_call", {"model": "gpt-4"}):
    response = llm.chat(messages)

with tracer.start("tool_execution", {"tool": "search"}):
    result = search_tool.execute(query)

print(tracer.get_summary())
```

### Common Execution Pitfalls

| Pitfall | Description | Solution |
|---------|-------------|----------|
| Blocking I/O | Sync calls block event loop | Use async/threading |
| No timeout | Tool hangs forever | Always set timeouts |
| State leakage | State persists across runs | Reset state properly |
| Missing errors | Errors swallowed silently | Log all errors |
| Resource leaks | Connections not closed | Use context managers |
| Race conditions | Parallel execution conflicts | Use proper locking |
