# Multi-Agent Systems

> Parent: [Agent Infrastructure Overview](00_Agent.md)

## Overview

Multi-agent systems coordinate multiple LLM agents to solve complex tasks. This covers communication patterns, coordination strategies, and frameworks for building multi-agent applications.

## Learning Objectives

- [ ] Multi-agent architectures (hierarchical, peer-to-peer, swarm)
- [ ] Agent communication protocols
- [ ] Task decomposition and delegation
- [ ] Consensus and conflict resolution
- [ ] Popular frameworks (AutoGen, CrewAI, LangGraph)

## Resources

### Papers

- [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155)
- [MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework](https://arxiv.org/abs/2308.00352)
- [AgentVerse: Facilitating Multi-Agent Collaboration](https://arxiv.org/abs/2308.10848)
- [CAMEL: Communicative Agents for "Mind" Exploration](https://arxiv.org/abs/2303.17760)
- [ChatDev: Communicative Agents for Software Development](https://arxiv.org/abs/2307.07924)

### Code References

- [AutoGen](https://github.com/microsoft/autogen)
- [CrewAI](https://github.com/joaomdmoura/crewAI)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [MetaGPT](https://github.com/geekan/MetaGPT)
- [ChatDev](https://github.com/OpenBMB/ChatDev)

---

## Notes

### Multi-Agent Architectures

```
┌─────────────────────────────────────────────────────────────────┐
│                Multi-Agent Architectures                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Hierarchical:              Peer-to-Peer:        Swarm:         │
│                                                                  │
│      ┌───────┐             ┌───┐   ┌───┐       ┌─┐ ┌─┐ ┌─┐     │
│      │Manager│             │ A │───│ B │       │ │ │ │ │ │     │
│      └───┬───┘             └─┬─┘   └─┬─┘       └┬┘ └┬┘ └┬┘     │
│     ┌────┼────┐              │       │          │   │   │       │
│     ▼    ▼    ▼              └───┬───┘          └───┼───┘       │
│   ┌───┐┌───┐┌───┐            ┌──┴──┐             ┌──┴──┐        │
│   │ A ││ B ││ C │            │  C  │             │Shared│        │
│   └───┘└───┘└───┘            └─────┘             │State │        │
│                                                   └─────┘        │
│  Central control         Equal agents        Emergent behavior  │
│  Clear hierarchy         Direct messaging    Decentralized      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Hierarchical Architecture

```python
class ManagerAgent:
    """
    Top-level agent that delegates to worker agents
    """
    def __init__(self, llm, workers: dict):
        self.llm = llm
        self.workers = workers

    def solve(self, task: str) -> str:
        # 1. Decompose task into subtasks
        subtasks = self._decompose_task(task)

        # 2. Assign subtasks to workers
        assignments = self._assign_workers(subtasks)

        # 3. Execute subtasks (can be parallel)
        results = {}
        for subtask, worker_name in assignments.items():
            worker = self.workers[worker_name]
            results[subtask] = worker.execute(subtask)

        # 4. Synthesize results
        return self._synthesize(task, results)

    def _decompose_task(self, task: str) -> list:
        prompt = f"""
        Break down this task into subtasks:
        Task: {task}

        Available workers: {list(self.workers.keys())}

        Return a list of subtasks, one per line.
        """
        response = self.llm.generate(prompt)
        return [line.strip() for line in response.split("\n") if line.strip()]

    def _assign_workers(self, subtasks: list) -> dict:
        prompt = f"""
        Assign each subtask to the best worker:

        Subtasks: {subtasks}

        Workers and their specialties:
        {self._describe_workers()}

        Return assignments as: subtask -> worker_name
        """
        response = self.llm.generate(prompt)
        return self._parse_assignments(response)

class WorkerAgent:
    """
    Specialized agent for specific tasks
    """
    def __init__(self, llm, tools: list, specialty: str):
        self.llm = llm
        self.tools = tools
        self.specialty = specialty

    def execute(self, subtask: str) -> str:
        # Execute using ReAct or similar pattern
        return self._react_loop(subtask)
```

### Peer-to-Peer Communication

```python
import asyncio
from dataclasses import dataclass
from typing import Optional

@dataclass
class Message:
    sender: str
    receiver: str
    content: str
    msg_type: str = "request"  # request, response, broadcast
    reply_to: Optional[str] = None

class MessageBus:
    """
    Central message bus for agent communication
    """
    def __init__(self):
        self.queues: dict[str, asyncio.Queue] = {}
        self.history: list[Message] = []

    def register(self, agent_id: str):
        self.queues[agent_id] = asyncio.Queue()

    async def send(self, message: Message):
        self.history.append(message)

        if message.msg_type == "broadcast":
            # Send to all agents
            for agent_id, queue in self.queues.items():
                if agent_id != message.sender:
                    await queue.put(message)
        else:
            # Send to specific agent
            if message.receiver in self.queues:
                await self.queues[message.receiver].put(message)

    async def receive(self, agent_id: str, timeout: float = None) -> Message:
        try:
            return await asyncio.wait_for(
                self.queues[agent_id].get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None

class PeerAgent:
    """
    Agent that communicates with peers via message bus
    """
    def __init__(self, agent_id: str, llm, bus: MessageBus):
        self.agent_id = agent_id
        self.llm = llm
        self.bus = bus
        self.bus.register(agent_id)

    async def run(self):
        while True:
            # Check for incoming messages
            message = await self.bus.receive(self.agent_id, timeout=1.0)

            if message:
                response = await self._handle_message(message)
                if response:
                    await self.bus.send(response)

            # Do autonomous work if idle
            await self._do_work()

    async def _handle_message(self, message: Message) -> Optional[Message]:
        if message.msg_type == "request":
            # Process request and send response
            answer = self.llm.generate(f"Respond to: {message.content}")
            return Message(
                sender=self.agent_id,
                receiver=message.sender,
                content=answer,
                msg_type="response",
                reply_to=message.sender
            )
        return None

    async def request_help(self, peer_id: str, question: str) -> str:
        """Ask another agent for help"""
        await self.bus.send(Message(
            sender=self.agent_id,
            receiver=peer_id,
            content=question,
            msg_type="request"
        ))

        # Wait for response
        while True:
            response = await self.bus.receive(self.agent_id)
            if response and response.reply_to == self.agent_id:
                return response.content
```

### Task Decomposition Strategies

**1. Sequential Decomposition:**
```
Task: Write a blog post about AI
  └─▶ Step 1: Research AI trends
      └─▶ Step 2: Create outline
          └─▶ Step 3: Write draft
              └─▶ Step 4: Edit and polish
```

**2. Parallel Decomposition:**
```
Task: Analyze competitor products
  ├─▶ Agent A: Analyze Product X
  ├─▶ Agent B: Analyze Product Y
  └─▶ Agent C: Analyze Product Z
      └─▶ Combine results
```

**3. Recursive Decomposition:**
```
Task: Build a web application
  ├─▶ Frontend
  │     ├─▶ UI Components
  │     └─▶ State Management
  └─▶ Backend
        ├─▶ API Design
        └─▶ Database Schema
```

```python
class TaskDecomposer:
    def __init__(self, llm):
        self.llm = llm

    def decompose(self, task: str, strategy: str = "auto") -> TaskTree:
        if strategy == "sequential":
            return self._sequential_decompose(task)
        elif strategy == "parallel":
            return self._parallel_decompose(task)
        elif strategy == "recursive":
            return self._recursive_decompose(task)
        else:
            # Auto-detect best strategy
            return self._auto_decompose(task)

    def _auto_decompose(self, task: str) -> TaskTree:
        prompt = f"""
        Analyze this task and determine the best decomposition strategy:

        Task: {task}

        Consider:
        1. Are subtasks dependent on each other? → Sequential
        2. Are subtasks independent? → Parallel
        3. Are subtasks hierarchical/nested? → Recursive

        Return:
        - Strategy: sequential/parallel/recursive
        - Subtasks: list of subtasks
        - Dependencies: which subtasks depend on others
        """
        response = self.llm.generate(prompt)
        return self._parse_task_tree(response)
```

### Consensus and Conflict Resolution

When agents disagree, resolve conflicts:

```python
class ConsensusProtocol:
    """
    Resolve conflicts between agent outputs
    """
    def __init__(self, agents: list, llm):
        self.agents = agents
        self.llm = llm

    async def reach_consensus(self, question: str, max_rounds: int = 3) -> str:
        # Collect initial answers
        answers = {}
        for agent in self.agents:
            answers[agent.id] = await agent.answer(question)

        # Check for agreement
        if self._all_agree(answers):
            return list(answers.values())[0]

        # Debate rounds
        for round in range(max_rounds):
            # Share all answers with each agent
            for agent in self.agents:
                other_answers = {k: v for k, v in answers.items() if k != agent.id}
                revised = await agent.revise_given_others(question, other_answers)
                answers[agent.id] = revised

            if self._all_agree(answers):
                return list(answers.values())[0]

        # Final resolution: use judge
        return await self._judge_resolve(question, answers)

    def _all_agree(self, answers: dict) -> bool:
        """Check if all answers are semantically equivalent"""
        values = list(answers.values())
        return all(self._similar(values[0], v) for v in values[1:])

    async def _judge_resolve(self, question: str, answers: dict) -> str:
        """Use a judge LLM to pick the best answer"""
        prompt = f"""
        Question: {question}

        Different agents gave these answers:
        {answers}

        Analyze each answer and select the best one.
        Explain your reasoning, then state the final answer.
        """
        return self.llm.generate(prompt)
```

**Voting Strategies:**

| Strategy | Description | Best For |
|----------|-------------|----------|
| Majority | Most common answer wins | Factual questions |
| Weighted | Expert agents count more | Domain-specific tasks |
| Unanimous | All must agree | Critical decisions |
| Judge | Separate LLM decides | Complex disagreements |

### AutoGen Framework

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Create agents
assistant = AssistantAgent(
    name="assistant",
    llm_config={"model": "gpt-4"},
    system_message="You are a helpful AI assistant."
)

coder = AssistantAgent(
    name="coder",
    llm_config={"model": "gpt-4"},
    system_message="You are a Python expert. Write clean, efficient code."
)

reviewer = AssistantAgent(
    name="reviewer",
    llm_config={"model": "gpt-4"},
    system_message="You review code for bugs and improvements."
)

user_proxy = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding"},
)

# Create group chat
group_chat = GroupChat(
    agents=[user_proxy, assistant, coder, reviewer],
    messages=[],
    max_round=10,
)

manager = GroupChatManager(groupchat=group_chat, llm_config={"model": "gpt-4"})

# Start conversation
user_proxy.initiate_chat(
    manager,
    message="Create a Python function to calculate fibonacci numbers efficiently."
)
```

### CrewAI Framework

```python
from crewai import Agent, Task, Crew, Process

# Define agents with roles
researcher = Agent(
    role="Senior Researcher",
    goal="Research and analyze market trends",
    backstory="Expert at finding and synthesizing information",
    tools=[search_tool, web_scraper],
    llm=llm,
)

analyst = Agent(
    role="Data Analyst",
    goal="Analyze data and create insights",
    backstory="Skilled at statistical analysis and visualization",
    tools=[calculator, chart_tool],
    llm=llm,
)

writer = Agent(
    role="Content Writer",
    goal="Create compelling reports",
    backstory="Expert at technical writing",
    tools=[],
    llm=llm,
)

# Define tasks
research_task = Task(
    description="Research AI market trends for 2024",
    agent=researcher,
    expected_output="Comprehensive research notes"
)

analysis_task = Task(
    description="Analyze the research and identify key patterns",
    agent=analyst,
    expected_output="Data analysis report with charts"
)

writing_task = Task(
    description="Write a market report based on analysis",
    agent=writer,
    expected_output="Professional market report"
)

# Create crew
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, writing_task],
    process=Process.sequential,  # or Process.hierarchical
    verbose=True,
)

# Execute
result = crew.kickoff()
```

### LangGraph for Multi-Agent

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_agent: str

def researcher_node(state: AgentState) -> AgentState:
    """Researcher agent node"""
    messages = state["messages"]
    response = researcher_llm.invoke(messages)
    return {
        "messages": [response],
        "next_agent": "analyzer"
    }

def analyzer_node(state: AgentState) -> AgentState:
    """Analyzer agent node"""
    messages = state["messages"]
    response = analyzer_llm.invoke(messages)
    return {
        "messages": [response],
        "next_agent": "writer"
    }

def writer_node(state: AgentState) -> AgentState:
    """Writer agent node"""
    messages = state["messages"]
    response = writer_llm.invoke(messages)
    return {
        "messages": [response],
        "next_agent": "end"
    }

def router(state: AgentState) -> str:
    """Route to next agent"""
    return state["next_agent"]

# Build graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("researcher", researcher_node)
workflow.add_node("analyzer", analyzer_node)
workflow.add_node("writer", writer_node)

# Add edges
workflow.set_entry_point("researcher")
workflow.add_conditional_edges(
    "researcher",
    router,
    {"analyzer": "analyzer", "end": END}
)
workflow.add_conditional_edges(
    "analyzer",
    router,
    {"writer": "writer", "end": END}
)
workflow.add_conditional_edges(
    "writer",
    router,
    {"end": END}
)

# Compile and run
app = workflow.compile()
result = app.invoke({"messages": [initial_message], "next_agent": ""})
```

### Communication Patterns

```
┌─────────────────────────────────────────────────────────────────┐
│                 Communication Patterns                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Request-Response:        2. Publish-Subscribe:              │
│     A ──request──▶ B            Publisher                       │
│     A ◀──response── B              │                            │
│                                 ┌──┴──┐                         │
│                                 ▼     ▼                         │
│                             Sub A   Sub B                       │
│                                                                  │
│  3. Blackboard:              4. Contract Net:                   │
│     ┌─────────────┐            Manager                          │
│     │  Blackboard │               │                             │
│     │  (shared)   │          ┌────┼────┐                        │
│     └─────────────┘          ▼    ▼    ▼                        │
│       ↑    ↑    ↑          Bid  Bid  Bid                        │
│       A    B    C            │    │    │                        │
│                              └────┼────┘                        │
│                                Winner                           │
└─────────────────────────────────────────────────────────────────┘
```

### Framework Comparison

| Feature | AutoGen | CrewAI | LangGraph | MetaGPT |
|---------|---------|--------|-----------|---------|
| Ease of use | ★★★★☆ | ★★★★★ | ★★★☆☆ | ★★★☆☆ |
| Flexibility | ★★★★★ | ★★★☆☆ | ★★★★★ | ★★★☆☆ |
| Code execution | ★★★★★ | ★★★☆☆ | ★★☆☆☆ | ★★★★★ |
| Workflow control | ★★★☆☆ | ★★★★☆ | ★★★★★ | ★★★★☆ |
| Human-in-loop | ★★★★★ | ★★★☆☆ | ★★★★☆ | ★★☆☆☆ |
| Production ready | ★★★★☆ | ★★★☆☆ | ★★★★☆ | ★★★☆☆ |

### Common Multi-Agent Pitfalls

| Pitfall | Description | Solution |
|---------|-------------|----------|
| Infinite loops | Agents keep delegating | Max iterations, termination conditions |
| Echo chambers | Agents reinforce errors | Diverse prompts, external validation |
| Deadlock | Agents waiting for each other | Timeouts, deadlock detection |
| Token explosion | Long conversation histories | Summarization, context pruning |
| Role confusion | Agents overlap responsibilities | Clear role definitions |
| Coordination overhead | Too much communication | Batch messages, reduce rounds |
