# Agent Serving Infrastructure

> Parent: [Agent Infrastructure Overview](00_Agent.md)

## Overview

Serving agents in production requires handling long-running sessions, streaming responses, state persistence, and efficient resource utilization. This layer bridges agent execution with user-facing APIs.

## Learning Objectives

- [ ] Agent session management
- [ ] Streaming responses for agents
- [ ] Long-context handling and compression
- [ ] State persistence strategies
- [ ] API design for agent services

## Resources

### Documentation

- [LangServe](https://python.langchain.com/docs/langserve)
- [FastAPI Streaming](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)
- [Redis Streams](https://redis.io/docs/data-types/streams/)

### Papers

- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
- [LongLoRA: Efficient Fine-tuning of Long-Context Models](https://arxiv.org/abs/2309.12307)

---

## Notes

### Agent Serving Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Serving Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │   Client    │────▶│   Gateway   │────▶│   Router    │        │
│  │   (HTTP)    │     │  (API/WS)   │     │             │        │
│  └─────────────┘     └─────────────┘     └──────┬──────┘        │
│                                                  │               │
│                      ┌───────────────────────────┼───────────┐  │
│                      │                           │           │  │
│                      ▼                           ▼           ▼  │
│               ┌─────────────┐           ┌─────────────┐         │
│               │   Agent     │           │   Agent     │   ...   │
│               │  Worker 1   │           │  Worker 2   │         │
│               └─────────────┘           └─────────────┘         │
│                      │                           │               │
│                      └───────────────────────────┘               │
│                                    │                             │
│                           ┌────────▼────────┐                   │
│                           │  State Store    │                   │
│                           │  (Redis/DB)     │                   │
│                           └─────────────────┘                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Session Management

```python
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import redis

@dataclass
class AgentSession:
    session_id: str
    user_id: str
    agent_type: str
    created_at: datetime
    last_active: datetime
    state: dict = field(default_factory=dict)
    messages: list = field(default_factory=list)
    ttl_seconds: int = 3600  # 1 hour default

class SessionManager:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    def create_session(self, user_id: str, agent_type: str) -> AgentSession:
        session = AgentSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            agent_type=agent_type,
            created_at=datetime.utcnow(),
            last_active=datetime.utcnow(),
        )
        self._save_session(session)
        return session

    def get_session(self, session_id: str) -> Optional[AgentSession]:
        data = self.redis.get(f"session:{session_id}")
        if data:
            return self._deserialize(data)
        return None

    def update_session(self, session: AgentSession):
        session.last_active = datetime.utcnow()
        self._save_session(session)

    def _save_session(self, session: AgentSession):
        key = f"session:{session.session_id}"
        self.redis.setex(
            key,
            session.ttl_seconds,
            self._serialize(session)
        )

    def cleanup_expired(self):
        """Remove expired sessions (handled by Redis TTL)"""
        pass

    def extend_session(self, session_id: str, extra_seconds: int):
        """Extend session TTL"""
        key = f"session:{session_id}"
        self.redis.expire(key, extra_seconds)
```

### Streaming Responses

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import asyncio
import json

app = FastAPI()

async def stream_agent_response(
    agent,
    session: AgentSession,
    user_message: str
) -> AsyncGenerator[str, None]:
    """
    Stream agent responses as Server-Sent Events (SSE)
    """
    # Add user message to session
    session.messages.append({"role": "user", "content": user_message})

    # Stream agent's thinking process
    async for event in agent.stream(session):
        if event.type == "thought":
            yield f"data: {json.dumps({'type': 'thought', 'content': event.content})}\n\n"

        elif event.type == "tool_call":
            yield f"data: {json.dumps({'type': 'tool_call', 'tool': event.tool, 'args': event.args})}\n\n"

        elif event.type == "tool_result":
            yield f"data: {json.dumps({'type': 'tool_result', 'result': event.result})}\n\n"

        elif event.type == "response":
            yield f"data: {json.dumps({'type': 'response', 'content': event.content})}\n\n"

        elif event.type == "done":
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            break

@app.post("/agent/{session_id}/chat")
async def chat(session_id: str, message: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    agent = get_agent_for_session(session)

    return StreamingResponse(
        stream_agent_response(agent, session, message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
```

**Client-side SSE handling:**

```javascript
const eventSource = new EventSource(`/agent/${sessionId}/chat?message=${encodeURIComponent(message)}`);

eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);

    switch(data.type) {
        case 'thought':
            updateThinkingUI(data.content);
            break;
        case 'tool_call':
            showToolCall(data.tool, data.args);
            break;
        case 'tool_result':
            showToolResult(data.result);
            break;
        case 'response':
            appendResponse(data.content);
            break;
        case 'done':
            eventSource.close();
            break;
    }
};
```

### Long-Context Management

```python
class ContextManager:
    """
    Manages conversation context to stay within token limits
    """
    def __init__(self, max_tokens: int = 8000, reserve_tokens: int = 2000):
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens  # For response generation

    def prepare_context(self, messages: list, system_prompt: str) -> list:
        """
        Prepare messages to fit within token budget
        """
        budget = self.max_tokens - self.reserve_tokens
        system_tokens = self._count_tokens(system_prompt)
        available = budget - system_tokens

        # Strategy 1: Keep recent messages
        result = []
        total_tokens = 0

        for message in reversed(messages):
            msg_tokens = self._count_tokens(message["content"])
            if total_tokens + msg_tokens <= available:
                result.insert(0, message)
                total_tokens += msg_tokens
            else:
                # Summarize older messages
                if len(result) > 0:
                    summary = self._summarize_dropped(messages[:len(messages) - len(result)])
                    result.insert(0, {"role": "system", "content": f"Earlier context: {summary}"})
                break

        return result

    def _summarize_dropped(self, messages: list) -> str:
        """Summarize messages that don't fit"""
        if not messages:
            return ""

        prompt = f"""
        Summarize this conversation history concisely:
        {messages}

        Focus on: key decisions, important facts, user preferences
        """
        return self.summarizer_llm.generate(prompt)

    def _count_tokens(self, text: str) -> int:
        # Use tiktoken or similar
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4")
        return len(enc.encode(text))
```

**Context compression strategies:**

```
┌─────────────────────────────────────────────────────────────────┐
│               Context Compression Strategies                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Sliding Window:           2. Summarization:                 │
│     Keep last N messages         Summarize older messages       │
│     [M1, M2, M3, M4, M5]        [Summary] + [M4, M5]            │
│     → [M3, M4, M5]                                              │
│                                                                  │
│  3. Importance Sampling:      4. Hierarchical:                  │
│     Keep important messages      Summary of summaries           │
│     [M1_important, M3, M5]      [Day1_sum, Day2_sum] + recent  │
│                                                                  │
│  5. Retrieval-Augmented:                                        │
│     Store all in vector DB, retrieve relevant                   │
│     Query → [Relevant chunks] + recent                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### State Persistence

```python
from abc import ABC, abstractmethod
import json
import pickle

class StateStore(ABC):
    @abstractmethod
    def save(self, key: str, state: dict) -> None:
        pass

    @abstractmethod
    def load(self, key: str) -> dict:
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        pass

class RedisStateStore(StateStore):
    def __init__(self, redis_client, prefix: str = "agent_state"):
        self.redis = redis_client
        self.prefix = prefix

    def save(self, key: str, state: dict, ttl: int = None) -> None:
        full_key = f"{self.prefix}:{key}"
        data = json.dumps(state)
        if ttl:
            self.redis.setex(full_key, ttl, data)
        else:
            self.redis.set(full_key, data)

    def load(self, key: str) -> dict:
        full_key = f"{self.prefix}:{key}"
        data = self.redis.get(full_key)
        if data:
            return json.loads(data)
        return {}

    def delete(self, key: str) -> None:
        full_key = f"{self.prefix}:{key}"
        self.redis.delete(full_key)

class PostgresStateStore(StateStore):
    """For long-term persistence with query capabilities"""
    def __init__(self, connection_pool):
        self.pool = connection_pool

    def save(self, key: str, state: dict) -> None:
        with self.pool.connection() as conn:
            conn.execute("""
                INSERT INTO agent_states (key, state, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (key) DO UPDATE SET state = %s, updated_at = NOW()
            """, (key, json.dumps(state), json.dumps(state)))

    def load(self, key: str) -> dict:
        with self.pool.connection() as conn:
            result = conn.execute(
                "SELECT state FROM agent_states WHERE key = %s",
                (key,)
            ).fetchone()
            if result:
                return json.loads(result[0])
            return {}
```

### API Design for Agents

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI()

# Request/Response Models
class CreateSessionRequest(BaseModel):
    agent_type: str
    initial_context: Optional[dict] = None

class ChatRequest(BaseModel):
    message: str
    stream: bool = True

class ChatResponse(BaseModel):
    response: str
    thought_process: Optional[List[str]] = None
    tool_calls: Optional[List[dict]] = None

class SessionInfo(BaseModel):
    session_id: str
    agent_type: str
    created_at: str
    message_count: int

# Endpoints
@app.post("/sessions", response_model=SessionInfo)
async def create_session(request: CreateSessionRequest):
    """Create a new agent session"""
    session = session_manager.create_session(
        user_id=get_current_user(),
        agent_type=request.agent_type
    )
    if request.initial_context:
        session.state.update(request.initial_context)
        session_manager.update_session(session)

    return SessionInfo(
        session_id=session.session_id,
        agent_type=session.agent_type,
        created_at=session.created_at.isoformat(),
        message_count=len(session.messages)
    )

@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get session info"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionInfo(...)

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """End an agent session"""
    session_manager.delete_session(session_id)
    return {"status": "deleted"}

@app.post("/sessions/{session_id}/chat")
async def chat(session_id: str, request: ChatRequest):
    """Send message to agent"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if request.stream:
        return StreamingResponse(
            stream_agent_response(agent, session, request.message),
            media_type="text/event-stream"
        )
    else:
        response = await agent.run(session, request.message)
        return ChatResponse(response=response)

@app.post("/sessions/{session_id}/cancel")
async def cancel(session_id: str):
    """Cancel ongoing agent execution"""
    agent_executor.cancel(session_id)
    return {"status": "cancelled"}

@app.get("/sessions/{session_id}/history")
async def get_history(session_id: str, limit: int = 50):
    """Get conversation history"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"messages": session.messages[-limit:]}
```

### WebSocket for Real-time Agents

```python
from fastapi import WebSocket, WebSocketDisconnect
import asyncio

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_event(self, session_id: str, event: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(event)

manager = ConnectionManager()

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(session_id, websocket)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()

            if data["type"] == "chat":
                # Run agent in background, stream events
                session = session_manager.get_session(session_id)
                agent = get_agent_for_session(session)

                async for event in agent.stream(session, data["message"]):
                    await manager.send_event(session_id, {
                        "type": event.type,
                        "data": event.data
                    })

            elif data["type"] == "cancel":
                agent_executor.cancel(session_id)
                await manager.send_event(session_id, {"type": "cancelled"})

    except WebSocketDisconnect:
        manager.disconnect(session_id)
```

### Load Balancing Strategies

```
┌─────────────────────────────────────────────────────────────────┐
│                 Load Balancing for Agents                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Session Affinity (Sticky):                                  │
│     Same session → Same worker                                   │
│     Pros: No state sync needed                                   │
│     Cons: Uneven load distribution                               │
│                                                                  │
│  2. Round-Robin with Shared State:                              │
│     Any worker can handle any request                            │
│     State stored in Redis/DB                                     │
│     Pros: Even distribution                                      │
│     Cons: State access latency                                   │
│                                                                  │
│  3. Least Connections:                                          │
│     Route to worker with fewest active agents                    │
│     Good for long-running agents                                 │
│                                                                  │
│  4. Capability-Based:                                           │
│     Route based on agent type and worker capabilities            │
│     Worker A: GPT-4 agents                                       │
│     Worker B: Claude agents                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Resource Management

```python
import asyncio
from dataclasses import dataclass

@dataclass
class ResourceLimits:
    max_concurrent_agents: int = 100
    max_tokens_per_session: int = 100000
    max_tool_calls_per_minute: int = 60
    max_session_duration_seconds: int = 3600

class ResourceManager:
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.active_agents = 0
        self.semaphore = asyncio.Semaphore(limits.max_concurrent_agents)
        self.session_usage: dict[str, dict] = {}

    async def acquire_agent_slot(self, session_id: str) -> bool:
        """Try to acquire a slot for running an agent"""
        acquired = await asyncio.wait_for(
            self.semaphore.acquire(),
            timeout=30.0
        )
        if acquired:
            self.active_agents += 1
            self._init_session_usage(session_id)
        return acquired

    def release_agent_slot(self, session_id: str):
        """Release agent slot"""
        self.semaphore.release()
        self.active_agents -= 1

    def check_token_budget(self, session_id: str, tokens: int) -> bool:
        """Check if session has token budget"""
        usage = self.session_usage.get(session_id, {})
        total = usage.get("tokens", 0) + tokens
        return total <= self.limits.max_tokens_per_session

    def record_usage(self, session_id: str, tokens: int, tool_calls: int):
        """Record usage for a session"""
        if session_id not in self.session_usage:
            self._init_session_usage(session_id)

        self.session_usage[session_id]["tokens"] += tokens
        self.session_usage[session_id]["tool_calls"] += tool_calls

    def _init_session_usage(self, session_id: str):
        self.session_usage[session_id] = {
            "tokens": 0,
            "tool_calls": 0,
            "start_time": time.time()
        }
```

### Latency Optimization

| Technique | Description | Latency Reduction |
|-----------|-------------|-------------------|
| Connection pooling | Reuse LLM connections | 50-100ms |
| Prompt caching | Cache system prompts | 100-200ms |
| Speculative execution | Pre-compute likely paths | Variable |
| Response streaming | Send partial results | Perceived latency |
| Tool parallelization | Run independent tools together | Variable |
| Model warm-up | Keep models loaded | 500ms-2s |

### Health Checks

```python
from fastapi import FastAPI
from datetime import datetime, timedelta

@app.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/health/detailed")
async def detailed_health():
    """Detailed health with dependencies"""
    checks = {}

    # Check Redis
    try:
        redis_client.ping()
        checks["redis"] = {"status": "healthy"}
    except Exception as e:
        checks["redis"] = {"status": "unhealthy", "error": str(e)}

    # Check LLM endpoint
    try:
        response = await llm.simple_completion("test")
        checks["llm"] = {"status": "healthy"}
    except Exception as e:
        checks["llm"] = {"status": "unhealthy", "error": str(e)}

    # Check active sessions
    checks["sessions"] = {
        "active": session_manager.count_active(),
        "max": resource_limits.max_concurrent_agents
    }

    overall = "healthy" if all(
        c.get("status") == "healthy" for c in checks.values()
        if isinstance(c, dict) and "status" in c
    ) else "degraded"

    return {"status": overall, "checks": checks}
```
