# Production Considerations for Agents

> Parent: [Agent Infrastructure Overview](00_Agent.md)

## Overview

Running agents in production requires careful attention to observability, evaluation, safety, cost control, and reliability. This covers the operational aspects of deploying agents at scale.

## Learning Objectives

- [ ] Agent observability and tracing
- [ ] Evaluation and benchmarking
- [ ] Safety guardrails and sandboxing
- [ ] Cost monitoring and optimization
- [ ] Incident response and debugging

## Resources

### Documentation

- [LangSmith](https://docs.smith.langchain.com/)
- [Weights & Biases Prompts](https://docs.wandb.ai/guides/prompts)
- [OpenTelemetry](https://opentelemetry.io/docs/)

### Papers

- [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688)
- [Toolemu: Identifying Risks of LM Agents](https://arxiv.org/abs/2309.15817)
- [Red Teaming Language Models](https://arxiv.org/abs/2202.03286)

---

## Notes

### Observability Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Agent Observability Stack                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │   Tracing   │     │   Metrics   │     │   Logging   │        │
│  │  (Spans)    │     │ (Counters)  │     │  (Events)   │        │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘        │
│         │                   │                   │                │
│         └───────────────────┼───────────────────┘                │
│                             ▼                                    │
│                    ┌─────────────────┐                          │
│                    │   Collector     │                          │
│                    │ (OpenTelemetry) │                          │
│                    └────────┬────────┘                          │
│                             │                                    │
│         ┌───────────────────┼───────────────────┐               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │   Jaeger    │     │ Prometheus  │     │    ELK      │       │
│  │  (Traces)   │     │  (Metrics)  │     │   (Logs)    │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Distributed Tracing for Agents

```python
from opentelemetry import trace
from opentelemetry.trace import SpanKind
from functools import wraps

tracer = trace.get_tracer("agent-service")

class TracedAgent:
    """Agent with built-in tracing"""

    def __init__(self, agent, session_id: str):
        self.agent = agent
        self.session_id = session_id

    async def run(self, task: str) -> str:
        with tracer.start_as_current_span(
            "agent_run",
            kind=SpanKind.SERVER,
            attributes={
                "session_id": self.session_id,
                "task_length": len(task),
            }
        ) as span:
            try:
                result = await self._traced_run(task)
                span.set_attribute("success", True)
                span.set_attribute("result_length", len(result))
                return result
            except Exception as e:
                span.set_attribute("success", False)
                span.set_attribute("error", str(e))
                span.record_exception(e)
                raise

    async def _traced_run(self, task: str) -> str:
        for step in range(self.agent.max_steps):
            with tracer.start_as_current_span(f"step_{step}") as step_span:
                # LLM call
                with tracer.start_as_current_span("llm_call") as llm_span:
                    response = await self.agent.llm.generate(task)
                    llm_span.set_attribute("tokens_used", response.usage.total_tokens)
                    llm_span.set_attribute("model", response.model)

                # Tool execution
                action = self.agent.parse_action(response)
                if action.tool:
                    with tracer.start_as_current_span(
                        "tool_call",
                        attributes={"tool_name": action.tool}
                    ) as tool_span:
                        result = await self.agent.execute_tool(action)
                        tool_span.set_attribute("success", result.success)

                if action.is_final:
                    return action.answer

        return "Max steps reached"

def trace_tool(func):
    """Decorator to trace tool execution"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        with tracer.start_as_current_span(
            f"tool_{func.__name__}",
            attributes={"args": str(args), "kwargs": str(kwargs)}
        ) as span:
            try:
                result = await func(*args, **kwargs)
                span.set_attribute("success", True)
                return result
            except Exception as e:
                span.set_attribute("success", False)
                span.record_exception(e)
                raise
    return wrapper
```

### Key Metrics to Track

```python
from prometheus_client import Counter, Histogram, Gauge

# Counters
agent_runs_total = Counter(
    "agent_runs_total",
    "Total agent runs",
    ["agent_type", "status"]
)

tool_calls_total = Counter(
    "tool_calls_total",
    "Total tool calls",
    ["tool_name", "status"]
)

llm_calls_total = Counter(
    "llm_calls_total",
    "Total LLM API calls",
    ["model", "status"]
)

# Histograms
agent_duration_seconds = Histogram(
    "agent_duration_seconds",
    "Agent execution duration",
    ["agent_type"],
    buckets=[1, 5, 10, 30, 60, 120, 300]
)

llm_latency_seconds = Histogram(
    "llm_latency_seconds",
    "LLM call latency",
    ["model"],
    buckets=[0.1, 0.5, 1, 2, 5, 10]
)

tokens_per_request = Histogram(
    "tokens_per_request",
    "Tokens used per request",
    ["model"],
    buckets=[100, 500, 1000, 2000, 4000, 8000]
)

# Gauges
active_agents = Gauge(
    "active_agents",
    "Currently running agents",
    ["agent_type"]
)

active_sessions = Gauge(
    "active_sessions",
    "Active agent sessions"
)

# Usage
def record_agent_run(agent_type: str, duration: float, success: bool):
    status = "success" if success else "failure"
    agent_runs_total.labels(agent_type=agent_type, status=status).inc()
    agent_duration_seconds.labels(agent_type=agent_type).observe(duration)
```

**Dashboard Metrics:**

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| Success Rate | % of tasks completed | < 90% |
| Avg Steps | Steps per task | > 15 |
| P99 Latency | 99th percentile time | > 60s |
| Token Usage | Tokens per task | > 50k |
| Cost per Task | USD per completion | > $1 |
| Error Rate | Failed tool calls | > 5% |

### Agent Evaluation

```python
from dataclasses import dataclass
from typing import List, Callable
import json

@dataclass
class EvalCase:
    task: str
    expected_output: str
    tools_required: List[str]
    max_steps: int = 10
    timeout_seconds: int = 120

@dataclass
class EvalResult:
    task: str
    success: bool
    output: str
    steps_taken: int
    tools_used: List[str]
    duration_seconds: float
    error: str = None

class AgentEvaluator:
    def __init__(self, agent_factory: Callable):
        self.agent_factory = agent_factory
        self.results: List[EvalResult] = []

    async def evaluate(self, test_cases: List[EvalCase]) -> dict:
        """Run evaluation on test cases"""
        for case in test_cases:
            result = await self._evaluate_case(case)
            self.results.append(result)

        return self._compute_metrics()

    async def _evaluate_case(self, case: EvalCase) -> EvalResult:
        agent = self.agent_factory()
        start_time = time.time()

        try:
            output = await asyncio.wait_for(
                agent.run(case.task),
                timeout=case.timeout_seconds
            )
            duration = time.time() - start_time

            success = self._check_success(output, case.expected_output)

            return EvalResult(
                task=case.task,
                success=success,
                output=output,
                steps_taken=agent.steps_taken,
                tools_used=agent.tools_used,
                duration_seconds=duration
            )

        except asyncio.TimeoutError:
            return EvalResult(
                task=case.task,
                success=False,
                output="",
                steps_taken=agent.steps_taken,
                tools_used=agent.tools_used,
                duration_seconds=case.timeout_seconds,
                error="Timeout"
            )

        except Exception as e:
            return EvalResult(
                task=case.task,
                success=False,
                output="",
                steps_taken=agent.steps_taken if hasattr(agent, 'steps_taken') else 0,
                tools_used=[],
                duration_seconds=time.time() - start_time,
                error=str(e)
            )

    def _check_success(self, output: str, expected: str) -> bool:
        """Check if output matches expected (semantic comparison)"""
        # Simple exact match
        if output.strip() == expected.strip():
            return True

        # LLM-based semantic comparison
        prompt = f"""
        Compare these two answers and determine if they are semantically equivalent:

        Expected: {expected}
        Actual: {output}

        Return only "YES" or "NO".
        """
        response = self.judge_llm.generate(prompt)
        return "YES" in response.upper()

    def _compute_metrics(self) -> dict:
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)

        return {
            "total_cases": total,
            "success_rate": successful / total if total > 0 else 0,
            "avg_steps": sum(r.steps_taken for r in self.results) / total,
            "avg_duration": sum(r.duration_seconds for r in self.results) / total,
            "error_types": self._count_errors(),
            "tool_usage": self._analyze_tool_usage(),
        }
```

### Safety Guardrails

```python
from abc import ABC, abstractmethod
from typing import Optional

class Guardrail(ABC):
    @abstractmethod
    def check_input(self, input_text: str) -> tuple[bool, str]:
        """Check if input is safe. Returns (is_safe, reason)"""
        pass

    @abstractmethod
    def check_output(self, output_text: str) -> tuple[bool, str]:
        """Check if output is safe. Returns (is_safe, reason)"""
        pass

class ContentFilterGuardrail(Guardrail):
    """Filter harmful content"""

    def __init__(self, blocked_patterns: list, moderation_api=None):
        self.blocked_patterns = blocked_patterns
        self.moderation_api = moderation_api

    def check_input(self, input_text: str) -> tuple[bool, str]:
        # Pattern matching
        for pattern in self.blocked_patterns:
            if pattern.search(input_text):
                return False, f"Blocked pattern detected"

        # API-based moderation
        if self.moderation_api:
            result = self.moderation_api.check(input_text)
            if result.flagged:
                return False, f"Content flagged: {result.categories}"

        return True, ""

    def check_output(self, output_text: str) -> tuple[bool, str]:
        return self.check_input(output_text)  # Same checks

class ToolSandboxGuardrail(Guardrail):
    """Restrict tool capabilities"""

    def __init__(self, allowed_tools: set, restricted_actions: dict):
        self.allowed_tools = allowed_tools
        self.restricted_actions = restricted_actions

    def check_tool_call(self, tool_name: str, args: dict) -> tuple[bool, str]:
        if tool_name not in self.allowed_tools:
            return False, f"Tool {tool_name} not allowed"

        if tool_name in self.restricted_actions:
            for key, blocked_values in self.restricted_actions[tool_name].items():
                if key in args and args[key] in blocked_values:
                    return False, f"Action blocked: {key}={args[key]}"

        return True, ""

class RateLimitGuardrail(Guardrail):
    """Prevent resource abuse"""

    def __init__(self, limits: dict):
        self.limits = limits
        self.counters = {}

    def check_rate(self, session_id: str, action: str) -> tuple[bool, str]:
        key = f"{session_id}:{action}"
        current = self.counters.get(key, 0)

        if action in self.limits and current >= self.limits[action]:
            return False, f"Rate limit exceeded for {action}"

        self.counters[key] = current + 1
        return True, ""

class GuardrailChain:
    """Chain multiple guardrails"""

    def __init__(self, guardrails: list[Guardrail]):
        self.guardrails = guardrails

    def check_all(self, input_text: str) -> tuple[bool, list[str]]:
        failures = []
        for guardrail in self.guardrails:
            is_safe, reason = guardrail.check_input(input_text)
            if not is_safe:
                failures.append(reason)

        return len(failures) == 0, failures
```

### Code Execution Sandboxing

```python
import subprocess
import tempfile
import os
from dataclasses import dataclass

@dataclass
class SandboxConfig:
    timeout_seconds: int = 30
    max_memory_mb: int = 512
    allowed_imports: set = None
    network_access: bool = False

class CodeSandbox:
    """Safe code execution environment"""

    def __init__(self, config: SandboxConfig):
        self.config = config

    def execute(self, code: str, language: str = "python") -> dict:
        """Execute code in sandboxed environment"""

        # Validate code
        is_safe, reason = self._validate_code(code)
        if not is_safe:
            return {"success": False, "error": reason}

        # Write code to temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=self._get_suffix(language),
            delete=False
        ) as f:
            f.write(code)
            code_path = f.name

        try:
            # Execute with restrictions
            result = self._run_sandboxed(code_path, language)
            return result
        finally:
            os.unlink(code_path)

    def _validate_code(self, code: str) -> tuple[bool, str]:
        """Static analysis for dangerous patterns"""
        dangerous_patterns = [
            r"import\s+os",
            r"import\s+subprocess",
            r"__import__",
            r"eval\s*\(",
            r"exec\s*\(",
            r"open\s*\(",
            r"file\s*\(",
        ]

        if self.config.allowed_imports:
            # Only allow specific imports
            import_pattern = r"import\s+(\w+)"
            imports = re.findall(import_pattern, code)
            for imp in imports:
                if imp not in self.config.allowed_imports:
                    return False, f"Import not allowed: {imp}"

        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                return False, f"Dangerous pattern detected"

        return True, ""

    def _run_sandboxed(self, code_path: str, language: str) -> dict:
        """Run code with resource limits"""
        cmd = self._build_command(code_path, language)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=self.config.timeout_seconds,
                text=True,
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Execution timeout",
            }

    def _build_command(self, code_path: str, language: str) -> list:
        """Build sandboxed execution command"""
        if language == "python":
            # Use firejail or similar for additional isolation
            return ["python3", code_path]
        elif language == "javascript":
            return ["node", "--max-old-space-size=512", code_path]
        else:
            raise ValueError(f"Unsupported language: {language}")
```

### Cost Monitoring

```python
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class CostConfig:
    # Cost per 1K tokens (example prices)
    input_cost_per_1k: dict = None
    output_cost_per_1k: dict = None
    daily_budget: float = 100.0
    per_session_budget: float = 5.0
    alert_threshold: float = 0.8  # Alert at 80% budget

    def __post_init__(self):
        self.input_cost_per_1k = self.input_cost_per_1k or {
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01,
            "gpt-3.5-turbo": 0.0005,
            "claude-3-opus": 0.015,
            "claude-3-sonnet": 0.003,
        }
        self.output_cost_per_1k = self.output_cost_per_1k or {
            "gpt-4": 0.06,
            "gpt-4-turbo": 0.03,
            "gpt-3.5-turbo": 0.0015,
            "claude-3-opus": 0.075,
            "claude-3-sonnet": 0.015,
        }

class CostTracker:
    def __init__(self, config: CostConfig, alerter=None):
        self.config = config
        self.alerter = alerter
        self.daily_costs = {}
        self.session_costs = {}

    def record_usage(
        self,
        session_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> dict:
        """Record token usage and compute cost"""
        input_cost = (input_tokens / 1000) * self.config.input_cost_per_1k.get(model, 0.01)
        output_cost = (output_tokens / 1000) * self.config.output_cost_per_1k.get(model, 0.02)
        total_cost = input_cost + output_cost

        # Update daily cost
        today = datetime.now().date().isoformat()
        if today not in self.daily_costs:
            self.daily_costs[today] = 0
        self.daily_costs[today] += total_cost

        # Update session cost
        if session_id not in self.session_costs:
            self.session_costs[session_id] = 0
        self.session_costs[session_id] += total_cost

        # Check budgets
        self._check_budgets(session_id, today)

        return {
            "cost": total_cost,
            "session_total": self.session_costs[session_id],
            "daily_total": self.daily_costs[today],
        }

    def _check_budgets(self, session_id: str, today: str):
        """Check and alert on budget thresholds"""
        # Daily budget check
        daily_usage = self.daily_costs[today] / self.config.daily_budget
        if daily_usage >= self.config.alert_threshold:
            self._alert(f"Daily budget at {daily_usage*100:.1f}%")

        if self.daily_costs[today] >= self.config.daily_budget:
            raise BudgetExceededError("Daily budget exceeded")

        # Session budget check
        session_usage = self.session_costs[session_id] / self.config.per_session_budget
        if session_usage >= self.config.alert_threshold:
            self._alert(f"Session {session_id} at {session_usage*100:.1f}% budget")

        if self.session_costs[session_id] >= self.config.per_session_budget:
            raise BudgetExceededError(f"Session {session_id} budget exceeded")

    def get_cost_report(self) -> dict:
        """Generate cost report"""
        return {
            "daily_costs": self.daily_costs,
            "session_costs": self.session_costs,
            "total_cost": sum(self.daily_costs.values()),
            "avg_session_cost": (
                sum(self.session_costs.values()) / len(self.session_costs)
                if self.session_costs else 0
            ),
        }
```

### Incident Response

```python
from enum import Enum
from datetime import datetime

class IncidentSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Incident:
    id: str
    severity: IncidentSeverity
    title: str
    description: str
    session_id: str = None
    trace_id: str = None
    created_at: datetime = None
    resolved_at: datetime = None
    resolution: str = None

class IncidentManager:
    def __init__(self, alerter, logger):
        self.alerter = alerter
        self.logger = logger
        self.incidents = []

    def report_incident(
        self,
        severity: IncidentSeverity,
        title: str,
        description: str,
        context: dict = None
    ) -> Incident:
        """Report and handle an incident"""
        incident = Incident(
            id=str(uuid.uuid4()),
            severity=severity,
            title=title,
            description=description,
            session_id=context.get("session_id") if context else None,
            trace_id=context.get("trace_id") if context else None,
            created_at=datetime.utcnow(),
        )

        self.incidents.append(incident)
        self._handle_incident(incident, context)

        return incident

    def _handle_incident(self, incident: Incident, context: dict):
        """Handle incident based on severity"""
        # Log
        self.logger.error(
            f"Incident: {incident.title}",
            extra={
                "incident_id": incident.id,
                "severity": incident.severity.value,
                "description": incident.description,
                "context": context,
            }
        )

        # Alert based on severity
        if incident.severity == IncidentSeverity.CRITICAL:
            self.alerter.page_oncall(incident)
            self._auto_mitigate(incident, context)

        elif incident.severity == IncidentSeverity.HIGH:
            self.alerter.send_alert(incident)

        elif incident.severity == IncidentSeverity.MEDIUM:
            self.alerter.send_notification(incident)

    def _auto_mitigate(self, incident: Incident, context: dict):
        """Automatic mitigation for critical incidents"""
        # Example: Kill runaway session
        if context and "session_id" in context:
            session_manager.terminate_session(context["session_id"])

        # Example: Disable problematic tool
        if "tool_name" in context:
            tool_registry.disable_tool(context["tool_name"])

    def resolve_incident(self, incident_id: str, resolution: str):
        """Mark incident as resolved"""
        for incident in self.incidents:
            if incident.id == incident_id:
                incident.resolved_at = datetime.utcnow()
                incident.resolution = resolution
                self.logger.info(f"Incident {incident_id} resolved: {resolution}")
                break
```

### Common Production Issues

| Issue | Symptoms | Resolution |
|-------|----------|------------|
| Token explosion | High costs, slow responses | Implement context limits, summarization |
| Tool failures | Cascading errors | Add retries, fallbacks, circuit breakers |
| Infinite loops | Sessions never complete | Max steps, loop detection |
| Memory leaks | Growing memory usage | Proper cleanup, session timeouts |
| Prompt injection | Unexpected behavior | Input sanitization, guardrails |
| Model degradation | Declining quality | Monitor metrics, A/B testing |

### Production Checklist

```
Pre-deployment:
□ All guardrails configured and tested
□ Rate limits and budgets set
□ Monitoring dashboards ready
□ Alerting configured
□ Runbooks documented
□ Rollback plan prepared

Post-deployment:
□ Monitor error rates
□ Check latency percentiles
□ Verify cost tracking
□ Review sample traces
□ Test guardrails in production
□ Validate session persistence
```
