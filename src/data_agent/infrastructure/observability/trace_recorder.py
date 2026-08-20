from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Protocol

from data_agent.domain.models import AgentRunStatus, NodeTrace, TraceContext, TraceEvent


class TraceRecorder(Protocol):
    """Persistence boundary for Agent run, node span, and decision event records."""

    def start_run(self, context: TraceContext) -> None: ...

    def finish_run(self, context: TraceContext, status: AgentRunStatus) -> None: ...

    def start_node(self, trace: NodeTrace) -> None: ...

    def finish_node(self, trace: NodeTrace) -> None: ...

    def fail_node(self, trace: NodeTrace) -> None: ...

    def record_event(self, event: TraceEvent) -> None: ...


class LoggingTraceRecorder:
    """Stage-one recorder that emits structured JSON without coupling Planner to storage."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("data_agent.trace")

    def start_run(self, context: TraceContext) -> None:
        self._write("agent_run_started", context.model_dump(mode="json"))

    def finish_run(self, context: TraceContext, status: AgentRunStatus) -> None:
        self._write(
            "agent_run_finished",
            {**context.model_dump(mode="json"), "status": status.value},
        )

    def start_node(self, trace: NodeTrace) -> None:
        self._write("agent_node_started", trace.model_dump(mode="json"))

    def finish_node(self, trace: NodeTrace) -> None:
        self._write("agent_node_finished", trace.model_dump(mode="json"))

    def fail_node(self, trace: NodeTrace) -> None:
        self._write("agent_node_failed", trace.model_dump(mode="json"), level=logging.ERROR)

    def record_event(self, event: TraceEvent) -> None:
        self._write("agent_trace_event", event.model_dump(mode="json"))

    def _write(self, event_name: str, payload: dict[str, object], level: int = logging.INFO) -> None:
        self.logger.log(level, "%s %s", event_name, json.dumps(payload, ensure_ascii=False, sort_keys=True))


class InMemoryTraceRecorder:
    """Deterministic recorder used by unit tests and local trace inspection."""

    def __init__(self) -> None:
        self.started_runs: list[TraceContext] = []
        self.finished_runs: list[tuple[TraceContext, AgentRunStatus]] = []
        self.started_nodes: list[NodeTrace] = []
        self.finished_nodes: list[NodeTrace] = []
        self.failed_nodes: list[NodeTrace] = []
        self.events: list[TraceEvent] = []

    def start_run(self, context: TraceContext) -> None:
        self.started_runs.append(context.model_copy(deep=True))

    def finish_run(self, context: TraceContext, status: AgentRunStatus) -> None:
        self.finished_runs.append((context.model_copy(deep=True), status))

    def start_node(self, trace: NodeTrace) -> None:
        self.started_nodes.append(trace.model_copy(deep=True))

    def finish_node(self, trace: NodeTrace) -> None:
        self.finished_nodes.append(trace.model_copy(deep=True))

    def fail_node(self, trace: NodeTrace) -> None:
        self.failed_nodes.append(trace.model_copy(deep=True))

    def record_event(self, event: TraceEvent) -> None:
        self.events.append(event.model_copy(deep=True))


@lru_cache(maxsize=1)
def get_trace_recorder() -> TraceRecorder:
    """Return the configured recorder; later phases can replace this with MySQL/OTel."""
    return LoggingTraceRecorder()
