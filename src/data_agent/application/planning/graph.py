"""LangGraph topology for the Data Agent planning workflow."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langgraph.graph import END, StateGraph

from data_agent.application.planning.nodes import (
    _attach_trace,
    _authorize_context,
    _await_clarification_response,
    _build_task_plan,
    _classify_intent,
    _decide_clarification_or_continue,
    _extract_entities_node,
    _init_trace_context,
    _normalize_entities,
    _post_validate_slots,
    _resolve_metadata_candidates,
    _return_clarification_result,
    _return_forbidden_result,
    _return_handoff_result,
    _return_planning_result,
    _route_after_clarification_decision,
    _route_after_trace,
    _validate_slots,
    _validate_task_plan,
    traced_node,
)
from data_agent.application.planning.state import PlannerState
from data_agent.infrastructure.persistence.checkpointing import get_planner_checkpointer


def create_planning_graph(checkpointer: Any | None = None) -> Any:
    """Build the workflow topology while keeping node implementation outside graph wiring."""
    graph = StateGraph(PlannerState)

    graph.add_node("init_trace_context", _init_trace_context)
    graph.add_node("classify_intent", traced_node("classify_intent", _classify_intent))
    graph.add_node("extract_entities", traced_node("extract_entities", _extract_entities_node))
    graph.add_node("normalize_entities", traced_node("normalize_entities", _normalize_entities))
    graph.add_node("validate_slots", traced_node("validate_slots", _validate_slots))
    graph.add_node(
        "resolve_metadata_candidates",
        traced_node("resolve_metadata_candidates", _resolve_metadata_candidates),
    )
    graph.add_node("authorize_context", traced_node("authorize_context", _authorize_context))
    graph.add_node("post_validate_slots", traced_node("post_validate_slots", _post_validate_slots))
    graph.add_node(
        "decide_clarification_or_continue",
        traced_node("decide_clarification_or_continue", _decide_clarification_or_continue),
    )
    graph.add_node("build_task_plan", traced_node("build_task_plan", _build_task_plan))
    graph.add_node(
        "return_clarification_result",
        traced_node("return_clarification_result", _return_clarification_result),
    )
    graph.add_node("await_clarification_response", _await_clarification_response)
    graph.add_node(
        "return_forbidden_result",
        traced_node("return_forbidden_result", _return_forbidden_result),
    )
    graph.add_node(
        "return_handoff_result",
        traced_node("return_handoff_result", _return_handoff_result),
    )
    graph.add_node("validate_task_plan", traced_node("validate_task_plan", _validate_task_plan))
    graph.add_node("attach_trace", _attach_trace)
    graph.add_node("return_planning_result", _return_planning_result)

    graph.set_entry_point("init_trace_context")
    graph.add_edge("init_trace_context", "classify_intent")
    graph.add_edge("classify_intent", "extract_entities")
    graph.add_edge("extract_entities", "normalize_entities")
    graph.add_edge("normalize_entities", "validate_slots")
    graph.add_edge("validate_slots", "resolve_metadata_candidates")
    graph.add_edge("resolve_metadata_candidates", "authorize_context")
    graph.add_edge("authorize_context", "post_validate_slots")
    graph.add_edge("post_validate_slots", "decide_clarification_or_continue")
    graph.add_conditional_edges(
        "decide_clarification_or_continue",
        _route_after_clarification_decision,
        {
            "continue": "build_task_plan",
            "clarify": "return_clarification_result",
            "forbidden": "return_forbidden_result",
            "handoff": "return_handoff_result",
        },
    )
    graph.add_edge("build_task_plan", "validate_task_plan")
    graph.add_edge("validate_task_plan", "attach_trace")
    graph.add_edge("return_clarification_result", "attach_trace")
    graph.add_edge("return_forbidden_result", "attach_trace")
    graph.add_edge("return_handoff_result", "attach_trace")
    graph.add_conditional_edges(
        "attach_trace",
        _route_after_trace,
        {
            "await_clarification": "await_clarification_response",
            "return": "return_planning_result",
        },
    )
    graph.add_edge("await_clarification_response", "normalize_entities")
    graph.add_edge("return_planning_result", END)
    return graph.compile(checkpointer=checkpointer or get_planner_checkpointer())


@lru_cache(maxsize=1)
def get_planning_graph() -> Any:
    """Return one compiled graph backed by the configured durable checkpointer."""
    return create_planning_graph(checkpointer=get_planner_checkpointer())
