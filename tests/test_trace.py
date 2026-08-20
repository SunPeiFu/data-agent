from datetime import datetime, timezone
from uuid import uuid4

import pytest

from data_agent.domain.models import (
    AgentRunStatus,
    ClarificationResponse,
    IntentType,
    NodeRunStatus,
    TraceContext,
)
from data_agent.application.planning.nodes import (
    _resolve_trace_run_status,
    traced_node,
)
from data_agent.application.planning.graph import get_planning_graph
from data_agent.application.planning.service import plan_question, resume_clarification
from data_agent.infrastructure.observability.trace_recorder import InMemoryTraceRecorder


def _trace_context() -> TraceContext:
    return TraceContext(
        trace_id="trace-test",
        run_id="run-test",
        thread_id="thread-test",
        planner_version="test",
        started_at=datetime.now(timezone.utc),
    )


def test_traced_node_records_completed_span_and_state_summary() -> None:
    recorder = InMemoryTraceRecorder()
    wrapped = traced_node(
        "classify_intent",
        lambda state: {"intent": IntentType.METADATA_SEARCH},
        recorder=recorder,
    )

    output = wrapped({"question": "查询表", "trace_context": _trace_context(), "node_traces": []})

    assert output["intent"] == IntentType.METADATA_SEARCH
    assert len(output["node_traces"]) == 1
    trace = output["node_traces"][0]
    assert trace.node_name == "classify_intent"
    assert trace.status == NodeRunStatus.COMPLETED
    assert trace.input_summary == {
        "available_state_fields": ["node_traces", "question", "trace_context"]
    }
    assert trace.output_summary == {"updated_state_fields": ["intent"]}
    assert len(recorder.started_nodes) == 1
    assert len(recorder.finished_nodes) == 1


def test_traced_node_records_failure_and_closes_failed_run() -> None:
    recorder = InMemoryTraceRecorder()

    def fail(_state):
        raise RuntimeError("metadata unavailable")

    wrapped = traced_node("resolve_metadata_candidates", fail, recorder=recorder)

    with pytest.raises(RuntimeError, match="metadata unavailable"):
        wrapped({"trace_context": _trace_context(), "node_traces": []})

    assert len(recorder.failed_nodes) == 1
    assert recorder.failed_nodes[0].status == NodeRunStatus.FAILED
    assert recorder.failed_nodes[0].error_code == "RuntimeError"
    assert recorder.finished_runs[0][1] == AgentRunStatus.FAILED


def test_trace_recorder_failure_does_not_break_business_node() -> None:
    class BrokenRecorder(InMemoryTraceRecorder):
        def start_node(self, trace):
            raise OSError("trace backend unavailable")

        def finish_node(self, trace):
            raise OSError("trace backend unavailable")

    wrapped = traced_node(
        "extract_entities",
        lambda state: {"entities_extracted": True},
        recorder=BrokenRecorder(),
    )

    output = wrapped({"trace_context": _trace_context(), "node_traces": []})

    assert output["entities_extracted"] is True
    assert output["node_traces"][0].status == NodeRunStatus.COMPLETED


def test_planning_result_exposes_entry_trace_and_run_ids() -> None:
    result = plan_question("dwd.orderInfo 的下游血缘有哪些")

    assert result.thread_id is not None and result.thread_id.startswith("thread-")
    assert result.trace_id is not None and result.trace_id.startswith("trace-")
    assert result.run_id is not None and result.run_id.startswith("run-")
    assert result.parent_run_id is None


def test_full_planner_records_every_business_node_span() -> None:
    thread_id = f"thread-trace-full-plan-{uuid4().hex}"
    plan_question("dwd.orderInfo 的下游血缘有哪些", thread_id=thread_id)
    snapshot = get_planning_graph().get_state({"configurable": {"thread_id": thread_id}})
    node_names = [trace.node_name for trace in snapshot.values["node_traces"]]

    assert node_names == [
        "classify_intent",
        "extract_entities",
        "normalize_entities",
        "determine_metadata_query_mode",
        "validate_slots",
        "resolve_metadata_candidates",
        "authorize_context",
        "post_validate_slots",
        "decide_clarification_or_continue",
        "build_task_plan",
        "validate_task_plan",
    ]


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        ("continue", AgentRunStatus.COMPLETED),
        ("clarify", AgentRunStatus.INTERRUPTED),
        ("forbidden", AgentRunStatus.FORBIDDEN),
        ("handoff", AgentRunStatus.HANDOFF),
    ],
)
def test_trace_run_status_matches_planner_branch(decision, expected) -> None:
    assert _resolve_trace_run_status({"planner_decision": decision}) == expected


def test_clarification_resume_creates_linked_run_under_same_thread() -> None:
    paused = plan_question("订单信息表修改，对下游报表和任务有什么影响")
    request = paused.clarification_request
    assert request is not None
    selected = request.options[0]

    resumed = resume_clarification(
        ClarificationResponse(
            thread_id=request.thread_id,
            clarification_id=request.clarification_id,
            option_id=selected.option_id,
            value=selected.value,
            state_version=request.state_version,
            idempotency_key=f"trace-resume-{request.clarification_id}",
        )
    )

    assert resumed.thread_id == paused.thread_id
    assert resumed.run_id != paused.run_id
    assert resumed.trace_id != paused.trace_id
    assert resumed.parent_run_id == paused.run_id
    assert "run_status=completed" in "\n".join(resumed.notes)
