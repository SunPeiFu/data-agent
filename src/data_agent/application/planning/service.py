"""Public application use cases for starting and resuming a planning workflow."""

from __future__ import annotations

from uuid import uuid4

from langgraph.types import Command

from data_agent.application.planning.errors import ClarificationProtocolError
from data_agent.application.planning.graph import get_planning_graph
from data_agent.application.planning.nodes import _validate_clarification_response
from data_agent.domain.models import AccessContext, ClarificationResponse, PlanningResult


def plan_question(
    question: str,
    access_context: AccessContext | None = None,
    *,
    thread_id: str | None = None,
    max_clarification_rounds: int = 3,
) -> PlanningResult:
    """Start one planning run under an authenticated access context."""
    app = get_planning_graph()
    workflow_thread_id = thread_id or f"thread-{uuid4().hex}"
    config = {"configurable": {"thread_id": workflow_thread_id}}
    final_state = app.invoke(
        {
            "question": question,
            "thread_id": workflow_thread_id,
            "access_context": access_context
            or AccessContext(user_id="demo-user", roles=["data_admin"], tenant_id="demo"),
            "clarification_round": 0,
            "max_clarification_rounds": max(1, max_clarification_rounds),
            "state_version": 1,
            "clarification_history": [],
            "confirmed_slots": [],
            "processed_idempotency_keys": [],
        },
        config=config,
    )
    return final_state["result"]


def resume_clarification(response: ClarificationResponse) -> PlanningResult:
    """Validate one human answer and resume the corresponding persisted graph thread."""
    app = get_planning_graph()
    config = {"configurable": {"thread_id": response.thread_id}}
    snapshot = app.get_state(config)
    state = snapshot.values
    if not state:
        raise ClarificationProtocolError(f"未找到 thread_id={response.thread_id} 的澄清会话。")

    if response.idempotency_key in state.get("processed_idempotency_keys", []):
        result = state.get("result")
        if result is None:
            raise ClarificationProtocolError("幂等请求已处理，但会话中缺少可返回结果。")
        return result

    result = state.get("result")
    request = result.clarification_request if result else None
    if request is None or "await_clarification_response" not in snapshot.next:
        raise ClarificationProtocolError("当前会话不处于等待澄清状态。")
    _validate_clarification_response(request, response)

    resumed_state = app.invoke(Command(resume=response.model_dump(mode="json")), config=config)
    resumed_result = resumed_state.get("result")
    if resumed_result is None:
        raise ClarificationProtocolError("澄清恢复后没有生成 PlanningResult。")
    return resumed_result
