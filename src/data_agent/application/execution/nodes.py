"""LangGraph nodes that execute validated TaskStep DAGs and collect observations."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from data_agent.application.planning.state import PlannerState
from data_agent.domain.models import PlanValidationStatus, TraceEvent
from data_agent.domain.plan_validation import compute_plan_hash
from data_agent.infrastructure.observability.trace_recorder import get_trace_recorder
from data_agent.tools.executor import TaskExecutor


def _execute_task_plan(state: PlannerState) -> PlannerState:
    """Execute only a validated plan under the authenticated user's tool permissions."""
    context = state.get("trace_context")
    result = state.get("result")
    access_context = state.get("access_context")
    if context is None or result is None or access_context is None:
        raise RuntimeError("执行任务计划前缺少 trace/result/access_context。")
    validation = state.get("plan_validation")
    if validation is None or validation.status != PlanValidationStatus.APPROVED:
        raise RuntimeError("只有 PlanValidationStatus.APPROVED 的计划允许进入执行器。")
    if validation.plan_hash != compute_plan_hash(result):
        raise RuntimeError("计划在校验后发生变更，拒绝执行；必须重新进行计划校验。")
    execution = TaskExecutor().execute(
        result.task_steps,
        intent=result.intent,
        access_context=access_context,
        entities=result.entities,
        run_id=context.run_id,
    )
    return {"task_execution_result": execution}


def _collect_observations(state: PlannerState) -> PlannerState:
    """Attach structured tool observations and terminal outputs to PlanningResult and Trace."""
    execution = state.get("task_execution_result")
    result = state.get("result")
    context = state.get("trace_context")
    if execution is None or result is None or context is None:
        raise RuntimeError("收集工具结果前缺少 execution/result/trace_context。")
    result.execution_status = execution.status
    result.tool_observations = execution.observations
    if len(execution.terminal_outputs) == 1:
        result.final_output = next(iter(execution.terminal_outputs.values()))
    else:
        result.final_output = execution.terminal_outputs
    result.notes.append(
        f"工具执行: status={execution.status.value}, calls={len(execution.observations)}。"
    )

    events = list(state.get("trace_events", []))
    recorder = get_trace_recorder()
    for observation in execution.observations:
        event = TraceEvent(
            event_id=f"event-{uuid4().hex}",
            trace_id=context.trace_id,
            run_id=context.run_id,
            node_name="execute_task_plan",
            event_type="TOOL_CALL_FINISHED",
            reason_code=observation.status.value,
            attributes={
                "tool_call_id": observation.tool_call_id,
                "step_id": observation.step_id,
                "tool": f"{observation.tool_name}.{observation.action}",
                "attempts": observation.attempts,
                "duration_ms": observation.duration_ms,
                "error_code": observation.error_code,
            },
            created_at=datetime.now(timezone.utc),
        )
        try:
            recorder.record_event(event)
        except Exception:
            pass
        events.append(event)
    return {"result": result, "trace_events": events}


def _route_after_plan_validation(state: PlannerState) -> str:
    """Route approved plans to return/execute and fail every other decision closed."""
    validation = state.get("plan_validation")
    if validation is None:
        return PlanValidationStatus.REJECTED.value
    if validation.status == PlanValidationStatus.APPROVED:
        return "execute" if state.get("execute_requested", False) else "return_plan"
    return validation.status.value


def _return_plan_validation_failure(state: PlannerState) -> PlannerState:
    """Return a non-executable result while preserving structured validation evidence."""
    validation = state.get("plan_validation")
    result = state.get("result")
    if validation is None or result is None:
        raise RuntimeError("返回计划校验失败结果前缺少 validation/result。")

    result.task_steps = []
    result.plan_validation = validation
    result.replan_required = validation.status == PlanValidationStatus.REPLAN_REQUIRED
    result.approval_required = validation.status == PlanValidationStatus.APPROVAL_REQUIRED
    result.need_clarification = validation.status == PlanValidationStatus.CLARIFICATION_REQUIRED
    if result.need_clarification and validation.violations:
        result.clarification_question = validation.violations[0].message

    planner_decision = {
        PlanValidationStatus.FORBIDDEN: "forbidden",
        PlanValidationStatus.REPLAN_REQUIRED: "replan_required",
        PlanValidationStatus.APPROVAL_REQUIRED: "approval_required",
        PlanValidationStatus.CLARIFICATION_REQUIRED: "validation_clarification",
    }.get(validation.status, "validation_failed")
    return {"result": result, "planner_decision": planner_decision}
