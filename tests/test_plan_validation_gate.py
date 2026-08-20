from __future__ import annotations

import pytest

from data_agent.application.execution.nodes import _execute_task_plan
from data_agent.application.planning.nodes import _validate_task_plan
from data_agent.application.planning.service import plan_question
from data_agent.domain.models import (
    AccessContext,
    ExtractedEntities,
    IntentType,
    LineageDirection,
    PlanValidationResult,
    PlanValidationStatus,
    PlanningResult,
    TableIdentifier,
    TaskStep,
    TraceContext,
)
from data_agent.domain.plan_validation import compute_plan_hash


def _lineage_result(steps: list[TaskStep]) -> PlanningResult:
    entities = ExtractedEntities(
        table=TableIdentifier.parse("dwd.payment_detail"),
        lineage_direction=LineageDirection.DOWNSTREAM,
    )
    return PlanningResult(
        question="查询 dwd.payment_detail 的下游血缘",
        intent=IntentType.LINEAGE_SEARCH,
        confidence=0.9,
        entities=entities,
        task_steps=steps,
    )


def _validation_state(result: PlanningResult) -> dict[str, object]:
    return {
        "result": result,
        "access_context": AccessContext(user_id="admin", roles=["data_admin"]),
        "metadata_candidates": {"table": ["dwd.payment_detail"]},
        "metadata_candidate_profiles": {
            "dwd.payment_detail": {"domain": "营销域", "biz_line": "安逸花"}
        },
    }


def test_validation_returns_structured_violations_for_invalid_plan() -> None:
    result = _lineage_result(
        [
            TaskStep(
                step_id=1,
                tool_name="unregistered_service",
                action="invented_action",
                params={},
                depends_on=[1],
            )
        ]
    )

    output = _validate_task_plan(_validation_state(result))  # type: ignore[arg-type]
    validation = output["plan_validation"]

    assert validation.status == PlanValidationStatus.REPLAN_REQUIRED
    assert validation.passed is False
    assert {item.code for item in validation.violations} >= {
        "PLAN_DAG_CYCLE",
        "PLAN_TOOL_NOT_REGISTERED",
        "PLAN_INTENT_TOOL_CONTRACT",
    }


def test_execution_policy_violation_rejects_plan() -> None:
    result = PlanningResult(
        question="搜索支付相关表",
        intent=IntentType.METADATA_SEARCH,
        confidence=0.9,
        entities=ExtractedEntities(topic_keywords=["支付"]),
        task_steps=[
            TaskStep(
                step_id=1,
                tool_name="milvus_rag",
                action="semantic_search",
                params={"query": "支付", "top_k": 51},
            )
        ],
    )

    output = _validate_task_plan(_validation_state(result))  # type: ignore[arg-type]
    validation = output["plan_validation"]

    assert validation.status == PlanValidationStatus.REJECTED
    assert any(item.code == "PLAN_TOP_K_LIMIT_EXCEEDED" for item in validation.violations)


def test_graph_fails_closed_when_planner_emits_unknown_tool(monkeypatch) -> None:
    def invalid_builder(**kwargs) -> PlanningResult:
        result = _lineage_result(
            [TaskStep(step_id=1, tool_name="unknown", action="unknown", params={})]
        )
        result.question = kwargs["question"]
        result.entities = kwargs["entities"]
        return result

    monkeypatch.setattr("data_agent.application.planning.nodes.build_task_plan", invalid_builder)

    result = plan_question("营销域 DWD 层 dwd.payment_detail 表的下游血缘有哪些")

    assert result.plan_validation is not None
    assert result.plan_validation.status == PlanValidationStatus.REPLAN_REQUIRED
    assert result.replan_required is True
    assert result.task_steps == []
    assert result.execution_status is None


def test_executor_rejects_plan_mutated_after_approval() -> None:
    result = _lineage_result(
        [
            TaskStep(
                step_id=1,
                tool_name="neo4j_lineage",
                action="lineage_search",
                params={"table": "dwd.payment_detail", "direction": "downstream", "depth": 3},
            )
        ]
    )
    result.plan_validation = PlanValidationResult(
        status=PlanValidationStatus.APPROVED,
        passed=True,
        plan_hash=compute_plan_hash(result),
        registry_version="local-tools-v1",
        policy_version="yaml-rbac-v1",
    )
    result.task_steps[0].params["depth"] = 4

    with pytest.raises(RuntimeError, match="计划在校验后发生变更"):
        _execute_task_plan(
            {
                "result": result,
                "plan_validation": result.plan_validation,
                "access_context": AccessContext(user_id="admin", roles=["data_admin"]),
                "trace_context": TraceContext(
                    trace_id="trace-test",
                    run_id="run-test",
                    thread_id="thread-test",
                    planner_version="test",
                    started_at="2026-08-20T00:00:00Z",
                ),
            }  # type: ignore[arg-type]
        )
