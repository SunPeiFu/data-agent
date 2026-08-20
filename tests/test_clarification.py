import pytest

from data_agent.models import (
    ClarificationInputType,
    ClarificationResponse,
    ExtractedEntities,
    IntentType,
    SlotIssue,
    SlotIssueType,
    SlotValidationResult,
    SlotValidationStage,
)
from data_agent.planner import (
    ClarificationProtocolError,
    _all_slot_issues,
    _build_clarification_request,
    plan_question,
    resume_clarification,
)


def test_clarification_prioritizes_table_before_governance_conflict() -> None:
    state = {
        "post_slot_validation": SlotValidationResult(
            stage=SlotValidationStage.POST_METADATA,
            passed=False,
            issues=[
                SlotIssue(
                    slot_name="domain",
                    issue_type=SlotIssueType.CONFLICT,
                    message="主题域冲突。",
                ),
                SlotIssue(
                    slot_name="table",
                    issue_type=SlotIssueType.AMBIGUOUS,
                    message="存在多个候选表。",
                ),
            ],
        )
    }

    issues = _all_slot_issues(state)  # type: ignore[arg-type]

    assert [issue.slot_name for issue in issues] == ["table", "domain"]


def test_clarification_deduplicates_missing_slot_across_validation_stages() -> None:
    state = {
        "pre_slot_validation": SlotValidationResult(
            stage=SlotValidationStage.PRE_METADATA,
            passed=False,
            issues=[SlotIssue(slot_name="table", issue_type=SlotIssueType.MISSING, message="缺少表线索。")],
        ),
        "post_slot_validation": SlotValidationResult(
            stage=SlotValidationStage.POST_METADATA,
            passed=False,
            issues=[SlotIssue(slot_name="table", issue_type=SlotIssueType.MISSING, message="没有可执行表候选。")],
        ),
    }

    issues = _all_slot_issues(state)  # type: ignore[arg-type]

    assert len(issues) == 1
    assert issues[0].message == "没有可执行表候选。"


def test_clarification_card_contains_candidate_profiles_and_pending_count() -> None:
    issues = [
        SlotIssue(slot_name="table", issue_type=SlotIssueType.AMBIGUOUS, message="存在多个候选表。"),
        SlotIssue(slot_name="domain", issue_type=SlotIssueType.CONFLICT, message="主题域冲突。"),
    ]
    state = {
        "metadata_candidates": {"table": ["dwd.orderInfo", "dwd.order_info"]},
        "metadata_candidate_profiles": {
            "dwd.orderInfo": {
                "domain": "交易域",
                "data_layer": "DWD",
                "biz_line": "安逸花",
                "table_comment": "订单明细事实表",
            },
            "dwd.order_info": {
                "domain": "交易域",
                "data_layer": "DWD",
                "biz_line": "安逸花",
                "table_comment": "订单信息宽表",
            },
        },
        "metadata_candidate_evidence": {},
    }

    request = _build_clarification_request(state, issues)  # type: ignore[arg-type]

    assert request.input_type == ClarificationInputType.SINGLE_SELECT
    assert request.pending_issue_count == 1
    assert request.allow_custom_value is True
    assert [option.value for option in request.options] == ["dwd.orderInfo", "dwd.order_info"]
    assert request.options[0].description == "订单明细事实表"
    assert request.options[0].metadata == {
        "domain": "交易域",
        "data_layer": "DWD",
        "biz_line": "安逸花",
    }


def test_planning_result_returns_structured_clarification_card() -> None:
    result = plan_question("订单信息表修改，对下游报表和任务有什么影响")

    assert result.need_clarification is True
    assert result.clarification_request is not None
    assert result.clarification_request.slot_name == "table"
    assert result.clarification_request.input_type == ClarificationInputType.SINGLE_SELECT
    assert len(result.clarification_request.options) == 2
    assert result.clarification_question == result.clarification_request.question
    assert result.task_steps == []


def test_missing_table_returns_free_text_card() -> None:
    request = _build_clarification_request(
        {"entities": ExtractedEntities(), "intent": IntentType.LINEAGE_SEARCH},  # type: ignore[arg-type]
        [SlotIssue(slot_name="table", issue_type=SlotIssueType.MISSING, message="缺少表名。")],
    )

    assert request.input_type == ClarificationInputType.TEXT
    assert request.allow_custom_value is True
    assert request.options == []


def test_resume_clarification_revalidates_selected_table_and_builds_plan() -> None:
    paused = plan_question("订单信息表修改，对下游报表和任务有什么影响")
    request = paused.clarification_request
    assert request is not None
    selected = request.options[0]
    response = ClarificationResponse(
        thread_id=request.thread_id,
        clarification_id=request.clarification_id,
        option_id=selected.option_id,
        value=selected.value,
        state_version=request.state_version,
        idempotency_key=f"resume-{request.clarification_id}",
    )

    resumed = resume_clarification(response)

    assert resumed.need_clarification is False
    assert resumed.handoff_required is False
    assert resumed.entities.table is not None
    assert resumed.entities.table.raw == selected.value
    assert len(resumed.task_steps) == 3
    assert resumed.clarification_history[0].source == "user_confirmed"
    assert resumed.clarification_history[0].confidence == 1.0
    assert "跳过原问题中的表术语候选扩展" in "\n".join(resumed.notes)

    # Replaying a successful request returns the stored result and does not append history twice.
    replayed = resume_clarification(response)
    assert replayed == resumed
    assert len(replayed.clarification_history) == 1


def test_resume_clarification_rejects_stale_state_version() -> None:
    paused = plan_question("订单信息表修改，对下游报表和任务有什么影响")
    request = paused.clarification_request
    assert request is not None
    selected = request.options[0]

    with pytest.raises(ClarificationProtocolError, match="state_version"):
        resume_clarification(
            ClarificationResponse(
                thread_id=request.thread_id,
                clarification_id=request.clarification_id,
                option_id=selected.option_id,
                value=selected.value,
                state_version=request.state_version + 1,
                idempotency_key=f"stale-{request.clarification_id}",
            )
        )


def test_resume_clarification_rejects_forged_option() -> None:
    paused = plan_question("订单信息表修改，对下游报表和任务有什么影响")
    request = paused.clarification_request
    assert request is not None

    with pytest.raises(ClarificationProtocolError, match="option_id"):
        resume_clarification(
            ClarificationResponse(
                thread_id=request.thread_id,
                clarification_id=request.clarification_id,
                option_id="table-forged",
                value="risk.secret_table",
                state_version=request.state_version,
                idempotency_key=f"forged-{request.clarification_id}",
            )
        )


def test_unresolved_answer_reaches_manual_handoff_limit() -> None:
    paused = plan_question("查询下游依赖关系", max_clarification_rounds=1)
    request = paused.clarification_request
    assert request is not None

    result = resume_clarification(
        ClarificationResponse(
            thread_id=request.thread_id,
            clarification_id=request.clarification_id,
            value="not_existing_table_xyz",
            state_version=request.state_version,
            idempotency_key=f"handoff-{request.clarification_id}",
        )
    )

    assert result.need_clarification is False
    assert result.handoff_required is True
    assert result.task_steps == []
    assert result.handoff_reason is not None
    assert len(result.clarification_history) == 1
