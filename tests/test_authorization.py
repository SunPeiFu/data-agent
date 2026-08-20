from data_agent.infrastructure.security.authorization import YamlAuthorizationProvider
from data_agent.domain.models import AccessContext, DomainType, ExtractedEntities, IntentType, TableIdentifier
from data_agent.application.planning.nodes import _authorize_context
from data_agent.application.planning.service import plan_question


def test_data_analyst_can_read_marketing_lineage() -> None:
    result = plan_question(
        "营销域 DWD 层 dwd.payment_detail 表的下游血缘有哪些",
        access_context=AccessContext(user_id="analyst-1", roles=["data_analyst"]),
    )

    assert result.need_clarification is False
    assert any(step.action == "lineage_search" for step in result.task_steps)
    assert "权限校验: action=lineage:read 通过" in "\n".join(result.notes)


def test_data_analyst_cannot_run_impact_analysis() -> None:
    result = plan_question(
        "营销域 DWD 层 dwd.payment_detail 表修改，对下游任务有什么影响",
        access_context=AccessContext(user_id="analyst-1", roles=["data_analyst"]),
    )

    assert result.task_steps == []
    assert result.need_clarification is False
    assert "AUTH_ACTION_DENIED" in "\n".join(result.notes)


def test_domain_scope_denies_transaction_table() -> None:
    provider = YamlAuthorizationProvider()
    decision = provider.authorize(
        AccessContext(user_id="analyst-1", roles=["data_analyst"]),
        "lineage:read",
        {
            "full_table_name": "dwd.orderInfo",
            "domain": "交易域",
            "biz_line": "安逸花",
        },
    )

    assert decision.allowed is False
    assert decision.reason_code == "AUTH_DOMAIN_DENIED"
    assert decision.policy_id == "role:data_analyst"
    assert decision.audit_id.startswith("auth-")


def test_authorize_context_filters_denied_candidate_and_notes() -> None:
    state = {
        "intent": IntentType.LINEAGE_SEARCH,
        "access_context": AccessContext(user_id="analyst-1", roles=["data_analyst"]),
        "entities": ExtractedEntities(domain=DomainType.MARKETING),
        "metadata_candidates": {"table": ["dwd.payment_detail", "dwd.orderInfo"]},
        "metadata_candidate_profiles": {
            "dwd.payment_detail": {"domain": "营销域", "biz_line": "安逸花"},
            "dwd.orderInfo": {"domain": "交易域", "biz_line": "安逸花"},
        },
        "metadata_candidate_evidence": {},
        "metadata_notes": [
            "元数据候选证据: table=dwd.payment_detail。",
            "元数据候选证据: table=dwd.orderInfo。",
        ],
    }

    authorized = _authorize_context(state)  # type: ignore[arg-type]

    assert authorized["authorized"] is True
    assert authorized["metadata_candidates"] == {"table": ["dwd.payment_detail"]}
    assert authorized["entities"].table == TableIdentifier.parse("dwd.payment_detail")
    assert "dwd.orderInfo" not in "\n".join(authorized["metadata_notes"])
    assert "过滤 1 个无权候选" in "\n".join(authorized["metadata_notes"])


def test_data_admin_can_analyze_any_domain() -> None:
    provider = YamlAuthorizationProvider()
    decision = provider.authorize(
        AccessContext(user_id="admin-1", roles=["data_admin"]),
        "impact:analyze",
        {"full_table_name": "dwd.orderInfo", "domain": "交易域", "biz_line": "安逸花"},
    )

    assert decision.allowed is True
    assert decision.reason_code == "AUTH_ALLOWED"
