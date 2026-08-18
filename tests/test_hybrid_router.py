from data_agent.hybrid_router import ConflictResolution, PolicyResolver, RulePreAnalysis
from data_agent.llm_analyzer import LLMAnalysis
from data_agent.models import ExtractedEntities, IntentType, LineageDirection, OperationType, TableIdentifier


def test_policy_resolver_strong_rule_intent_overrides_llm_conflict() -> None:
    rule = RulePreAnalysis(
        intent=IntentType.IMPACT_ANALYSIS,
        confidence=0.92,
        entities=ExtractedEntities(
            operation=OperationType.MODIFY_FIELD,
            lineage_direction=LineageDirection.DOWNSTREAM,
        ),
    )
    llm = LLMAnalysis(
        intent=IntentType.METADATA_SEARCH,
        confidence=0.95,
        table_name="dwd.orderInfo",
        topic_keywords=["订单"],
    )

    result = PolicyResolver().resolve(rule=rule, llm=llm)

    assert result.intent == IntentType.IMPACT_ANALYSIS
    assert result.entities.operation == OperationType.MODIFY_FIELD
    assert result.entities.lineage_direction == LineageDirection.DOWNSTREAM
    assert result.entities.table is not None
    assert result.entities.table.raw == "dwd.orderInfo"
    assert result.notes


def test_policy_resolver_uses_rule_when_llm_is_missing() -> None:
    rule = RulePreAnalysis(
        intent=IntentType.LINEAGE_SEARCH,
        confidence=0.88,
        entities=ExtractedEntities(lineage_direction=LineageDirection.UPSTREAM),
    )

    result = PolicyResolver().resolve(rule=rule, llm=None)

    assert result.intent == IntentType.LINEAGE_SEARCH
    assert result.confidence == 0.88
    assert result.entities.lineage_direction == LineageDirection.UPSTREAM
    assert result.notes


def test_policy_resolver_marks_table_as_metadata_validation_candidate() -> None:
    rule = RulePreAnalysis(
        intent=IntentType.IMPACT_ANALYSIS,
        confidence=0.92,
        entities=ExtractedEntities(
            table=TableIdentifier.parse("dwd.orderInfo"),
            operation=OperationType.MODIFY_FIELD,
            lineage_direction=LineageDirection.DOWNSTREAM,
        ),
    )

    result = PolicyResolver().resolve(rule=rule, llm=None)

    assert result.entity_resolution.fields["table"].resolution == ConflictResolution.NEEDS_METADATA_VALIDATION
    assert result.entities.table is not None
    assert result.entities.table.raw == "dwd.orderInfo"


def test_policy_resolver_requests_clarification_for_close_table_candidates() -> None:
    rule = RulePreAnalysis(
        intent=IntentType.LINEAGE_SEARCH,
        confidence=0.88,
        entities=ExtractedEntities(table=TableIdentifier.parse("dwd.orderInfo")),
    )
    llm = LLMAnalysis(
        intent=IntentType.LINEAGE_SEARCH,
        confidence=0.9,
        table_name="dwd.order_info",
    )

    result = PolicyResolver().resolve(rule=rule, llm=llm)

    assert result.entity_resolution.fields["table"].resolution == ConflictResolution.NEEDS_CLARIFICATION
    assert result.entities.table is not None
    assert result.entities.table.raw == "dwd.order_info"
