from data_agent.intelligence.extractor import extract_entities
from data_agent.domain.models import DataLayer, DomainType, IntentType, LineageDirection, OperationType, TableIdentifier
from data_agent.application.planning.service import plan_question


def test_case_1_metadata_search_plan_contains_tidb_and_milvus() -> None:
    result = plan_question("安逸花业务线营销域下DWD层关于支付的表有哪些")

    assert result.intent == IntentType.METADATA_SEARCH
    assert result.entities.biz_line == "安逸花"
    assert result.entities.domain == DomainType.MARKETING
    assert result.entities.data_layer == DataLayer.DWD
    assert "支付" in result.entities.topic_keywords
    assert [(step.tool_name, step.action) for step in result.task_steps] == [
        ("tidb_metadata", "filter_tables"),
        ("milvus_rag", "semantic_search"),
        ("result_ranker", "merge_and_rank"),
    ]


def test_case_2_impact_plan_uses_lineage_search_downstream() -> None:
    result = plan_question("安逸花 dwd 层 dwd.orderInfo 中的字段修改，对下游哪些表产生影响")

    assert result.intent == IntentType.IMPACT_ANALYSIS
    assert result.entities.lineage_direction == LineageDirection.DOWNSTREAM
    assert result.entities.operation == OperationType.MODIFY_FIELD
    assert result.entities.table is not None
    assert result.entities.table.raw == "dwd.orderInfo"
    assert result.task_steps[0].tool_name == "tidb_metadata"
    assert result.task_steps[0].action == "resolve_table"
    assert result.task_steps[1].tool_name == "neo4j_lineage"
    assert result.task_steps[1].action == "lineage_search"
    assert result.task_steps[1].params["direction"] == "downstream"
    assert result.task_steps[2].tool_name == "impact_analyzer"
    assert result.task_steps[2].action == "classify_impact"
    assert result.notes


def test_case_3_impact_plan_uses_both_direction_and_semantic_context() -> None:
    result = plan_question("营销域下的 dwd层的userInfo表修改字段，关联影响的上游和下游表有哪些")

    assert result.intent == IntentType.IMPACT_ANALYSIS
    assert result.entities.domain == DomainType.MARKETING
    assert result.entities.data_layer == DataLayer.DWD
    assert result.entities.lineage_direction == LineageDirection.BOTH
    assert result.entities.table is not None
    assert result.entities.table.raw == "dwd.userInfo"

    actions = [(step.tool_name, step.action) for step in result.task_steps]
    assert ("tidb_metadata", "resolve_table") in actions
    assert ("neo4j_lineage", "lineage_search") in actions
    assert ("milvus_rag", "semantic_search") in actions
    assert ("impact_analyzer", "merge_lineage_and_metadata") in actions
    lineage_step = next(step for step in result.task_steps if step.tool_name == "neo4j_lineage")
    assert lineage_step.params["table"] == "dwd.userInfo"
    assert lineage_step.params["direction"] == "both"


def test_lineage_only_query_maps_to_lineage_search_with_direction() -> None:
    result = plan_question("dwd.orderInfo 的上游依赖有哪些")

    assert result.intent == IntentType.LINEAGE_SEARCH
    assert result.entities.lineage_direction == LineageDirection.UPSTREAM
    assert result.task_steps[1].params["direction"] == "upstream"


def test_table_identifier_supports_three_two_one_part_names() -> None:
    three = TableIdentifier.parse("catalog.dwd.orderInfo")
    two = TableIdentifier.parse("dwd.orderInfo")
    one = TableIdentifier.parse("userInfo")

    assert three.parts_count == 3
    assert three.catalog == "catalog"
    assert three.schema_name == "dwd"
    assert three.table_name == "orderInfo"
    assert two.parts_count == 2
    assert two.schema_name == "dwd"
    assert one.parts_count == 1
    assert one.table_name == "userInfo"


def test_entity_enums_are_normalized() -> None:
    entities = extract_entities("营销域 DWD 层 userInfo 表修改字段")

    assert entities.domain == DomainType.MARKETING
    assert entities.data_layer == DataLayer.DWD
    assert entities.operation == OperationType.MODIFY_FIELD


def test_missing_table_for_lineage_needs_clarification() -> None:
    result = plan_question("查询下游依赖关系")

    assert result.need_clarification is True
    assert result.clarification_question is not None
