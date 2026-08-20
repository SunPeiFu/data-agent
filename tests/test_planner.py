from data_agent.intelligence.extractor import extract_entities
from data_agent.domain.models import (
    DataLayer,
    DomainType,
    IntentType,
    LineageDirection,
    MetadataQueryMode,
    OperationType,
    TableIdentifier,
)
from data_agent.application.planning.service import plan_question


def test_case_1_metadata_search_plan_contains_tidb_and_milvus() -> None:
    result = plan_question("安逸花业务线营销域下DWD层关于支付的表有哪些")

    assert result.intent == IntentType.METADATA_SEARCH
    assert result.metadata_query_mode == MetadataQueryMode.DISCOVERY
    assert result.entities.biz_line == "安逸花"
    assert result.entities.domain == DomainType.MARKETING
    assert result.entities.data_layer == DataLayer.DWD
    assert "支付" in result.entities.topic_keywords
    assert [(step.tool_name, step.action) for step in result.task_steps] == [
        ("tidb_metadata", "filter_tables"),
        ("milvus_rag", "semantic_search"),
        ("result_ranker", "merge_and_rank"),
    ]
    assert result.task_steps[2].input_bindings == {"exact_result": 1, "semantic_result": 2}
    assert any("discovery 模式跳过" in note for note in result.notes)


def test_metadata_detail_resolves_one_table_then_gets_detail() -> None:
    result = plan_question("查询 dwd.orderInfo 表说明和负责人")

    assert result.intent == IntentType.METADATA_SEARCH
    assert result.metadata_query_mode == MetadataQueryMode.DETAIL
    assert result.entities.table is not None
    assert [(step.tool_name, step.action) for step in result.task_steps] == [
        ("tidb_metadata", "get_table_detail"),
    ]
    assert result.task_steps[0].params["table"] == "dwd.orderInfo"


def test_case_2_impact_plan_uses_lineage_search_downstream() -> None:
    result = plan_question("安逸花 dwd 层 dwd.orderInfo 中的字段修改，对下游哪些表产生影响")

    assert result.intent == IntentType.IMPACT_ANALYSIS
    assert result.entities.lineage_direction == LineageDirection.DOWNSTREAM
    assert result.entities.operation == OperationType.MODIFY_FIELD
    assert result.entities.table is not None
    assert result.entities.table.raw == "dwd.orderInfo"
    assert [(step.tool_name, step.action) for step in result.task_steps] == [
        ("neo4j_lineage", "lineage_search"),
        ("impact_analyzer", "classify_impact"),
    ]
    assert result.task_steps[0].params["direction"] == "downstream"
    assert result.task_steps[0].depends_on == []
    assert result.task_steps[1].depends_on == [1]
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
    assert ("tidb_metadata", "resolve_table") not in actions
    assert ("neo4j_lineage", "lineage_search") in actions
    assert ("milvus_rag", "semantic_search") in actions
    assert ("impact_analyzer", "merge_lineage_and_metadata") in actions
    lineage_step = next(step for step in result.task_steps if step.tool_name == "neo4j_lineage")
    semantic_step = next(step for step in result.task_steps if step.tool_name == "milvus_rag")
    merge_step = next(step for step in result.task_steps if step.tool_name == "impact_analyzer")
    assert lineage_step.params["table"] == "dwd.userInfo"
    assert lineage_step.params["direction"] == "both"
    assert lineage_step.depends_on == []
    assert semantic_step.depends_on == []
    assert lineage_step.parallel_group == semantic_step.parallel_group == "impact_inputs"
    assert merge_step.depends_on == [1, 2]
    assert merge_step.input_bindings == {"lineage_result": 1, "semantic_result": 2}


def test_lineage_only_query_maps_to_lineage_search_with_direction() -> None:
    result = plan_question("dwd.orderInfo 的上游依赖有哪些")

    assert result.intent == IntentType.LINEAGE_SEARCH
    assert result.entities.lineage_direction == LineageDirection.UPSTREAM
    assert len(result.task_steps) == 1
    assert result.task_steps[0].params["direction"] == "upstream"
    assert result.task_steps[0].depends_on == []


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
