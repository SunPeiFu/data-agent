from data_agent.hybrid_router import ConflictResolution, HybridQuestionRouter
from data_agent.models import IntentType, LineageDirection
from data_agent.planner import plan_question


def test_data_catalog_search_uses_metadata_filters_and_semantic_recall() -> None:
    result = plan_question("营销域 DWD 层搜索支付明细相关表和表说明")

    assert result.intent == IntentType.METADATA_SEARCH
    assert result.entities.data_layer is not None
    assert result.entities.data_layer.value == "DWD"
    assert "支付" in result.entities.topic_keywords
    assert [(step.tool_name, step.action) for step in result.task_steps] == [
        ("tidb_metadata", "filter_tables"),
        ("milvus_rag", "semantic_search"),
        ("result_ranker", "merge_and_rank"),
    ]


def test_lineage_query_supports_configurable_upstream_downstream_direction() -> None:
    result = plan_question("查询 dwd.orderInfo 表的上游和下游血缘，深度三层")

    assert result.intent == IntentType.LINEAGE_SEARCH
    assert result.entities.lineage_direction == LineageDirection.BOTH
    lineage_step = next(step for step in result.task_steps if step.tool_name == "neo4j_lineage")
    assert lineage_step.params["direction"] == "both"
    assert lineage_step.params["depth"] == 3


def test_table_level_impact_analysis_uses_table_granularity() -> None:
    result = plan_question("dwd.orderInfo 表修改，对下游报表和任务有什么影响")

    assert result.intent == IntentType.IMPACT_ANALYSIS
    lineage_step = next(step for step in result.task_steps if step.tool_name == "neo4j_lineage")
    assert "field_name" not in lineage_step.params
    assert lineage_step.params["lineage_granularity"] == "table"


def test_one_part_table_name_is_candidate_requiring_metadata_resolution() -> None:
    route = HybridQuestionRouter().route("营销域 DWD 层 userInfo 表的下游血缘有哪些")

    assert route.entities.table is not None
    assert route.entities.table.raw == "userInfo"
    assert route.entity_resolution.fields["table"].resolution == ConflictResolution.NEEDS_METADATA_VALIDATION
