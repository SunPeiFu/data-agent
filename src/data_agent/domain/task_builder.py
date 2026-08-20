from __future__ import annotations

from data_agent.domain.models import (
    ExtractedEntities,
    IntentType,
    LineageDirection,
    MetadataQueryMode,
    PlanningResult,
    TaskStep,
)


def build_task_plan(
    question: str,
    intent: IntentType,
    confidence: float,
    entities: ExtractedEntities,
    metadata_query_mode: MetadataQueryMode | None = None,
) -> PlanningResult:
    steps: list[TaskStep] = []
    notes: list[str] = []
    resolved_metadata_mode: MetadataQueryMode | None = None

    if intent == IntentType.METADATA_SEARCH:
        resolved_metadata_mode = metadata_query_mode or MetadataQueryMode.DISCOVERY
        steps = _metadata_steps(entities, resolved_metadata_mode)
    elif intent == IntentType.LINEAGE_SEARCH:
        steps = _lineage_steps(entities)
    elif intent == IntentType.IMPACT_ANALYSIS:
        steps = _impact_steps(entities)
        if entities.operation:
            notes.append("当前版本聚焦表级数据探查，变更影响分析统一按表级血缘生成计划。")

    return PlanningResult(
        question=question,
        intent=intent,
        confidence=confidence,
        entities=entities,
        metadata_query_mode=resolved_metadata_mode,
        task_steps=steps,
        need_clarification=False,
        clarification_question=None,
        notes=notes,
    )


def _metadata_steps(entities: ExtractedEntities, mode: MetadataQueryMode) -> list[TaskStep]:
    """Build either a set-valued asset search or a canonical-table detail lookup."""
    if mode == MetadataQueryMode.DETAIL:
        return [
            TaskStep(
                step_id=1,
                tool_name="tidb_metadata",
                action="get_table_detail",
                params={
                    "table": _table_raw(entities),
                    "include": ["basic", "business", "ownership"],
                },
            )
        ]

    exact_params = {
        "biz_line": entities.biz_line,
        "domain": _enum_value(entities.domain),
        "data_layer": _enum_value(entities.data_layer),
        "topic_keywords": entities.topic_keywords,
    }
    semantic_query = " ".join(
        item for item in [
            entities.biz_line,
            _enum_value(entities.domain),
            _enum_value(entities.data_layer),
            *entities.topic_keywords,
            "相关表 表说明 业务含义",
        ]
        if item
        # 代码的逻辑 把用户问题里抽取到的业务线、主题域、数仓层级、关键词，拼成一个更适合 Milvus/RAG 语义检索的查询句子
        # *的意思是 如果topic_keywords是一个数组, 则会被拆开成普通的元素
    )
    return [
        TaskStep(
            step_id=1,
            tool_name="tidb_metadata",
            action="filter_tables",
            params={**exact_params, "limit": 20},
            parallel_group="metadata_discovery",
        ),
        TaskStep(
            step_id=2,
            tool_name="milvus_rag",
            action="semantic_search",
            params={
                "query": semantic_query,
                "top_k": 20,
                "biz_line": entities.biz_line,
                "domain": _enum_value(entities.domain),
                "data_layer": _enum_value(entities.data_layer),
            },
            parallel_group="metadata_discovery",
        ),
        TaskStep(
            step_id=3,
            tool_name="result_ranker",
            action="merge_and_rank",
            params={"rank_by": ["exact_match", "semantic_score", "business_desc"]},
            input_bindings={"exact_result": 1, "semantic_result": 2},
            depends_on=[1, 2],
            # step_id是步骤标识 并不是调用顺序  depends_on是依赖的步骤步骤标识 此处是必须等step1和2都执行完 3才可以执行 1和2可以并行执行
            # action是代码工具中的方法 一个工具可以包含很多功能 action即工具中的一个 类比controller是工具 action是不同的method
        ),
    ]

def _lineage_steps(entities: ExtractedEntities) -> list[TaskStep]:
    """Build execution steps after the planner has resolved one canonical table."""
    direction = entities.lineage_direction or LineageDirection.BOTH
    return [
        TaskStep(
            step_id=1,
            tool_name="neo4j_lineage",
            action="lineage_search",
            params={"table": _table_raw(entities), "direction": direction.value, "depth": 3},
        ),
    ]


def _impact_steps(entities: ExtractedEntities) -> list[TaskStep]:
    """Build table-level impact steps using the canonical table from metadata resolution."""
    direction = entities.lineage_direction or LineageDirection.BOTH
    semantic_query = " ".join(
        item
        for item in [
            _table_raw(entities),
            _enum_value(entities.domain),
            _enum_value(entities.data_layer),
            *entities.topic_keywords,
            "表说明 业务影响",
        ]
        if item
    )
    steps = [
        TaskStep(
            step_id=1,
            tool_name="neo4j_lineage",
            action="lineage_search",
            params={
                "table": _table_raw(entities),
                "direction": direction.value,
                "depth": 3,
                "lineage_granularity": "table",
            },
            parallel_group="impact_inputs" if direction == LineageDirection.BOTH else None,
        ),
    ]
    if direction == LineageDirection.BOTH:
        steps.append(
            TaskStep(
                step_id=2,
                tool_name="milvus_rag",
                action="semantic_search",
                params={"query": semantic_query, "top_k": 5},
                input_bindings={},
                parallel_group="impact_inputs",
            )
        )
        steps.append(
            TaskStep(
                step_id=3,
                tool_name="impact_analyzer",
                action="merge_lineage_and_metadata",
                params={"operation": _enum_value(entities.operation), "direction": direction.value},
                input_bindings={"lineage_result": 1, "semantic_result": 2},
                depends_on=[1, 2],
            )
        )
    else:
        steps.append(
            TaskStep(
                step_id=2,
                tool_name="impact_analyzer",
                action="classify_impact",
                params={"operation": _enum_value(entities.operation), "direction": direction.value},
                input_bindings={"lineage_result": 1},
                depends_on=[1],
            )
        )
    return steps


def _enum_value(value: object) -> str | None:
    if value is None:
        return None
    normalized = getattr(value, "value", value)
    return str(normalized)


def _table_raw(entities: ExtractedEntities) -> str | None:
    return entities.table.raw if entities.table else None
