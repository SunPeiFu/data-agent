from __future__ import annotations

from data_agent.models import (
    ExtractedEntities,
    IntentType,
    LineageDirection,
    PlanningResult,
    TaskStep,
)


def build_task_plan(question: str, intent: IntentType, confidence: float, entities: ExtractedEntities) -> PlanningResult:
    steps: list[TaskStep] = []
    notes: list[str] = []

    if intent == IntentType.METADATA_SEARCH:
        steps = _metadata_steps(entities)
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
        task_steps=steps,
        need_clarification=False,
        clarification_question=None,
        notes=notes,
    )


def _metadata_steps(entities: ExtractedEntities) -> list[TaskStep]:
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
        TaskStep(step_id=1, tool_name="tidb_metadata", action="filter_tables", params=exact_params),
        TaskStep(
            step_id=2,
            tool_name="milvus_rag",
            action="semantic_search",
            params={"query": semantic_query, "top_k": 20},
        ),
        TaskStep(
            step_id=3,
            tool_name="result_ranker",
            action="merge_and_rank",
            params={"rank_by": ["exact_match", "semantic_score", "business_desc"]},
            depends_on=[1, 2],
            # step_id是步骤标识 并不是调用顺序  depends_on是依赖的步骤步骤标识 此处是必须等step1和2都执行完 3才可以执行 1和2可以并行执行
            # action是代码工具中的方法 一个工具可以包含很多功能 action即工具中的一个 类比controller是工具 action是不同的method
        ),
    ]

#  
def _lineage_steps(entities: ExtractedEntities) -> list[TaskStep]:
    direction = entities.lineage_direction or LineageDirection.BOTH
    return [
        _resolve_table_step(entities, step_id=1),
        TaskStep(
            step_id=2,
            tool_name="neo4j_lineage",
            action="lineage_search",
            params={"table": _table_raw(entities), "direction": direction.value, "depth": 3},
            depends_on=[1],
        ),
    ]


def _impact_steps(entities: ExtractedEntities) -> list[TaskStep]:
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
        _resolve_table_step(entities, step_id=1),
        TaskStep(
            step_id=2,
            tool_name="neo4j_lineage",
            action="lineage_search",
            params={
                "table": _table_raw(entities),
                "direction": direction.value,
                "depth": 3,
                "lineage_granularity": "table",
            },
            depends_on=[1],
            parallel_group="after_table_resolved",
        ),
    ]
    if direction == LineageDirection.BOTH:
        steps.append(
            TaskStep(
                step_id=3,
                tool_name="milvus_rag",
                action="semantic_search",
                params={"query": semantic_query, "top_k": 5},
                depends_on=[1],
                parallel_group="after_table_resolved",
            )
        )
        steps.append(
            TaskStep(
                step_id=4,
                tool_name="impact_analyzer",
                action="merge_lineage_and_metadata",
                params={"operation": _enum_value(entities.operation), "direction": direction.value},
                depends_on=[2, 3],
            )
        )
    else:
        steps.append(
            TaskStep(
                step_id=3,
                tool_name="impact_analyzer",
                action="classify_impact",
                params={"operation": _enum_value(entities.operation), "direction": direction.value},
                depends_on=[2],
            )
        )
    return steps


def _resolve_table_step(entities: ExtractedEntities, step_id: int) -> TaskStep:
    return TaskStep(
        step_id=step_id,
        tool_name="tidb_metadata",
        action="resolve_table",
        params={
            "biz_line": entities.biz_line,
            "domain": _enum_value(entities.domain),
            "data_layer": _enum_value(entities.data_layer),
            "table": _table_raw(entities),
            "table_parts_count": entities.table.parts_count if entities.table else None,
        },
    )


def _enum_value(value: object) -> str | None:
    return getattr(value, "value", value) if value is not None else None


def _table_raw(entities: ExtractedEntities) -> str | None:
    return entities.table.raw if entities.table else None
