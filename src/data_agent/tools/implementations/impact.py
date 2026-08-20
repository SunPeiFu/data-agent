"""Deterministic table-level impact classification and evidence fusion tools."""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from data_agent.tools.contracts import (
    ClassifyImpactInput,
    ImpactAnalysisOutput,
    ImpactItem,
    MergeImpactInput,
)


def create_classify_impact_tool() -> StructuredTool:
    def classify_impact(**kwargs: object) -> dict[str, object]:
        payload = ClassifyImpactInput.model_validate(kwargs)
        impacts = _lineage_impacts(payload.lineage_result)
        return ImpactAnalysisOutput(operation=payload.operation, impacts=impacts).model_dump(mode="json")

    return StructuredTool.from_function(
        func=classify_impact,
        name="impact_analyzer__classify_impact",
        description="根据表级血缘深度将变更影响划分为直接影响和间接影响。",
        args_schema=ClassifyImpactInput,
    )


def create_merge_impact_tool() -> StructuredTool:
    def merge_lineage_and_metadata(**kwargs: object) -> dict[str, object]:
        payload = MergeImpactInput.model_validate(kwargs)
        semantic_scores = {
            asset["full_table_name"]: asset.get("semantic_score")
            for asset in payload.semantic_result.get("assets", [])
        }
        impacts = [
            impact.model_copy(update={"semantic_score": semantic_scores.get(impact.table)})
            for impact in _lineage_impacts(payload.lineage_result)
        ]
        degraded = [] if payload.semantic_result else ["semantic_search"]
        return ImpactAnalysisOutput(
            operation=payload.operation,
            impacts=impacts,
            degraded_sources=degraded,
        ).model_dump(mode="json")

    return StructuredTool.from_function(
        func=merge_lineage_and_metadata,
        name="impact_analyzer__merge_lineage_and_metadata",
        description="融合确定性血缘结果和表业务语义，生成双向影响分析结果。",
        args_schema=MergeImpactInput,
    )


def _lineage_impacts(lineage_result: dict[str, object]) -> list[ImpactItem]:
    impacts: list[ImpactItem] = []
    raw_relations = lineage_result.get("relations", [])
    if not isinstance(raw_relations, list):
        return impacts
    for raw in raw_relations:
        if not isinstance(raw, dict):
            continue
        relation = dict(raw)
        depth = int(relation["depth"])
        impacts.append(
            ImpactItem(
                table=str(relation["target_table"]),
                direction=str(relation["direction"]),
                depth=depth,
                impact_level="direct" if depth == 1 else "indirect",
            )
        )
    return sorted(impacts, key=lambda item: (item.depth, item.table))
