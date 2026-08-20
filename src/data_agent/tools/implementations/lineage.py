"""Executable Neo4j table-lineage tool."""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from data_agent.infrastructure.repositories.neo4j_lineage import Neo4jLineageRepository
from data_agent.tools.contracts import LineageRelation, LineageSearchInput, LineageSearchOutput


def create_lineage_search_tool(repository: Neo4jLineageRepository | None = None) -> StructuredTool:
    repo = repository or Neo4jLineageRepository()

    def lineage_search(**kwargs: object) -> dict[str, object]:
        payload = LineageSearchInput.model_validate(kwargs)
        records = repo.lineage_search(payload.table, payload.direction, payload.depth)
        output = LineageSearchOutput(
            root_table=payload.table,
            direction=payload.direction,
            relations=[LineageRelation(**record.__dict__) for record in records],
        )
        return output.model_dump(mode="json")

    return StructuredTool.from_function(
        func=lineage_search,
        name="neo4j_lineage__lineage_search",
        description="查询一张物理表的上游、下游或双向表级血缘关系。",
        args_schema=LineageSearchInput,
    )
