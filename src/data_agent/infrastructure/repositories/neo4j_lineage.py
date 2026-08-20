"""Neo4j repository for deterministic table-level lineage traversal."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import LiteralString, cast


class Neo4jLineageRepositoryError(RuntimeError):
    """Normalize Neo4j driver, connection, and query failures."""


@dataclass(frozen=True)
class LineageRecord:
    source_table: str
    target_table: str
    depth: int
    direction: str


class Neo4jLineageRepository:
    """Query `(:Table)-[:DEPENDS_ON]->(:Table)` table-level lineage relationships."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> None:
        self.uri = uri or os.getenv("DATA_AGENT_NEO4J_URI", "bolt://127.0.0.1:7687")
        self.user = user or os.getenv("DATA_AGENT_NEO4J_USER", "neo4j")
        self.password = password or os.getenv("DATA_AGENT_NEO4J_PASSWORD", "agent123")
        self.database = database or os.getenv("DATA_AGENT_NEO4J_DATABASE", "neo4j")

    def lineage_search(self, table: str, direction: str, depth: int) -> list[LineageRecord]:
        try:
            from neo4j import GraphDatabase, Query, RoutingControl
        except ModuleNotFoundError as exc:
            raise Neo4jLineageRepositoryError("neo4j driver 未安装。") from exc

        safe_depth = max(1, min(depth, 5))
        directions = ["upstream", "downstream"] if direction == "both" else [direction]
        records: list[LineageRecord] = []
        try:
            with GraphDatabase.driver(self.uri, auth=(self.user, self.password)) as driver:
                for current_direction in directions:
                    query = _lineage_query(current_direction, safe_depth)
                    result = driver.execute_query(
                        Query(cast(LiteralString, query)),
                        table=table,
                        database_=self.database,
                        routing_=RoutingControl.READ,
                    )
                    for row in result.records:
                        records.append(
                            LineageRecord(
                                source_table=row["source_table"],
                                target_table=row["target_table"],
                                depth=int(row["depth"]),
                                direction=current_direction,
                            )
                        )
        except Neo4jLineageRepositoryError:
            raise
        except Exception as exc:
            raise Neo4jLineageRepositoryError(f"Neo4j 血缘查询失败: {exc}") from exc
        return _deduplicate_records(records)


def _lineage_query(direction: str, depth: int) -> str:
    if direction == "upstream":
        return f"""
            MATCH path=(root:Table {{full_table_name: $table}})-[:DEPENDS_ON*1..{depth}]->(related:Table)
            RETURN root.full_table_name AS source_table,
                   related.full_table_name AS target_table,
                   length(path) AS depth
            ORDER BY depth, target_table
        """
    if direction == "downstream":
        return f"""
            MATCH path=(related:Table)-[:DEPENDS_ON*1..{depth}]->(root:Table {{full_table_name: $table}})
            RETURN root.full_table_name AS source_table,
                   related.full_table_name AS target_table,
                   length(path) AS depth
            ORDER BY depth, target_table
        """
    raise Neo4jLineageRepositoryError(f"不支持的血缘方向: {direction}")


def _deduplicate_records(records: list[LineageRecord]) -> list[LineageRecord]:
    unique: dict[tuple[str, str], LineageRecord] = {}
    for record in records:
        key = record.direction, record.target_table
        previous = unique.get(key)
        if previous is None or record.depth < previous.depth:
            unique[key] = record
    return sorted(unique.values(), key=lambda item: (item.direction, item.depth, item.target_table))
