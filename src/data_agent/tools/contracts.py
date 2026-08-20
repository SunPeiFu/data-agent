"""Pydantic contracts shared by tool implementations and the task executor."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TableAsset(BaseModel):
    full_table_name: str
    catalog_name: str | None = None
    db_name: str
    table_name: str
    table_comment: str | None = None
    biz_line: str | None = None
    domain: str | None = None
    data_layer: str | None = None
    owner: str | None = None
    exact_score: float | None = None
    semantic_score: float | None = None


class FilterTablesInput(BaseModel):
    biz_line: str | None = None
    domain: str | None = None
    data_layer: str | None = None
    topic_keywords: list[str] = Field(default_factory=list)
    limit: int = Field(default=20, ge=1, le=100)


class FilterTablesOutput(BaseModel):
    assets: list[TableAsset] = Field(default_factory=list)
    source: Literal["mysql"] = "mysql"


class GetTableDetailInput(BaseModel):
    table: str
    include: list[Literal["basic", "business", "ownership"]] = Field(default_factory=lambda: ["basic"])


class GetTableDetailOutput(BaseModel):
    asset: TableAsset
    included_sections: list[str]


class SemanticSearchInput(BaseModel):
    query: str
    top_k: int = Field(default=20, ge=1, le=50)
    biz_line: str | None = None
    domain: str | None = None
    data_layer: str | None = None


class SemanticSearchOutput(BaseModel):
    assets: list[TableAsset] = Field(default_factory=list)
    retrieval_mode: str
    source: Literal["milvus_mysql_validated"] = "milvus_mysql_validated"


class MergeRankInput(BaseModel):
    rank_by: list[str] = Field(default_factory=list)
    exact_result: dict[str, Any] = Field(default_factory=dict)
    semantic_result: dict[str, Any] = Field(default_factory=dict)


class MergeRankOutput(BaseModel):
    assets: list[TableAsset] = Field(default_factory=list)
    total: int


class LineageSearchInput(BaseModel):
    table: str
    direction: Literal["upstream", "downstream", "both"] = "both"
    depth: int = Field(default=3, ge=1, le=5)
    lineage_granularity: Literal["table"] = "table"


class LineageRelation(BaseModel):
    source_table: str
    target_table: str
    depth: int = Field(ge=1)
    direction: Literal["upstream", "downstream"]


class LineageSearchOutput(BaseModel):
    root_table: str
    direction: Literal["upstream", "downstream", "both"]
    relations: list[LineageRelation] = Field(default_factory=list)


class ClassifyImpactInput(BaseModel):
    operation: str | None = None
    direction: Literal["upstream", "downstream", "both"]
    lineage_result: dict[str, Any]


class ImpactItem(BaseModel):
    table: str
    direction: str
    depth: int
    impact_level: Literal["direct", "indirect"]
    semantic_score: float | None = None


class ImpactAnalysisOutput(BaseModel):
    operation: str | None = None
    impacts: list[ImpactItem] = Field(default_factory=list)
    degraded_sources: list[str] = Field(default_factory=list)


class MergeImpactInput(ClassifyImpactInput):
    semantic_result: dict[str, Any] = Field(default_factory=dict)
