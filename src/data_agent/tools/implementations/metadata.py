"""Executable metadata and semantic retrieval tools."""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from data_agent.domain.models import DataLayer, DomainType, ExtractedEntities
from data_agent.infrastructure.repositories.milvus_metadata import MilvusMetadataRepository
from data_agent.infrastructure.repositories.mysql_metadata import MetadataCandidate, MySQLMetadataRepository
from data_agent.tools.contracts import (
    FilterTablesInput,
    FilterTablesOutput,
    GetTableDetailInput,
    GetTableDetailOutput,
    MergeRankInput,
    MergeRankOutput,
    SemanticSearchInput,
    SemanticSearchOutput,
    TableAsset,
)


def create_filter_tables_tool(repository: MySQLMetadataRepository | None = None) -> StructuredTool:
    repo = repository or MySQLMetadataRepository()

    def filter_tables(**kwargs: object) -> dict[str, object]:
        payload = FilterTablesInput.model_validate(kwargs)
        entities = _entities_from_filters(payload.biz_line, payload.domain, payload.data_layer)
        candidates = repo.filter_tables(entities, payload.topic_keywords, payload.limit)
        return FilterTablesOutput(assets=[_asset(candidate) for candidate in candidates]).model_dump(mode="json")

    return StructuredTool.from_function(
        func=filter_tables,
        name="tidb_metadata__filter_tables",
        description="按业务线、主题域、数仓分层和关键词精确过滤数据表资产。",
        args_schema=FilterTablesInput,
    )


def create_get_table_detail_tool(repository: MySQLMetadataRepository | None = None) -> StructuredTool:
    repo = repository or MySQLMetadataRepository()

    def get_table_detail(**kwargs: object) -> dict[str, object]:
        payload = GetTableDetailInput.model_validate(kwargs)
        candidate = repo.get_table_detail(payload.table)
        if candidate is None:
            raise ValueError(f"未找到唯一在线表资产: {payload.table}")
        return GetTableDetailOutput(
            asset=_asset(candidate),
            included_sections=list(payload.include),
        ).model_dump(mode="json")

    return StructuredTool.from_function(
        func=get_table_detail,
        name="tidb_metadata__get_table_detail",
        description="读取已解析唯一物理表的基础信息、业务说明和负责人。",
        args_schema=GetTableDetailInput,
    )


def create_semantic_search_tool(
    milvus_repository: MilvusMetadataRepository | None = None,
    mysql_repository: MySQLMetadataRepository | None = None,
) -> StructuredTool:
    milvus = milvus_repository or MilvusMetadataRepository()
    mysql = mysql_repository or MySQLMetadataRepository()

    def semantic_search(**kwargs: object) -> dict[str, object]:
        payload = SemanticSearchInput.model_validate(kwargs)
        entities = _entities_from_filters(payload.biz_line, payload.domain, payload.data_layer)
        recalled = milvus.hybrid_search(payload.query, entities, payload.top_k)
        scores = {candidate.full_table_name: candidate.score for candidate in recalled.candidates}
        validated = mysql.find_by_full_table_names(list(scores), entities)
        assets = [
            _asset(candidate).model_copy(update={"semantic_score": scores.get(candidate.full_table_name)})
            for candidate in validated
        ]
        return SemanticSearchOutput(
            assets=assets,
            retrieval_mode=recalled.retrieval_mode,
        ).model_dump(mode="json")

    return StructuredTool.from_function(
        func=semantic_search,
        name="milvus_rag__semantic_search",
        description="使用 Milvus 混合召回表资产，并通过 MySQL/TiDB 事实源回查验证。",
        args_schema=SemanticSearchInput,
    )


def create_merge_rank_tool() -> StructuredTool:
    def merge_and_rank(**kwargs: object) -> dict[str, object]:
        payload = MergeRankInput.model_validate(kwargs)
        merged: dict[str, TableAsset] = {}
        for raw in payload.exact_result.get("assets", []):
            asset = TableAsset.model_validate(raw)
            merged[asset.full_table_name] = asset
        for raw in payload.semantic_result.get("assets", []):
            semantic = TableAsset.model_validate(raw)
            current = merged.get(semantic.full_table_name)
            merged[semantic.full_table_name] = semantic if current is None else current.model_copy(
                update={"semantic_score": semantic.semantic_score}
            )
        assets = sorted(
            merged.values(),
            key=lambda item: (
                item.exact_score is not None,
                item.exact_score or 0.0,
                item.semantic_score or 0.0,
            ),
            reverse=True,
        )
        return MergeRankOutput(assets=assets, total=len(assets)).model_dump(mode="json")

    return StructuredTool.from_function(
        func=merge_and_rank,
        name="result_ranker__merge_and_rank",
        description="合并 TiDB 精确结果和 Milvus 语义结果，去重后进行可解释排序。",
        args_schema=MergeRankInput,
    )


def _entities_from_filters(biz_line: str | None, domain: str | None, data_layer: str | None) -> ExtractedEntities:
    return ExtractedEntities(
        biz_line=biz_line,
        domain=DomainType(domain) if domain else None,
        data_layer=DataLayer(data_layer.upper()) if data_layer else None,
    )


def _asset(candidate: MetadataCandidate) -> TableAsset:
    return TableAsset(
        full_table_name=candidate.full_table_name,
        catalog_name=candidate.catalog_name,
        db_name=candidate.db_name,
        table_name=candidate.table_name,
        table_comment=candidate.table_comment,
        biz_line=candidate.biz_line,
        domain=candidate.domain,
        data_layer=candidate.data_layer,
        owner=candidate.owner,
        exact_score=float(candidate.score),
    )
