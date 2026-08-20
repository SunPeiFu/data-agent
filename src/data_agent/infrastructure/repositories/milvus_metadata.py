from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx

from data_agent.domain.models import ExtractedEntities
from data_agent.config.settings import EmbeddingSettings


class MilvusRepositoryError(RuntimeError):
    """Normalize Milvus, SDK, collection, and embedding failures for planner fallback."""


@dataclass(frozen=True)
class MilvusMetadataCandidate:
    """A non-authoritative table candidate recalled from the semantic index."""

    full_table_name: str
    score: float


@dataclass(frozen=True)
class MilvusSearchResponse:
    """Return candidates together with the actual retrieval mode used."""

    candidates: list[MilvusMetadataCandidate]
    retrieval_mode: str


class OpenAICompatibleEmbeddingClient:
    """Call an OpenAI-compatible `/embeddings` endpoint such as LM Studio."""

    def __init__(self, settings: EmbeddingSettings | None = None) -> None:
        self.settings = settings or EmbeddingSettings.from_env()

    def embed_query(self, text: str) -> list[float]:
        if not self.settings.model:
            raise MilvusRepositoryError(
                "未配置 DATA_AGENT_EMBEDDING_MODEL，Dense 检索不可用。"
            )
        url = self.settings.base_url.rstrip("/") + "/embeddings"
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=self.settings.timeout_seconds) as client:
                response = client.post(
                    url,
                    headers=headers,
                    json={"model": self.settings.model, "input": text},
                )
                response.raise_for_status()
            vector = response.json()["data"][0]["embedding"]
        except Exception as exc:  # noqa: BLE001 - normalized at repository boundary.
            raise MilvusRepositoryError(f"Embedding 服务调用失败: {exc}") from exc
        if len(vector) != self.settings.dimension:
            raise MilvusRepositoryError(
                f"Embedding 维度 {len(vector)} 与 Collection 维度 {self.settings.dimension} 不一致。"
            )
        return [float(value) for value in vector]


class MilvusMetadataRepository:
    """Recall table assets with dense semantic search + BM25 + scalar filters.

    Milvus is deliberately treated as a recall engine. Returned names must still be
    validated by MySQL/TiDB before downstream lineage tools can execute.
    """

    OUTPUT_FIELDS = ["full_table_name"]

    def __init__(
        self,
        uri: str | None = None,
        collection_name: str | None = None,
        embedding_client: OpenAICompatibleEmbeddingClient | None = None,
    ) -> None:
        self.uri = uri or os.getenv("DATA_AGENT_MILVUS_URI", "http://127.0.0.1:19531")
        self.collection_name = collection_name or os.getenv(
            "DATA_AGENT_MILVUS_COLLECTION", "data_agent_table_assets_v1"
        )
        self.embedding_client = embedding_client or OpenAICompatibleEmbeddingClient()

    def hybrid_search(
        self,
        query: str,
        entities: ExtractedEntities,
        top_k: int = 20,
    ) -> MilvusSearchResponse:
        """Run hybrid retrieval and degrade to BM25 when dense embedding is unavailable."""
        try:
            from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker
        except ModuleNotFoundError as exc:
            raise MilvusRepositoryError("pymilvus 未安装，无法查询 Milvus。") from exc

        try:
            client = MilvusClient(uri=self.uri)
            if not client.has_collection(self.collection_name):
                raise MilvusRepositoryError(f"Milvus Collection {self.collection_name} 不存在。")
            filter_expression = _build_scalar_filter(entities)
            sparse_request = AnnSearchRequest(
                data=[query],
                anns_field="sparse_vector",
                param={"metric_type": "BM25"},
                limit=top_k,
                expr=filter_expression,
            )
            try:
                dense_vector = self.embedding_client.embed_query(query)
            except MilvusRepositoryError:
                results = client.search(
                    collection_name=self.collection_name,
                    data=[query],
                    anns_field="sparse_vector",
                    search_params={"metric_type": "BM25"},
                    filter=filter_expression,
                    limit=top_k,
                    output_fields=self.OUTPUT_FIELDS,
                    consistency_level="Bounded",
                )
                return MilvusSearchResponse(_parse_candidates(results), "bm25_fallback")

            dense_request = AnnSearchRequest(
                data=[dense_vector],
                anns_field="dense_vector",
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=top_k,
                expr=filter_expression,
            )
            results = client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[dense_request, sparse_request],
                ranker=RRFRanker(60),
                limit=top_k,
                output_fields=self.OUTPUT_FIELDS,
                consistency_level="Bounded",
            )
            return MilvusSearchResponse(_parse_candidates(results), "dense_bm25_rrf")
        except MilvusRepositoryError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalized for planner degradation.
            raise MilvusRepositoryError(f"Milvus 元数据召回失败: {exc}") from exc


def _build_scalar_filter(entities: ExtractedEntities) -> str:
    """Translate trusted normalized slots into Milvus scalar filter expressions."""
    conditions = ["lifecycle_status == \"online\""]
    if entities.biz_line:
        conditions.append(f"biz_line == {json.dumps(entities.biz_line, ensure_ascii=False)}")
    if entities.domain:
        conditions.append(f"domain == {json.dumps(entities.domain.value, ensure_ascii=False)}")
    if entities.data_layer:
        conditions.append(f"data_layer == {json.dumps(entities.data_layer.value, ensure_ascii=False)}")
    return " and ".join(conditions)


def _parse_candidates(results: list[list[dict[str, Any]]]) -> list[MilvusMetadataCandidate]:
    """Convert SDK hits to stable domain objects while removing duplicate table names."""
    candidates: list[MilvusMetadataCandidate] = []
    seen: set[str] = set()
    for hit in results[0] if results else []:
        entity = hit.get("entity") or {}
        table_name = entity.get("full_table_name")
        if not table_name or table_name in seen:
            continue
        seen.add(table_name)
        score = hit.get("distance", hit.get("score", 0.0))
        candidates.append(MilvusMetadataCandidate(full_table_name=table_name, score=float(score)))
    return candidates
