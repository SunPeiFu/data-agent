from __future__ import annotations

from data_agent.infrastructure.repositories.milvus_metadata import (
    MilvusMetadataCandidate,
    MilvusSearchResponse,
)
from data_agent.infrastructure.repositories.mysql_metadata import MetadataCandidate
from data_agent.infrastructure.repositories.neo4j_lineage import LineageRecord
from data_agent.tools.implementations.impact import create_classify_impact_tool
from data_agent.tools.implementations.lineage import create_lineage_search_tool
from data_agent.tools.implementations.metadata import (
    create_filter_tables_tool,
    create_semantic_search_tool,
)


def _candidate(name: str = "dwd.orderInfo") -> MetadataCandidate:
    return MetadataCandidate(
        full_table_name=name,
        catalog_name=None,
        db_name="dwd",
        table_name=name.split(".")[-1],
        table_comment="订单明细",
        biz_line="安逸花",
        domain="营销域",
        data_layer="DWD",
        owner="data-team",
        score=7,
    )


def test_filter_tables_is_an_executable_langchain_tool() -> None:
    class FakeMySQL:
        def filter_tables(self, entities, topic_keywords, limit):
            assert entities.domain.value == "营销域"
            assert topic_keywords == ["支付"]
            assert limit == 10
            return [_candidate()]

    output = create_filter_tables_tool(FakeMySQL()).invoke(
        {"domain": "营销域", "topic_keywords": ["支付"], "limit": 10}
    )

    assert output["source"] == "mysql"
    assert output["assets"][0]["full_table_name"] == "dwd.orderInfo"


def test_semantic_search_validates_milvus_hits_against_mysql() -> None:
    class FakeMilvus:
        def hybrid_search(self, query, entities, top_k):
            return MilvusSearchResponse(
                candidates=[MilvusMetadataCandidate("dwd.orderInfo", 0.91)],
                retrieval_mode="dense_bm25_rrf",
            )

    class FakeMySQL:
        def find_by_full_table_names(self, names, entities):
            assert names == ["dwd.orderInfo"]
            return [_candidate()]

    output = create_semantic_search_tool(FakeMilvus(), FakeMySQL()).invoke(
        {"query": "支付订单", "top_k": 5, "domain": "营销域"}
    )

    assert output["source"] == "milvus_mysql_validated"
    assert output["assets"][0]["semantic_score"] == 0.91


def test_lineage_and_impact_tools_form_an_executable_chain() -> None:
    class FakeNeo4j:
        def lineage_search(self, table, direction, depth):
            return [
                LineageRecord(table, "dws.order_summary", 1, "downstream"),
                LineageRecord(table, "ads.order_report", 2, "downstream"),
            ]

    lineage = create_lineage_search_tool(FakeNeo4j()).invoke(
        {"table": "dwd.orderInfo", "direction": "downstream", "depth": 3}
    )
    impact = create_classify_impact_tool().invoke(
        {"operation": "modify_field", "direction": "downstream", "lineage_result": lineage}
    )

    assert [item["impact_level"] for item in impact["impacts"]] == ["direct", "indirect"]
