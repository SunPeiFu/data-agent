from __future__ import annotations

from data_agent.infrastructure.repositories.mysql_metadata import MetadataCandidate
from data_agent.infrastructure.repositories.milvus_metadata import (
    MilvusMetadataCandidate,
    MilvusSearchResponse,
    _build_scalar_filter,
)
from data_agent.domain.models import (
    DataLayer,
    DomainType,
    ExtractedEntities,
    IntentType,
    MetadataCandidateEvidence,
    MetadataCandidateSource,
    MetadataValidationStatus,
    SlotIssueType,
    TableIdentifier,
)
from data_agent.application.planning import nodes as planner
from data_agent.application.planning.service import plan_question


def _mysql_candidate(full_table_name: str, table_name: str) -> MetadataCandidate:
    return MetadataCandidate(
        full_table_name=full_table_name,
        catalog_name=None,
        db_name="dwd",
        table_name=table_name,
        table_comment="广告投放转化明细",
        biz_line="安逸花",
        domain="营销域",
        data_layer="DWD",
        owner="data-team",
        score=7,
    )


def test_exact_identifier_uses_mysql_and_skips_milvus(monkeypatch) -> None:
    class FakeMySQL:
        def find_by_table_identifier(self, table, entities):
            return [_mysql_candidate("dwd.orderInfo", "orderInfo")]

    class FailIfCalledMilvus:
        def hybrid_search(self, query, entities, top_k=20):
            raise AssertionError("exact identifier must not call Milvus")

    monkeypatch.setattr(planner, "MySQLMetadataRepository", FakeMySQL)
    monkeypatch.setattr(planner, "MilvusMetadataRepository", FailIfCalledMilvus)
    state = {
        "question": "查询 dwd.orderInfo 的下游",
        "entities": ExtractedEntities(table=TableIdentifier.parse("dwd.orderInfo")),
        "semantic_table_query": None,
    }

    resolved = planner._resolve_metadata_candidates(state)

    assert resolved["metadata_candidates"]["table"] == ["dwd.orderInfo"]
    assert resolved["metadata_candidate_evidence"]["dwd.orderInfo"].validation_status == MetadataValidationStatus.VALIDATED
    assert any("跳过 Milvus" in note for note in resolved["metadata_notes"])


def test_one_part_identifier_uses_milvus_only_after_mysql_miss(monkeypatch) -> None:
    class FakeMySQL:
        def find_by_table_identifier(self, table, entities):
            return []

        def find_by_full_table_names(self, table_names, entities):
            assert table_names == ["dwd.ad_campaign_conversion"]
            return [_mysql_candidate("dwd.ad_campaign_conversion", "ad_campaign_conversion")]

    class FakeMilvus:
        def hybrid_search(self, query, entities, top_k=20):
            return MilvusSearchResponse(
                candidates=[MilvusMetadataCandidate("dwd.ad_campaign_conversion", 0.91)],
                retrieval_mode="dense_bm25_rrf",
            )

    monkeypatch.setattr(planner, "MySQLMetadataRepository", FakeMySQL)
    monkeypatch.setattr(planner, "MilvusMetadataRepository", FakeMilvus)
    state = {
        "question": "营销域 DWD 层 adConversion 表的下游影响",
        "entities": ExtractedEntities(
            table=TableIdentifier.parse("adConversion"),
            domain=DomainType.MARKETING,
            data_layer=DataLayer.DWD,
        ),
        "semantic_table_query": "营销域 DWD 层 adConversion 表的下游影响",
    }

    resolved = planner._resolve_metadata_candidates(state)

    assert resolved["metadata_candidates"]["table"] == ["dwd.ad_campaign_conversion"]
    assert resolved["entities"].table.raw == "dwd.ad_campaign_conversion"
    assert any("dense_bm25_rrf" in note for note in resolved["metadata_notes"])


def test_semantic_candidate_is_backfilled_only_after_mysql_validation(monkeypatch) -> None:
    class FakeMySQL:
        def find_by_full_table_names(self, table_names, entities):
            return [_mysql_candidate("dwd.ad_campaign_conversion", "ad_campaign_conversion")]

    class FakeMilvus:
        def hybrid_search(self, query, entities, top_k=20):
            return MilvusSearchResponse(
                candidates=[MilvusMetadataCandidate("dwd.ad_campaign_conversion", 0.88)],
                retrieval_mode="dense_bm25_rrf",
            )

    monkeypatch.setattr(planner, "MySQLMetadataRepository", FakeMySQL)
    monkeypatch.setattr(planner, "MilvusMetadataRepository", FakeMilvus)
    state = {
        "question": "营销域 DWD 层广告投放转化率相关表的下游影响",
        "entities": ExtractedEntities(domain=DomainType.MARKETING, data_layer=DataLayer.DWD),
        "semantic_table_query": "营销域 DWD 层广告投放转化率相关表的下游影响",
    }

    resolved = planner._resolve_metadata_candidates(state)

    assert resolved["entities"].table is not None
    assert resolved["entities"].table.raw == "dwd.ad_campaign_conversion"
    assert resolved["metadata_candidate_profiles"]["dwd.ad_campaign_conversion"]["domain"] == "营销域"


def test_milvus_scalar_filter_uses_normalized_business_context() -> None:
    expression = _build_scalar_filter(
        ExtractedEntities(
            biz_line="安逸花",
            domain=DomainType.MARKETING,
            data_layer=DataLayer.DWD,
        )
    )

    assert 'lifecycle_status == "online"' in expression
    assert 'biz_line == "安逸花"' in expression
    assert 'domain == "营销域"' in expression
    assert 'data_layer == "DWD"' in expression


def test_post_validation_blocks_mock_fallback_candidate(monkeypatch) -> None:
    class FailingMySQL:
        def find_by_table_identifier(self, table, entities):
            raise planner.MetadataRepositoryError("database unavailable")

    class FailIfCalledMilvus:
        def hybrid_search(self, query, entities, top_k=20):
            raise AssertionError("exact identifier must not call Milvus")

    monkeypatch.setattr(planner, "MySQLMetadataRepository", FailingMySQL)
    monkeypatch.setattr(planner, "MilvusMetadataRepository", FailIfCalledMilvus)

    result = plan_question("dwd.orderInfo 的下游血缘有哪些")

    assert result.need_clarification is True
    assert result.task_steps == []
    assert any("mock_fallback" in note and "尚未通过 MySQL" in note for note in result.notes)


def test_post_validation_blocks_low_score_semantic_candidate(monkeypatch) -> None:
    monkeypatch.setenv("DATA_AGENT_MILVUS_MIN_SCORE", "0.50")
    evidence = MetadataCandidateEvidence(
        full_table_name="dwd.ad_campaign_conversion",
        source=MetadataCandidateSource.MILVUS_MYSQL_VALIDATED,
        validation_status=MetadataValidationStatus.VALIDATED,
        score=0.20,
        rank=1,
        retrieval_mode="dense_bm25_rrf",
    )
    entities = ExtractedEntities(
        table=TableIdentifier.parse("dwd.ad_campaign_conversion"),
        domain=DomainType.MARKETING,
        data_layer=DataLayer.DWD,
    )
    state = {
        "intent": IntentType.IMPACT_ANALYSIS,
        "entities": entities,
        "metadata_candidates": {"table": ["dwd.ad_campaign_conversion"]},
        "metadata_candidate_evidence": {"dwd.ad_campaign_conversion": evidence},
        "metadata_candidate_profiles": {
            "dwd.ad_campaign_conversion": _mysql_candidate(
                "dwd.ad_campaign_conversion", "ad_campaign_conversion"
            ).profile()
        },
    }

    validation = planner._post_validate_slots(state)["post_slot_validation"]

    assert validation.passed is False
    assert any(issue.issue_type == SlotIssueType.LOW_CONFIDENCE for issue in validation.issues)


def test_post_validation_requires_authoritative_candidate_profile() -> None:
    table_name = "dwd.orderInfo"
    state = {
        "intent": IntentType.LINEAGE_SEARCH,
        "entities": ExtractedEntities(table=TableIdentifier.parse(table_name)),
        "metadata_candidates": {"table": [table_name]},
        "metadata_candidate_evidence": {
            table_name: MetadataCandidateEvidence(
                full_table_name=table_name,
                source=MetadataCandidateSource.MYSQL_IDENTIFIER,
                validation_status=MetadataValidationStatus.VALIDATED,
            )
        },
        "metadata_candidate_profiles": {},
        "requested_table": TableIdentifier.parse(table_name),
    }

    validation = planner._post_validate_slots(state)["post_slot_validation"]

    assert validation.passed is False
    assert any(issue.slot_name == "table_profile" for issue in validation.issues)


def test_post_validation_detects_requested_table_identity_conflict() -> None:
    table_name = "dwd.orderInfo"
    state = {
        "intent": IntentType.LINEAGE_SEARCH,
        "entities": ExtractedEntities(table=TableIdentifier.parse(table_name)),
        "metadata_candidates": {"table": [table_name]},
        "metadata_candidate_evidence": {
            table_name: MetadataCandidateEvidence(
                full_table_name=table_name,
                source=MetadataCandidateSource.MYSQL_IDENTIFIER,
                validation_status=MetadataValidationStatus.VALIDATED,
            )
        },
        "metadata_candidate_profiles": {table_name: _mysql_candidate(table_name, "orderInfo").profile()},
        "requested_table": TableIdentifier.parse("other_db.orderInfo"),
    }

    validation = planner._post_validate_slots(state)["post_slot_validation"]

    assert validation.passed is False
    assert any(issue.issue_type == SlotIssueType.CONFLICT for issue in validation.issues)


def test_post_validation_deduplicates_missing_table_issue() -> None:
    state = {
        "intent": IntentType.LINEAGE_SEARCH,
        "entities": ExtractedEntities(),
        "metadata_candidates": {"table": []},
    }

    validation = planner._post_validate_slots(state)["post_slot_validation"]
    missing_table_issues = [
        issue
        for issue in validation.issues
        if issue.slot_name == "table" and issue.issue_type == SlotIssueType.MISSING
    ]

    assert len(missing_table_issues) == 1
