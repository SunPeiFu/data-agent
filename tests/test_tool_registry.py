from __future__ import annotations

from langchain_core.tools import BaseTool

from data_agent.domain.models import AccessContext, DomainType, ExtractedEntities, IntentType
from data_agent.tools.factory import get_default_tool_registry


def test_default_registry_contains_every_planned_tool() -> None:
    registry = get_default_tool_registry()

    assert {(item.service, item.action) for item in registry.all()} == {
        ("tidb_metadata", "filter_tables"),
        ("tidb_metadata", "get_table_detail"),
        ("milvus_rag", "semantic_search"),
        ("result_ranker", "merge_and_rank"),
        ("neo4j_lineage", "lineage_search"),
        ("impact_analyzer", "classify_impact"),
        ("impact_analyzer", "merge_lineage_and_metadata"),
    }
    assert all(isinstance(item.tool, BaseTool) for item in registry.all())
    assert all(item.tool.args_schema is not None for item in registry.all())


def test_registry_filters_tools_by_intent_and_user_permissions() -> None:
    registry = get_default_tool_registry()
    context = AccessContext(user_id="analyst", roles=["data_analyst"], tenant_id="demo")
    entities = ExtractedEntities(domain=DomainType.MARKETING, biz_line="安逸花")

    metadata = registry.available_for(IntentType.METADATA_SEARCH, context, entities)
    impact = registry.available_for(IntentType.IMPACT_ANALYSIS, context, entities)

    assert {item.action for item in metadata} == {
        "filter_tables",
        "get_table_detail",
        "semantic_search",
        "merge_and_rank",
    }
    assert impact == []
