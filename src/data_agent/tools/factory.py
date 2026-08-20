"""Construct the process-wide local tool registry."""

from __future__ import annotations

from functools import lru_cache

from data_agent.domain.models import IntentType
from data_agent.tools.implementations.impact import create_classify_impact_tool, create_merge_impact_tool
from data_agent.tools.implementations.lineage import create_lineage_search_tool
from data_agent.tools.implementations.metadata import (
    create_filter_tables_tool,
    create_get_table_detail_tool,
    create_merge_rank_tool,
    create_semantic_search_tool,
)
from data_agent.tools.contracts import (
    FilterTablesOutput,
    FilterTablesInput,
    GetTableDetailOutput,
    GetTableDetailInput,
    ImpactAnalysisOutput,
    ClassifyImpactInput,
    LineageSearchOutput,
    LineageSearchInput,
    MergeRankOutput,
    MergeRankInput,
    MergeImpactInput,
    SemanticSearchOutput,
    SemanticSearchInput,
)
from data_agent.tools.registry import RegisteredTool, RetryPolicy, ToolRegistry


@lru_cache(maxsize=1)
def get_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_many(
        [
            RegisteredTool(
                "tidb_metadata", "filter_tables", create_filter_tables_tool(), FilterTablesInput, FilterTablesOutput,
                frozenset({IntentType.METADATA_SEARCH}), "metadata:read", retry_policy=RetryPolicy(2, 0.1),
            ),
            RegisteredTool(
                "tidb_metadata", "get_table_detail", create_get_table_detail_tool(), GetTableDetailInput,
                GetTableDetailOutput,
                frozenset({IntentType.METADATA_SEARCH}), "metadata:read", retry_policy=RetryPolicy(2, 0.1),
            ),
            RegisteredTool(
                "milvus_rag", "semantic_search", create_semantic_search_tool(), SemanticSearchInput,
                SemanticSearchOutput,
                frozenset({IntentType.METADATA_SEARCH, IntentType.IMPACT_ANALYSIS}), "metadata:read",
                timeout_seconds=30.0, retry_policy=RetryPolicy(2, 0.2),
            ),
            RegisteredTool(
                "result_ranker", "merge_and_rank", create_merge_rank_tool(), MergeRankInput, MergeRankOutput,
                frozenset({IntentType.METADATA_SEARCH}), "metadata:read", allow_failed_dependencies=True,
            ),
            RegisteredTool(
                "neo4j_lineage", "lineage_search", create_lineage_search_tool(), LineageSearchInput,
                LineageSearchOutput,
                frozenset({IntentType.LINEAGE_SEARCH, IntentType.IMPACT_ANALYSIS}), "lineage:read",
                timeout_seconds=15.0, retry_policy=RetryPolicy(2, 0.1),
            ),
            RegisteredTool(
                "impact_analyzer", "classify_impact", create_classify_impact_tool(), ClassifyImpactInput,
                ImpactAnalysisOutput,
                frozenset({IntentType.IMPACT_ANALYSIS}), "impact:analyze",
            ),
            RegisteredTool(
                "impact_analyzer", "merge_lineage_and_metadata", create_merge_impact_tool(),
                MergeImpactInput, ImpactAnalysisOutput, frozenset({IntentType.IMPACT_ANALYSIS}), "impact:analyze",
                allow_failed_dependencies=True,
            ),
        ]
    )
    return registry
