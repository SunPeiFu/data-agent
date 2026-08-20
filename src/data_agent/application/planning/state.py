"""Typed state contract shared by every node in the planning graph."""

from __future__ import annotations

from typing import TypedDict

from data_agent.domain.models import (
    AccessContext,
    AuthorizationDecision,
    ClarificationAnswerRecord,
    ExtractedEntities,
    IntentType,
    MetadataQueryMode,
    MetadataCandidateEvidence,
    NodeTrace,
    NormalizationTrace,
    NormalizedTerm,
    PlanningResult,
    PlanValidationResult,
    SlotValidationResult,
    TableIdentifier,
    TraceContext,
    TraceEvent,
    TaskExecutionResult,
)
from data_agent.intelligence.hybrid_router import HybridRouteResult


class PlannerState(TypedDict, total=False):
    """Single source of truth for values exchanged between LangGraph nodes."""

    question: str
    thread_id: str
    access_context: AccessContext
    route_result: HybridRouteResult
    routing_notes: list[str]
    normalization_notes: list[str]
    normalized_terms: list[NormalizedTerm]
    normalization_traces: list[NormalizationTrace]
    metadata_notes: list[str]
    authorization_notes: list[str]
    clarification_notes: list[str]
    plan_validation_notes: list[str]
    plan_validation: PlanValidationResult
    trace_notes: list[str]
    trace_context: TraceContext
    node_traces: list[NodeTrace]
    trace_events: list[TraceEvent]
    slot_errors: list[str]
    pre_slot_validation: SlotValidationResult
    post_slot_validation: SlotValidationResult
    planner_decision: str
    clarification_round: int
    max_clarification_rounds: int
    state_version: int
    clarification_history: list[ClarificationAnswerRecord]
    confirmed_slots: list[str]
    processed_idempotency_keys: list[str]
    execute_requested: bool
    task_execution_result: TaskExecutionResult
    metadata_candidates: dict[str, list[str]]
    metadata_candidate_profiles: dict[str, dict[str, str | None]]
    metadata_candidate_evidence: dict[str, MetadataCandidateEvidence]
    table_term_candidates: dict[str, list[str]]
    semantic_table_query: str | None
    requested_table: TableIdentifier | None
    authorized: bool
    authorization_decisions: list[AuthorizationDecision]
    trace_id: str
    intent: IntentType
    metadata_query_mode: MetadataQueryMode | None
    metadata_query_mode_notes: list[str]
    confidence: float
    entities: ExtractedEntities
    result: PlanningResult
