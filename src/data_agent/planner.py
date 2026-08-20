from __future__ import annotations

import os
import re
import logging
from datetime import datetime, timezone
from hashlib import sha256
from functools import lru_cache
from time import perf_counter
from typing import Any, Callable, TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from data_agent.authorization import YamlAuthorizationProvider
from data_agent.checkpointing import get_planner_checkpointer
from data_agent.hybrid_router import HybridRouteResult, HybridQuestionRouter
from data_agent.metadata_repository import MetadataCandidate, MetadataRepositoryError, MySQLMetadataRepository
from data_agent.milvus_repository import MilvusMetadataRepository, MilvusRepositoryError
from data_agent.models import (
    AccessContext,
    AgentRunStatus,
    AuthorizationDecision,
    ClarificationInputType,
    ClarificationAnswerRecord,
    ClarificationOption,
    ClarificationRequest,
    ClarificationResponse,
    DataLayer,
    DomainType,
    ExtractedEntities,
    IntentType,
    LineageDirection,
    MetadataCandidateEvidence,
    MetadataCandidateSource,
    MetadataValidationStatus,
    NormalizationTrace,
    NormalizedTerm,
    NormalizedTermType,
    NodeRunStatus,
    NodeTrace,
    OperationType,
    PlanningResult,
    SlotIssue,
    SlotIssueType,
    SlotValidationResult,
    SlotValidationStage,
    TableIdentifier,
    TraceContext,
    TraceEvent,
)
from data_agent.normalization import load_normalization_config
from data_agent.slot_rules import load_slot_rule_config
from data_agent.task_builder import build_task_plan
from data_agent.trace_recorder import TraceRecorder, get_trace_recorder


PLANNER_VERSION = "v1-enterprise-planner"
_TRACE_LOGGER = logging.getLogger("data_agent.trace.instrumentation")


class PlannerState(TypedDict, total=False):
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
    clarification_notes: list[str] # 澄清
    plan_validation_notes: list[str]
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
    confidence: float
    entities: ExtractedEntities
    result: PlanningResult


def traced_node(
    node_name: str,
    handler: Callable[[PlannerState], PlannerState],
    recorder: TraceRecorder | None = None,
) -> Callable[[PlannerState], PlannerState]:
    """Wrap one business node with structured success/failure tracing.

    Stage one deliberately records summaries instead of complete State values. Recorder failures are
    isolated so observability infrastructure cannot break the Agent's main business path.
    """

    def wrapped(state: PlannerState) -> PlannerState:
        context = state.get("trace_context")
        if context is None:
            raise RuntimeError(f"node={node_name} 执行前缺少 TraceContext。")

        active_recorder = recorder or get_trace_recorder()
        started_at = _utc_now()
        started_tick = perf_counter()
        trace = NodeTrace(
            node_run_id=f"node-{uuid4().hex}",
            trace_id=context.trace_id,
            run_id=context.run_id,
            node_name=node_name,
            status=NodeRunStatus.RUNNING,
            started_at=started_at,
            input_summary={"available_state_fields": sorted(state.keys())},
        )
        _record_safely(active_recorder.start_node, trace)

        try:
            output = handler(state)
        except Exception as exc:
            failed_trace = trace.model_copy(
                update={
                    "status": NodeRunStatus.FAILED,
                    "finished_at": _utc_now(),
                    "duration_ms": round((perf_counter() - started_tick) * 1000, 3),
                    "error_code": type(exc).__name__,
                    "error_message": str(exc)[:500],
                }
            )
            _record_safely(active_recorder.fail_node, failed_trace)
            _record_safely(active_recorder.finish_run, context, AgentRunStatus.FAILED)
            raise

        completed_trace = trace.model_copy(
            update={
                "status": NodeRunStatus.COMPLETED,
                "finished_at": _utc_now(),
                "duration_ms": round((perf_counter() - started_tick) * 1000, 3),
                "output_summary": {"updated_state_fields": sorted(output.keys())},
            }
        )
        _record_safely(active_recorder.finish_node, completed_trace)
        return {**output, "node_traces": [*state.get("node_traces", []), completed_trace]}

    wrapped.__name__ = f"traced_{node_name}"
    return wrapped


def _record_safely(operation: Callable[..., None], *args: Any) -> None:
    """Keep Trace backend failures from changing Agent routing or user-visible behavior."""
    try:
        operation(*args)
    except Exception:
        _TRACE_LOGGER.exception("Trace recorder operation failed: %s", getattr(operation, "__name__", operation))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_trace_context(thread_id: str, parent_run_id: str | None = None) -> TraceContext:
    """Create a new execution segment; resumed clarification keeps thread_id but links parent_run_id."""
    return TraceContext(
        trace_id=f"trace-{uuid4().hex}",
        run_id=f"run-{uuid4().hex}",
        thread_id=thread_id,
        parent_run_id=parent_run_id,
        planner_version=PLANNER_VERSION,
        started_at=_utc_now(),
    )


def _init_trace_context(state: PlannerState) -> PlannerState:
    """Create TraceContext before the first business node and open the initial Agent Run."""
    context = _new_trace_context(state["thread_id"])
    _record_safely(get_trace_recorder().start_run, context)
    return {"trace_context": context, "node_traces": [], "trace_events": []}


def create_planning_graph(checkpointer: Any | None = None) -> Any:

    # 创建一个状态图 
    # add_node节点即不同的python函数 即声明功能
    # add_edge即声明节点之间的流程编排顺序
    graph = StateGraph(PlannerState)

    # step0 在任何业务处理前创建 trace_id/run_id，保证前置节点异常也可以追踪。
    graph.add_node("init_trace_context", _init_trace_context)

    # step1 意图识别 state返回完整的意图识别结果和相关信息 都是从route_result中获
    graph.add_node("classify_intent", traced_node("classify_intent", _classify_intent))

    # step2 实体抽取 意图识别的result中直接提取entities
    graph.add_node("extract_entities", traced_node("extract_entities", _extract_entities_node))

    # step3 归一化实体 (让进入工具之前的实体完全符合工具要求)
    graph.add_node("normalize_entities", traced_node("normalize_entities", _normalize_entities))

    # step4 元数据解析前槽位校验: 只基于用户输入和实体抽取结果，判断是否具备最小可执行线索
    # 典型阻断: 用户只说“查下游”，但没有给表名或表级业务术语
    graph.add_node("validate_slots", traced_node("validate_slots", _validate_slots))

    # step5 元数据候选解析: 优先查询 MySQL meta_table/meta_table_ext，失败时使用 mock fallback
    # 典型转换: userInfo / 订单信息表 -> dwd.userInfo / dim.userInfo / dwd.orderInfo
    graph.add_node(
        "resolve_metadata_candidates",
        traced_node("resolve_metadata_candidates", _resolve_metadata_candidates),
    )

    # step6 表级权限校验: 当前使用 YAML RBAC Provider，生产可替换统一权限中心 Provider
    graph.add_node("authorize_context", traced_node("authorize_context", _authorize_context))

    # step7 元数据解析后槽位校验: 判断候选表是否唯一、是否还缺关键槽位
    graph.add_node("post_validate_slots", traced_node("post_validate_slots", _post_validate_slots))

    # step8 📌 澄清决策: 判断缺槽位、多候选、无权限等是否需要先问用户
    graph.add_node(
        "decide_clarification_or_continue",
        traced_node("decide_clarification_or_continue", _decide_clarification_or_continue),
    )

    # step9 构建计划
    graph.add_node("build_task_plan", traced_node("build_task_plan", _build_task_plan))

    # step10 生成澄清结果: conditional edge 命中后不继续生成工具计划
    graph.add_node(
        "return_clarification_result",
        traced_node("return_clarification_result", _return_clarification_result),
    )

    # step10.1 持久化暂停点: 等待用户提交结构化澄清回答，然后回到实体标准化重新校验
    graph.add_node("await_clarification_response", _await_clarification_response)

    # step11 生成拒绝结果: conditional edge 命中后不继续生成工具计划
    graph.add_node(
        "return_forbidden_result",
        traced_node("return_forbidden_result", _return_forbidden_result),
    )

    # step11.1 超过最大澄清轮数后转人工，避免 Agent 无限追问
    graph.add_node(
        "return_handoff_result",
        traced_node("return_handoff_result", _return_handoff_result),
    )

    # step12 📌校验任务计划: 工具名、参数、依赖关系等
    graph.add_node("validate_task_plan", traced_node("validate_task_plan", _validate_task_plan))

    # step13 附加 trace: 记录路由、候选解析、权限、计划校验等备注
    graph.add_node("attach_trace", _attach_trace)

    # step14 返回计划结果
    graph.add_node("return_planning_result", _return_planning_result)

    # 设置整个图的first节点是什么 开始节点
    graph.set_entry_point("init_trace_context")

    # 设置节点之间的编排流程
    graph.add_edge("init_trace_context", "classify_intent")
    graph.add_edge("classify_intent", "extract_entities")
    graph.add_edge("extract_entities", "normalize_entities")
    graph.add_edge("normalize_entities", "validate_slots")
    graph.add_edge("validate_slots", "resolve_metadata_candidates")
    graph.add_edge("resolve_metadata_candidates", "authorize_context")
    graph.add_edge("authorize_context", "post_validate_slots")
    graph.add_edge("post_validate_slots", "decide_clarification_or_continue")
    graph.add_conditional_edges(
        "decide_clarification_or_continue",
        _route_after_clarification_decision,
        {
            "continue": "build_task_plan",
            "clarify": "return_clarification_result",
            "forbidden": "return_forbidden_result",
            "handoff": "return_handoff_result",
        },
    )
    graph.add_edge("build_task_plan", "validate_task_plan")
    graph.add_edge("validate_task_plan", "attach_trace")
    graph.add_edge("return_clarification_result", "attach_trace")
    graph.add_edge("return_forbidden_result", "attach_trace")
    graph.add_edge("return_handoff_result", "attach_trace")
    graph.add_conditional_edges(
        "attach_trace",
        _route_after_trace,
        {
            "await_clarification": "await_clarification_response",
            "return": "return_planning_result",
        },
    )
    graph.add_edge("await_clarification_response", "normalize_entities")
    graph.add_edge("return_planning_result", END)
    return graph.compile(checkpointer=checkpointer or get_planner_checkpointer())


class ClarificationProtocolError(ValueError):
    """Raised when a clarification response is stale, mismatched, or unauthorized."""


@lru_cache(maxsize=1)
def get_planning_graph() -> Any:
    """Return one compiled graph backed by the durable local SQLite checkpointer."""
    return create_planning_graph(checkpointer=get_planner_checkpointer())


def plan_question(
    question: str,
    access_context: AccessContext | None = None,
    *,
    thread_id: str | None = None,
    max_clarification_rounds: int = 3,
) -> PlanningResult:
    """Plan a data question under an authenticated access context.

    The default data_admin identity keeps the local learning CLI backward compatible. A production
    API must construct AccessContext from its authenticated gateway instead of trusting request data.
    """
    app = get_planning_graph()
    workflow_thread_id = thread_id or f"thread-{uuid4().hex}"
    config = {"configurable": {"thread_id": workflow_thread_id}}
    final_state = app.invoke(
        {
            "question": question,
            "thread_id": workflow_thread_id,
            "access_context": access_context
            or AccessContext(user_id="demo-user", roles=["data_admin"], tenant_id="demo"),
            "clarification_round": 0,
            "max_clarification_rounds": max(1, max_clarification_rounds),
            "state_version": 1,
            "clarification_history": [],
            "confirmed_slots": [],
            "processed_idempotency_keys": [],
        },
        config=config,
    )
    return final_state["result"]


def resume_clarification(response: ClarificationResponse) -> PlanningResult:
    """验证外部回答并恢复对应的持久化 LangGraph 会话。

    核心职责：
    1. 使用 thread_id 从 SQLite checkpointer 读取最新状态快照。
    2. 在恢复前检查幂等键；重复请求直接返回最新结果，不重复写实体或历史。
    3. 确认当前图确实暂停在 await_clarification_response，拒绝向已结束会话注入回答。
    4. 校验卡片 ID、版本和候选后，通过 Command(resume=...) 恢复原工作流。

    面试总结：幂等校验放在 resume 入口而不是仅依赖前端，能够覆盖网络超时重试、重复点击
    和消息队列重复投递；checkpoint snapshot 是判断会话状态的权威来源。
    """
    app = get_planning_graph()
    config = {"configurable": {"thread_id": response.thread_id}}
    snapshot = app.get_state(config)
    state = snapshot.values
    if not state:
        raise ClarificationProtocolError(f"未找到 thread_id={response.thread_id} 的澄清会话。")

    if response.idempotency_key in state.get("processed_idempotency_keys", []):
        result = state.get("result")
        if result is None:
            raise ClarificationProtocolError("幂等请求已处理，但会话中缺少可返回结果。")
        return result

    result = state.get("result")
    request = result.clarification_request if result else None
    if request is None or "await_clarification_response" not in snapshot.next:
        raise ClarificationProtocolError("当前会话不处于等待澄清状态。")
    _validate_clarification_response(request, response)

    resumed_state = app.invoke(Command(resume=response.model_dump(mode="json")), config=config)
    resumed_result = resumed_state.get("result")
    if resumed_result is None:
        raise ClarificationProtocolError("澄清恢复后没有生成 PlanningResult。")
    return resumed_result


def _classify_intent(state: PlannerState) -> PlannerState:
    route_result = HybridQuestionRouter().route(state["question"])
    return {
        "route_result": route_result,
        "routing_notes": route_result.notes,
        "intent": route_result.intent,
        "confidence": route_result.confidence,
    }


def _extract_entities_node(state: PlannerState) -> PlannerState:
    return {"entities": state["route_result"].entities}


def _normalize_entities(state: PlannerState) -> PlannerState:
    """Normalize extracted entities before slot validation and planning.

    企业级 DataAgent 里，LLM/规则抽取结果不能直接进入工具层：
    - 表名要去掉引号、空格，并统一 schema/catalog 大小写。
    - 表级业务术语要映射为标准 table_term，并沉淀候选物理表。
    - topic_keywords 要去重、去噪，避免把主题域/数仓分层当成检索词。
    - 归一化动作要写入 notes，方便排查“为什么工具参数变成这样”。
    """
    entities = state["entities"]
    question = state["question"]
    config = load_normalization_config()
    table_is_user_confirmed = "table" in state.get("confirmed_slots", [])
    if table_is_user_confirmed:
        table_terms: list[NormalizedTerm] = []
        table_traces: list[NormalizationTrace] = []
        table_term_candidates: dict[str, list[str]] = {}
    else:
        table_terms, table_traces, table_term_candidates = _extract_table_terms_from_question(question)
    topic_keywords, topic_terms, topic_traces = _normalize_topic_keywords(entities, question)
    normalized = ExtractedEntities(
        biz_line=_normalize_text(entities.biz_line),
        domain=entities.domain,
        data_layer=entities.data_layer,
        table=_normalize_table_identifier(entities.table),
        field_name=_normalize_identifier_text(entities.field_name),
        operation=entities.operation,
        topic_keywords=topic_keywords,
        lineage_direction=entities.lineage_direction,
    )

    traces = [
        *_build_basic_normalization_traces(before=entities, after=normalized),
        *table_traces,
        *topic_traces,
    ]
    terms = [*table_terms, *topic_terms]
    notes = _build_normalization_notes(before=entities, after=normalized, terms=terms, traces=traces)
    if config.stopwords:
        notes.append("实体标准化: stopwords/synonyms/table_terms 已从配置加载。")
    if table_is_user_confirmed:
        notes.append("实体标准化: table 来源为 user_confirmed，跳过原问题中的表术语候选扩展。")
    return {
        "entities": normalized,
        "normalization_notes": notes,
        "normalized_terms": terms,
        "normalization_traces": traces,
        "table_term_candidates": table_term_candidates,
        "semantic_table_query": _build_semantic_table_query(question, normalized, table_term_candidates),
    }


def _validate_slots(state: PlannerState) -> PlannerState:
    """Pre-metadata slot validation.

    核心职责：
    - 根据配置化 intent slot rule 判断用户是否提供最低可执行线索。
    - 这里只校验“有没有线索”，不判断候选是否唯一，因为真实候选要等元数据解析后才知道。
    - 输出结构化 SlotValidationResult，后续节点可以按 issue_type 做澄清、拒绝或继续执行。

    面试精华：
    这一步是“前置门禁”，不是查库校验。它的目标是尽早挡住完全不可执行的问题，
    避免后面浪费元数据查询、血缘查询和模型调用成本。
    """
    intent = state.get("intent", IntentType.UNKNOWN)
    entities = state["entities"]
    config = load_slot_rule_config()
    rule = config.rule_for(intent)
    issues: list[SlotIssue] = []
    notes = [f"槽位预校验: intent={intent.value} 使用配置化 required_any={rule.pre_required_any}。"]

    # intent 都无法判断时，后续无法选择工具模板，只能先澄清用户目标。
    if intent == IntentType.UNKNOWN:
        issues.append(
            _slot_issue(
                slot_name="intent",
                issue_type=SlotIssueType.MISSING,
                message="无法识别用户想查询元数据、血缘关系还是表变更影响。",
            )
        )
    # required_any 表示“这些槽位满足任意一个即可”。例如血缘查询有 table 或 table_term 即可进入元数据解析。
    elif rule.pre_required_any and not _has_any_slot(rule.pre_required_any, entities, state):
        issues.append(
            _slot_issue(
                slot_name=",".join(rule.pre_required_any),
                issue_type=SlotIssueType.MISSING,
                message=_missing_slot_message(intent),
            )
        )

    result = SlotValidationResult(
        stage=SlotValidationStage.PRE_METADATA,
        passed=not any(issue.blocking for issue in issues),
        issues=issues,
        notes=[*notes, *_slot_issue_notes("槽位预校验", issues)],
    )
    return {
        "pre_slot_validation": result,
        "slot_errors": [issue.message for issue in issues],
    }


def _post_validate_slots(state: PlannerState) -> PlannerState:
    """Post-metadata slot validation.

    核心职责：
    - 元数据候选解析完成后，检查表候选是否存在、是否唯一、是否可继续规划。
    - 这一步比 pre_validate 更接近生产，因为它基于元数据服务返回的候选质量做判断。
    - 校验候选来源、事实验证状态、规范表名、语义置信度和候选画像完整性。

    面试精华：
    前置校验回答“用户有没有给线索”，后置校验回答“这些线索能不能定位到唯一且一致的真实表”。
    真实生产里，表存在性、唯一性、主题域/分层一致性通常都放在这一层。
    """
    intent = state.get("intent", IntentType.UNKNOWN)
    entities = state["entities"]
    config = load_slot_rule_config()
    rule = config.rule_for(intent)
    candidates = state.get("metadata_candidates", {})
    table_candidates = candidates.get("table", [])
    candidate_profiles = state.get("metadata_candidate_profiles", {})
    candidate_evidence = state.get("metadata_candidate_evidence", {})
    requested_table = state.get("requested_table")
    issues: list[SlotIssue] = []
    notes = [f"槽位后校验: intent={intent.value} 使用配置化 post_required_any={rule.post_required_any}。"]

    # post_required_any 是元数据解析后的最终要求；表级血缘和影响分析必须已有可执行的表候选。
    if rule.post_required_any and not _has_any_slot(rule.post_required_any, entities, state):
        issues.append(
            _slot_issue(
                slot_name=",".join(rule.post_required_any),
                issue_type=SlotIssueType.MISSING,
                message="元数据解析后仍缺少可执行的表名。",
            )
        )

    # 对血缘/影响分析来说，候选表必须唯一且具备可审计的事实验证证据。
    if intent in {IntentType.LINEAGE_SEARCH, IntentType.IMPACT_ANALYSIS}:
        if not table_candidates:
            issues.append(
                _slot_issue(
                    slot_name="table",
                    issue_type=SlotIssueType.MISSING,
                    message="血缘查询或影响分析缺少表元数据候选。",
                )
            )
        elif len(table_candidates) > 1:
            issues.append(
                _slot_issue(
                    slot_name="table",
                    issue_type=SlotIssueType.AMBIGUOUS,
                    message=f"表名存在多个候选 {table_candidates}，需要用户选择唯一表。",
                )
            )
        else:
            table_name = table_candidates[0]
            evidence = candidate_evidence.get(table_name)
            profile = candidate_profiles.get(table_name)
            issues.extend(_validate_candidate_trust(table_name, evidence))
            issues.extend(_validate_semantic_candidate_confidence(table_name, evidence))
            issues.extend(_validate_executable_table_identity(entities, table_name))
            issues.extend(_validate_requested_table_identity(requested_table, table_name, profile))
            issues.extend(_validate_candidate_profile_completeness(entities, table_name, profile))
            issues.extend(_validate_cross_slot_consistency(entities, table_candidates, candidate_profiles))

    # 通用 required_any 与 intent 专项校验可能发现同一个问题，输出前按槽位和问题类型去重。
    issues = _deduplicate_slot_issues(issues)

    result = SlotValidationResult(
        stage=SlotValidationStage.POST_METADATA,
        passed=not any(issue.blocking for issue in issues),
        issues=issues,
        notes=[*notes, *_slot_issue_notes("槽位后校验", issues)],
    )
    return {"post_slot_validation": result}


def _resolve_metadata_candidates(state: PlannerState) -> PlannerState:
    """Resolve table candidates with certainty-aware MySQL/Milvus routing.

    生产路由：
    - 两段式/三段式技术表名：MySQL 精确校验，不调用 Milvus。
    - 一段式技术表名：先查 MySQL；无候选时再用 Milvus 补召回。
    - 业务描述或表级术语：Milvus Dense + BM25 + 标量过滤召回。
    - Milvus 返回的表名必须回 MySQL 校验，才允许进入血缘和影响分析。

    面试精华：
    MySQL/TiDB 是元数据事实源，Milvus 是候选召回器。路由依据是实体确定性，
    而不是所有问题固定先查一个库再查另一个库。
    """
    entities = state["entities"]
    requested_table = entities.table.model_copy(deep=True) if entities.table else None
    candidates: dict[str, list[str]] = {}
    candidate_profiles: dict[str, dict[str, str | None]] = {}
    candidate_evidence: dict[str, MetadataCandidateEvidence] = {}
    notes: list[str] = []
    mysql_repository = MySQLMetadataRepository()
    milvus_repository = MilvusMetadataRepository()

    table_term_candidates = state.get("table_term_candidates", {})
    table_terms = _table_term_lookup_values(state)
    table_candidates: list[str] = []

    # 技术表名路径：确定性越高，越优先使用结构化事实查询。
    if entities.table:
        try:
            resolved_tables = mysql_repository.find_by_table_identifier(entities.table, entities)
            table_candidates = _candidate_names(resolved_tables)
            candidate_profiles.update(_candidate_profiles(resolved_tables))
            _record_candidate_evidence(
                candidate_evidence,
                resolved_tables,
                source=MetadataCandidateSource.MYSQL_IDENTIFIER,
                status=MetadataValidationStatus.VALIDATED,
            )
            route = "exact_identifier" if entities.table.parts_count >= 2 else "partial_identifier"
            notes.append(f"元数据候选解析: route={route}，已通过 MySQL meta_table 查询物理表候选。")
        except MetadataRepositoryError as exc:
            table_candidates = _mock_table_candidates(entities.table)
            candidate_profiles.update(_mock_candidate_profiles(table_candidates))
            _record_fallback_evidence(candidate_evidence, table_candidates)
            notes.append(f"元数据候选解析: MySQL 查询失败，使用 mock 候选兜底。原因: {exc}")
        if entities.table.parts_count == 1:
            notes.append(f"元数据候选解析: 一段式表名 {entities.table.raw} 已生成候选 {table_candidates}。")
        else:
            notes.append(f"元数据候选解析: 表名 {entities.table.raw} 已完成事实源校验，跳过 Milvus。")

    # 结构化业务术语仍先查字典映射；它与后面的 Milvus 语义召回可以合并候选。
    if table_term_candidates:
        try:
            resolved_terms = mysql_repository.find_by_table_terms(table_terms, entities)
            term_candidates = _candidate_names(resolved_terms)
            candidate_profiles.update(_candidate_profiles(resolved_terms))
            _record_candidate_evidence(
                candidate_evidence,
                resolved_terms,
                source=MetadataCandidateSource.MYSQL_TABLE_TERM,
                status=MetadataValidationStatus.VALIDATED,
            )
            notes.append("元数据候选解析: 已通过 MySQL meta_table_ext 查询表级业务术语候选。")
        except MetadataRepositoryError as exc:
            term_candidates = [table for tables in table_term_candidates.values() for table in tables]
            candidate_profiles.update(_mock_candidate_profiles(term_candidates))
            _record_fallback_evidence(candidate_evidence, term_candidates)
            notes.append(f"元数据候选解析: MySQL 术语查询失败，使用配置候选兜底。原因: {exc}")
        table_candidates = _merge_preserve_order([*table_candidates, *term_candidates])
        notes.append(f"元数据候选解析: table_terms 命中候选表 {table_term_candidates}。")

    # 弱语义路径：没有明确技术表名，或一段式表名在 MySQL 中未命中时，调用 Milvus 混合召回。
    semantic_query = state.get("semantic_table_query")
    validated_before_semantic = _validated_candidate_names(candidate_evidence)
    if _should_use_semantic_recall(entities, validated_before_semantic, semantic_query):
        try:
            response = milvus_repository.hybrid_search(semantic_query or state["question"], entities, top_k=20)
            recalled_names = [candidate.full_table_name for candidate in response.candidates]
            notes.append(
                f"元数据候选解析: route=semantic_description，Milvus mode={response.retrieval_mode} "
                f"召回 {len(recalled_names)} 个候选。"
            )
            try:
                validated = mysql_repository.find_by_full_table_names(recalled_names, entities)
            except MetadataRepositoryError as exc:
                validated = []
                notes.append(f"元数据候选解析: Milvus 候选无法回 MySQL 校验，候选不进入执行链路。原因: {exc}")
            table_candidates = _merge_preserve_order([*table_candidates, *_candidate_names(validated)])
            candidate_profiles.update(_candidate_profiles(validated))
            _record_milvus_validated_evidence(candidate_evidence, response, validated)
            notes.append(f"元数据候选解析: {len(validated)} 个 Milvus 候选通过 MySQL 事实校验。")
        except MilvusRepositoryError as exc:
            notes.append(f"元数据候选解析: Milvus 召回不可用，保留结构化查询结果。原因: {exc}")

    # 一旦存在事实验证候选，就丢弃仅用于本地演示的 mock/config fallback，避免污染真实消歧。
    validated_candidates = [
        table_name
        for table_name in table_candidates
        if (evidence := candidate_evidence.get(table_name))
        and evidence.validation_status == MetadataValidationStatus.VALIDATED
    ]
    if validated_candidates:
        table_candidates = validated_candidates

    # 统一在所有召回路径结束后做上下文过滤，避免每条路径形成不同的消歧规则。
    table_candidates = _filter_table_candidates_by_context(table_candidates, entities)
    if table_candidates or entities.table or table_term_candidates or semantic_query:
        candidates["table"] = table_candidates

    # 唯一候选始终覆盖原始一段式实体，保证 Neo4j 接收到 db.table，而不是 userInfo。
    if len(table_candidates) == 1:
        entities = entities.model_copy(update={"table": TableIdentifier.parse(table_candidates[0])})
        notes.append(f"元数据候选解析: 唯一候选表 {table_candidates[0]} 已回填到实体。")
    elif entities.table is None and len(table_candidates) > 1:
        notes.append("元数据候选解析: 多个候选表未自动选择，等待 post_validate_slots 消歧。")

    if not notes:
        notes.append("元数据候选解析: 当前问题无需表候选解析。")

    candidate_profiles = {
        table_name: profile
        for table_name, profile in candidate_profiles.items()
        if table_name in candidates.get("table", [])
    }
    candidate_evidence = {
        table_name: evidence
        for table_name, evidence in candidate_evidence.items()
        if table_name in candidates.get("table", [])
    }
    for table_name, evidence in candidate_evidence.items():
        score_text = f", score={evidence.score:.4f}" if evidence.score is not None else ""
        notes.append(
            f"元数据候选证据: table={table_name}, source={evidence.source.value}, "
            f"status={evidence.validation_status.value}{score_text}。"
        )
    #entities 是解析后可继续使用的实体
    # metadata_candidates 是候选表列表 只有一个key -> table即fullTableName, value是元数据的候选表名
    # metadata_candidate_profiles 是候选表画像 key -> fullTableName, value -> 表的完整结构
    # metadata_notes 是元数据解析过程说明。
    return {
        "entities": entities,
        "metadata_candidates": candidates,
        "metadata_candidate_profiles": candidate_profiles,
        "metadata_candidate_evidence": candidate_evidence,
        "metadata_notes": notes,
        "requested_table": requested_table,
    }


def _authorize_context(state: PlannerState) -> PlannerState:
    """Authorize the requested intent against resolved table resources.

    核心职责：
    - 把 Agent 意图转换为企业权限系统可识别的 action。
    - 使用候选表的 domain、biz_line 画像执行 subject-action-resource 鉴权。
    - 在进入槽位后校验前删除无权候选，避免候选表名通过澄清结果或 trace 泄露。
    - 输出可审计 decision；当前使用 YAML Provider，生产可替换为 IAM/Ranger HTTP Provider。

    权限节点负责提前阻断和减少无效计划，真实工具执行端仍必须使用用户身份再次鉴权，
    Planner 的判断不能替代 TiDB、Milvus、Neo4j 服务端的最终安全边界。
    """
    context = state.get("access_context") or AccessContext(user_id="anonymous", roles=[])
    action = _authorization_action(state.get("intent", IntentType.UNKNOWN))
    provider = YamlAuthorizationProvider()
    candidates = state.get("metadata_candidates", {})
    table_candidates = candidates.get("table", [])
    profiles = state.get("metadata_candidate_profiles", {})
    evidence = state.get("metadata_candidate_evidence", {})

    decisions: list[AuthorizationDecision] = []
    authorized_tables: list[str] = []
    denied_tables: list[str] = []

    # 有表候选时逐表鉴权。权限过滤可能把多个技术候选收敛为一个用户可见候选。
    for table_name in table_candidates:
        resource = {"full_table_name": table_name, **profiles.get(table_name, {})}
        decision = provider.authorize(context, action, resource)
        decisions.append(decision)
        if decision.allowed:
            authorized_tables.append(table_name)
        else:
            denied_tables.append(table_name)

    # 元数据搜索可能在 Planner 阶段还没有具体候选，此时按用户问题中的业务域做范围鉴权。
    if not table_candidates:
        entities = state["entities"]
        scope_resource = {
            "domain": entities.domain.value if entities.domain else None,
            "biz_line": entities.biz_line,
        }
        decisions.append(provider.authorize(context, action, scope_resource))

    authorized = bool(decisions) and any(decision.allowed for decision in decisions)
    filtered_candidates = {**candidates}
    if "table" in candidates:
        filtered_candidates["table"] = authorized_tables
    filtered_profiles = {name: profiles[name] for name in authorized_tables if name in profiles}
    filtered_evidence = {name: evidence[name] for name in authorized_tables if name in evidence}

    entities = state["entities"]
    if len(authorized_tables) == 1:
        entities = entities.model_copy(update={"table": TableIdentifier.parse(authorized_tables[0])})
    elif table_candidates and not authorized_tables:
        # Preserve only a table identifier explicitly supplied by the user; never expose a discovered denied table.
        entities = entities.model_copy(update={"table": state.get("requested_table")})

    metadata_notes = _remove_denied_resource_notes(state.get("metadata_notes", []), denied_tables)
    allowed_count = len(authorized_tables)
    denied_count = len(denied_tables)
    if authorized:
        notes = [
            f"权限校验: action={action} 通过，允许候选={allowed_count}，过滤候选={denied_count}。",
            f"权限审计: user={context.user_id}, audit_ids={[decision.audit_id for decision in decisions]}。",
        ]
    else:
        reason_codes = sorted({decision.reason_code for decision in decisions})
        notes = [
            f"权限校验: action={action} 未通过，无可访问资源或操作权限。",
            f"权限审计: user={context.user_id}, reason_codes={reason_codes}, "
            f"audit_ids={[decision.audit_id for decision in decisions]}。",
        ]

    return {
        "authorized": authorized,
        "authorization_decisions": decisions,
        "authorization_notes": notes,
        "entities": entities,
        "metadata_candidates": filtered_candidates,
        "metadata_candidate_profiles": filtered_profiles,
        "metadata_candidate_evidence": filtered_evidence,
        "metadata_notes": metadata_notes,
    }


def _authorization_action(intent: IntentType) -> str:
    """Map business intent to a stable permission action understood by policy providers."""
    return {
        IntentType.METADATA_SEARCH: "metadata:read",
        IntentType.LINEAGE_SEARCH: "lineage:read",
        IntentType.IMPACT_ANALYSIS: "impact:analyze",
    }.get(intent, "data:unknown")


def _remove_denied_resource_notes(notes: list[str], denied_resources: list[str]) -> list[str]:
    """Remove internal retrieval notes that would disclose an unauthorized physical table name."""
    if not denied_resources:
        return notes
    filtered = [note for note in notes if not any(resource in note for resource in denied_resources)]
    filtered.append(f"元数据候选解析: 已在输出前过滤 {len(denied_resources)} 个无权候选。")
    return filtered


def _decide_clarification_or_continue(state: PlannerState) -> PlannerState:
    """Decide whether the planner should ask the user for more information.

    当前节点会输出 planner_decision，LangGraph conditional edge 会根据它决定：
    - continue: 继续 build_task_plan。
    - clarify: 返回澄清结果，不生成工具计划。
    - forbidden: 返回拒绝结果，不生成工具计划。
    """
    notes: list[str] = []
    pre_validation = state.get("pre_slot_validation")
    post_validation = state.get("post_slot_validation")
    blocking_issues = [
        issue
        for validation in [pre_validation, post_validation]
        if validation
        for issue in validation.blocking_issues
    ]

    if not state.get("authorized", True):
        blocking_issues.append(
            _slot_issue(
                slot_name="authorization",
                issue_type=SlotIssueType.FORBIDDEN,
                message="当前用户无权限，需要拒绝或发起权限申请。",
            )
        )

    if any(issue.issue_type == SlotIssueType.FORBIDDEN for issue in blocking_issues):
        decision = "forbidden"
        notes.append("澄清决策: 命中权限阻断，返回拒绝结果。")
    elif blocking_issues and state.get("clarification_round", 0) >= state.get("max_clarification_rounds", 3):
        decision = "handoff"
        notes.append("澄清决策: 已达到最大澄清轮数，转交人工数据服务台。")
    elif blocking_issues:
        decision = "clarify" # 澄清的意思
        notes.extend(f"澄清决策: {issue.message}" for issue in blocking_issues)
    else:
        decision = "continue"
        notes.append("澄清决策: 关键信息充分，继续生成任务计划。")
    return {"planner_decision": decision, "clarification_notes": notes}


def _route_after_clarification_decision(state: PlannerState) -> str:
    return state.get("planner_decision", "continue")


def _route_after_trace(state: PlannerState) -> str:
    """Pause only clarification branches; completed, forbidden and handoff branches return normally."""
    return "await_clarification" if state.get("planner_decision") == "clarify" else "return"


def _build_task_plan(state: PlannerState) -> PlannerState:
    result = build_task_plan(
        question=state["question"],
        intent=state.get("intent", IntentType.UNKNOWN),
        confidence=state.get("confidence", 0.0),
        entities=state["entities"],
    )
    # 📌 此种写法是什么意思  增加*的 拆解数组
    result.notes = [
        *state.get("routing_notes", []),
        *state.get("normalization_notes", []),
        *_slot_validation_notes(state),
        *state.get("metadata_notes", []),
        *state.get("authorization_notes", []),
        *state.get("clarification_notes", []),
        *result.notes,
    ]
    result.normalized_terms = state.get("normalized_terms", [])
    result.normalization_traces = state.get("normalization_traces", [])
    result.clarification_history = state.get("clarification_history", [])
    return {"result": result}


def _return_clarification_result(state: PlannerState) -> PlannerState:
    """把阻断型槽位问题封装为可展示、可恢复的澄清结果。

    核心职责：
    1. 收集 pre/post validation 产生的全部阻断问题并确定主问题。
    2. 生成前端可直接渲染的 ClarificationRequest，同时保留其余 pending issues。
    3. 返回 task_steps=[]，保证信息不充分时不会提前调用 TiDB、Milvus 或 Neo4j。
    4. 汇总路由、标准化、槽位、元数据和权限 notes，形成完整决策证据。

    边界说明：
    本节点只“准备卡片”，不负责暂停和处理回答。attach_trace 完成审计信息后，
    _await_clarification_response 才调用 interrupt() 持久化暂停。

    面试总结：
    将“生成交互协议”和“等待人工输入”拆开，可以保证 interrupt 前的状态已经完整落盘，
    同时让卡片构建逻辑保持确定性、易测试。
    """
    # 原子步骤 1：统一收集、跨阶段去重并排序，issues[0] 才是本轮最值得询问的问题。
    issues = _all_slot_issues(state)
    # 原子步骤 2：把内部 SlotIssue 转换成带候选项、版本号和会话 ID 的交互卡片。
    clarification_request = _build_clarification_request(state, issues)
    # 原子步骤 3：构造“暂停态”结果；明确清空 task_steps，防止错误执行工具。
    result = PlanningResult(
        question=state["question"],
        intent=state.get("intent", IntentType.UNKNOWN),
        confidence=min(state.get("confidence", 0.0), 0.62),
        entities=state["entities"],
        task_steps=[],
        need_clarification=True,
        clarification_question=clarification_request.question,
        clarification_request=clarification_request,
        pending_clarification_issues=issues[1:],
        # notes 是可观测性信息，不作为前端提交澄清答案的协议字段。
        notes=[
            *state.get("routing_notes", []),
            *state.get("normalization_notes", []),
            *_slot_validation_notes(state),
            *state.get("metadata_notes", []),
            *state.get("authorization_notes", []),
            *state.get("clarification_notes", []),
        ],
    )
    # 原子步骤 4：保留实体标准化证据和历史人工确认，恢复后可解释值的来源。
    result.normalized_terms = state.get("normalized_terms", [])
    result.normalization_traces = state.get("normalization_traces", [])
    result.clarification_history = state.get("clarification_history", [])
    return {"result": result}


def _return_forbidden_result(state: PlannerState) -> PlannerState:
    """Build a PlanningResult for authorization-blocked branches."""
    result = PlanningResult(
        question=state["question"],
        intent=state.get("intent", IntentType.UNKNOWN),
        confidence=min(state.get("confidence", 0.0), 0.5),
        entities=state["entities"],
        task_steps=[],
        need_clarification=False,
        clarification_question=None,
        notes=[
            *state.get("routing_notes", []),
            *state.get("normalization_notes", []),
            *_slot_validation_notes(state),
            *state.get("metadata_notes", []),
            *state.get("authorization_notes", []),
            *state.get("clarification_notes", []),
        ],
    )
    result.normalized_terms = state.get("normalized_terms", [])
    result.normalization_traces = state.get("normalization_traces", [])
    result.clarification_history = state.get("clarification_history", [])
    return {"result": result}


def _return_handoff_result(state: PlannerState) -> PlannerState:
    """达到最大澄清轮数后停止自动追问并生成安全的人工转交结果。

    核心职责：保留尚未解决的问题和已确认历史，清空工具计划，并通过
    handoff_required/handoff_reason 告诉上层系统创建人工数据服务工单。

    面试总结：无限澄清既伤害用户体验，也会增加模型和检索成本；生产系统必须设置
    轮数上限和人工兜底，而不是让 Agent 无休止重试。
    """
    result = PlanningResult(
        question=state["question"],
        intent=state.get("intent", IntentType.UNKNOWN),
        confidence=min(state.get("confidence", 0.0), 0.5),
        entities=state["entities"],
        task_steps=[],
        need_clarification=False,
        clarification_question=None,
        handoff_required=True,
        handoff_reason="已达到最大澄清轮数，请转交人工数据服务台确认数据资产。",
        pending_clarification_issues=_all_slot_issues(state),
        clarification_history=state.get("clarification_history", []),
        notes=[
            *state.get("routing_notes", []),
            *state.get("normalization_notes", []),
            *_slot_validation_notes(state),
            *state.get("metadata_notes", []),
            *state.get("authorization_notes", []),
            *state.get("clarification_notes", []),
        ],
    )
    result.normalized_terms = state.get("normalized_terms", [])
    result.normalization_traces = state.get("normalization_traces", [])
    return {"result": result}


def _await_clarification_response(state: PlannerState) -> PlannerState:
    """暂停 LangGraph，接收一次人工回答，并生成重新入图所需的状态增量。

    核心职责：
    1. 读取上一节点已经构造好的 ClarificationRequest。
    2. 通过 interrupt() 暂停当前 thread，等待 Command(resume=...)。
    3. 校验响应协议并提取服务端认可的标准值。
    4. 将用户确认值写回 typed entities，记录确认历史、轮数、版本和幂等键。

    关键约束：
    LangGraph 恢复时会从本节点开头重新执行，resume payload 会成为 interrupt() 的返回值。
    因此 interrupt 前不能发送消息、写业务表或执行其他不可幂等副作用。

    面试总结：该节点只更新 PlannerState，后续固定回到 normalize_entities，确保人工确认值
    仍需重新通过元数据事实校验、权限校验和 post slot validation。
    """
    request = state["result"].clarification_request
    if request is None:
        raise ClarificationProtocolError("澄清分支缺少 ClarificationRequest。")

    # interrupt payload 必须可 JSON 序列化，便于 API/前端跨进程传递。
    payload = interrupt(request.model_dump(mode="json"))
    # Pydantic 在业务逻辑前完成字段类型和必填项校验。
    response = ClarificationResponse.model_validate(payload)
    _validate_clarification_response(request, response)
    previous_context = state["trace_context"]
    resume_context = _new_trace_context(
        thread_id=previous_context.thread_id,
        parent_run_id=previous_context.run_id,
    )
    _record_safely(get_trace_recorder().start_run, resume_context)
    try:
        confirmed_value = _confirmed_response_value(request, response)
        entities, intent = _merge_clarification_answer(
            entities=state["entities"],
            intent=state.get("intent", IntentType.UNKNOWN),
            slot_name=request.slot_name,
            value=confirmed_value,
        )
    except Exception:
        _record_safely(get_trace_recorder().finish_run, resume_context, AgentRunStatus.FAILED)
        raise
    # 只有成功校验并写回的回答才消耗一次澄清轮数。
    clarification_round = state.get("clarification_round", 0) + 1
    history = [
        *state.get("clarification_history", []),
        ClarificationAnswerRecord(
            slot_name=request.slot_name,
            value=confirmed_value,
            clarification_round=clarification_round,
            idempotency_key=response.idempotency_key,
        ),
    ]
    return {
        "entities": entities,
        "intent": intent,
        "clarification_round": clarification_round,
        "state_version": state.get("state_version", 1) + 1,
        "clarification_history": history,
        "processed_idempotency_keys": [
            *state.get("processed_idempotency_keys", []),
            response.idempotency_key,
        ],
        "confirmed_slots": _merge_preserve_order([*state.get("confirmed_slots", []), request.slot_name]),
        # 每次 resume 都是新的 Run/Trace，通过 parent_run_id 连接上一轮，thread_id 保持不变。
        "trace_context": resume_context,
        "node_traces": [],
        "trace_events": [],
    }


def _validate_clarification_response(
    request: ClarificationRequest,
    response: ClarificationResponse,
) -> None:
    """在修改状态前校验澄清回答的会话归属、时效性和候选完整性。

    核心职责：
    - thread_id 防止把 A 会话的答案写入 B 会话。
    - clarification_id 防止回答错误卡片。
    - state_version 实现乐观锁，拒绝已经过期的页面提交。
    - option_id 必须来自服务端返回的授权候选，且 value 必须与候选绑定值一致。

    面试总结：前端传回的 value 不可信，尤其候选表可能受权限控制；服务端必须重新验证
    option_id 与 value 的绑定关系，不能把 UI 卡片当成安全边界。
    """
    if response.thread_id != request.thread_id:
        raise ClarificationProtocolError("澄清回答的 thread_id 与当前会话不一致。")
    if response.clarification_id != request.clarification_id:
        raise ClarificationProtocolError("澄清卡片已过期或 clarification_id 不匹配。")
    if response.state_version != request.state_version:
        raise ClarificationProtocolError("澄清卡片 state_version 已过期，请刷新后重试。")

    if response.option_id:
        option = next((item for item in request.options if item.option_id == response.option_id), None)
        if option is None:
            raise ClarificationProtocolError("提交的 option_id 不属于当前授权候选集。")
        if response.value != option.value:
            raise ClarificationProtocolError("option_id 与 value 不一致。")
    elif request.options and not request.allow_custom_value:
        raise ClarificationProtocolError("当前澄清卡片必须选择一个有效选项。")


def _confirmed_response_value(
    request: ClarificationRequest,
    response: ClarificationResponse,
) -> str:
    """从服务端卡片中解析最终确认值，而不是直接信任客户端重复提交的 value。

    有 option_id 时，以原 ClarificationRequest 中保存的 option.value 为事实值；只有允许
    自定义输入且未选择 option 时，才清洗并采用用户文本。

    面试总结：这是典型的 server-side canonicalization，可防止客户端篡改候选值。
    """
    if response.option_id:
        option = next(item for item in request.options if item.option_id == response.option_id)
        return option.value
    return response.value.strip()


def _merge_clarification_answer(
    entities: ExtractedEntities,
    intent: IntentType,
    slot_name: str,
    value: str,
) -> tuple[ExtractedEntities, IntentType]:
    """把字符串回答转换为强类型槽位，并以不可变复制方式写回实体状态。

    核心职责：
    - intent 写回 Planner 顶层路由状态。
    - table 使用 TableIdentifier.parse 支持一/二/三段式名称。
    - domain、data_layer、direction、operation 必须通过枚举构造，阻止非法工具参数。
    - 使用 model_copy(update=...) 避免原地修改 checkpoint 中的历史对象。

    面试总结：人工输入的置信度高，但不等于格式天然合法；仍需先做 typed conversion，
    再回到标准 Planner 流程验证资产存在性和权限。
    """
    try:
        if slot_name == "intent":
            return entities, IntentType(value)
        if slot_name == "table":
            return entities.model_copy(update={"table": TableIdentifier.parse(value)}), intent
        if slot_name == "domain":
            return entities.model_copy(update={"domain": DomainType(value)}), intent
        if slot_name == "data_layer":
            return entities.model_copy(update={"data_layer": DataLayer(value.upper())}), intent
        if slot_name == "biz_line":
            return entities.model_copy(update={"biz_line": value}), intent
        if slot_name == "lineage_direction":
            return entities.model_copy(update={"lineage_direction": LineageDirection(value)}), intent
        if slot_name == "operation":
            return entities.model_copy(update={"operation": OperationType(value)}), intent
    except (ValueError, IndexError) as exc:
        raise ClarificationProtocolError(f"澄清回答无法转换为槽位 {slot_name} 的合法值。") from exc
    raise ClarificationProtocolError(f"暂不支持写回澄清槽位 {slot_name}。")


def _validate_task_plan(state: PlannerState) -> PlannerState:
    """Validate generated task plan before returning it.

    生产级 check_task_plan 不能只看工具名，还要覆盖：
    1. DAG 结构
    2. 工具 action 注册关系
    3. 参数 schema
    4. intent 与工具组合契约
    5. 元数据候选状态
    6. 执行策略边界
    """
    result = state["result"]
    notes = [
        *_validate_dag_structure(result),
        *_validate_tool_actions(result),
        *_validate_action_params(result),
        *_validate_intent_tool_contract(result),
        *_validate_metadata_resolution_status(state),
        *_validate_execution_policy(result),
    ]
    if not notes:
        notes.append("计划校验: DAG、工具 action、参数 schema、意图契约、元数据状态和执行策略均通过。")
    return {"plan_validation_notes": notes}


def _attach_trace(state: PlannerState) -> PlannerState:
    """Finalize the current Agent Run and attach stable identifiers to PlanningResult.

    trace_id/run_id 已由入口 init_trace_context 创建；本节点不再临时生成 ID，只负责根据
    最终业务分支关闭 Run。节点异常无法到达这里时，由 traced_node 负责以 failed 关闭。
    """
    context = state["trace_context"]
    run_status = _resolve_trace_run_status(state)
    result = state["result"]
    result.thread_id = context.thread_id
    result.trace_id = context.trace_id
    result.run_id = context.run_id
    result.parent_run_id = context.parent_run_id
    event = TraceEvent(
        event_id=f"event-{uuid4().hex}",
        trace_id=context.trace_id,
        run_id=context.run_id,
        node_name="attach_trace",
        event_type="RUN_FINALIZED",
        reason_code=run_status.value,
        attributes={"planner_decision": state.get("planner_decision", "continue")},
        created_at=_utc_now(),
    )
    recorder = get_trace_recorder()
    _record_safely(recorder.record_event, event)
    _record_safely(recorder.finish_run, context, run_status)
    trace_notes = [
        f"Trace: trace_id={context.trace_id}, run_id={context.run_id}",
        f"Trace: parent_run_id={context.parent_run_id or 'none'}, run_status={run_status.value}",
        f"Trace: intent={state.get('intent', IntentType.UNKNOWN).value}, confidence={state.get('confidence', 0.0):.2f}",
        f"Trace: planner_version={context.planner_version}",
    ]
    result.notes = [*result.notes, *state.get("plan_validation_notes", []), *trace_notes]
    return {
        "trace_id": context.trace_id,
        "trace_notes": trace_notes,
        "trace_events": [*state.get("trace_events", []), event],
        "result": result,
    }


def _resolve_trace_run_status(state: PlannerState) -> AgentRunStatus:
    """Map deterministic Planner branch decisions to one terminal Agent Run status."""
    decision = state.get("planner_decision", "continue")
    return {
        "clarify": AgentRunStatus.INTERRUPTED,
        "forbidden": AgentRunStatus.FORBIDDEN,
        "handoff": AgentRunStatus.HANDOFF,
    }.get(decision, AgentRunStatus.COMPLETED)


def _return_planning_result(state: PlannerState) -> PlannerState:
    return {"result": state["result"]}


TOOL_ACTION_REGISTRY: dict[tuple[str, str], dict[str, Any]] = {
    ("tidb_metadata", "filter_tables"): {"required": [], "allowed": {"biz_line", "domain", "data_layer", "topic_keywords"}},
    ("tidb_metadata", "resolve_table"): {
        "required": ["table"],
        "allowed": {"biz_line", "domain", "data_layer", "table", "table_parts_count"},
    },
    ("milvus_rag", "semantic_search"): {"required": ["query", "top_k"], "allowed": {"query", "top_k"}},
    ("neo4j_lineage", "lineage_search"): {
        "required": ["table", "direction", "depth"],
        "allowed": {"table", "direction", "depth", "lineage_granularity"},
    },
    ("impact_analyzer", "classify_impact"): {"required": ["operation", "direction"], "allowed": {"operation", "direction"}},
    ("impact_analyzer", "merge_lineage_and_metadata"): {
        "required": ["operation", "direction"],
        "allowed": {"operation", "direction"},
    },
    ("result_ranker", "merge_and_rank"): {"required": ["rank_by"], "allowed": {"rank_by"}},
}


def _validate_dag_structure(result: PlanningResult) -> list[str]:
    notes: list[str] = []
    step_ids = [step.step_id for step in result.task_steps]
    duplicate_step_ids = sorted({step_id for step_id in step_ids if step_ids.count(step_id) > 1})
    if duplicate_step_ids:
        notes.append(f"计划校验: step_id 重复 {duplicate_step_ids}。")

    step_id_set = set(step_ids)
    invalid_dependencies = [
        step.step_id
        for step in result.task_steps
        if any(dependency not in step_id_set for dependency in step.depends_on)
    ]
    if invalid_dependencies:
        notes.append(f"计划校验: 存在非法依赖步骤 {invalid_dependencies}。")

    if _has_cycle(result):
        notes.append("计划校验: depends_on 存在循环依赖。")
    if not notes:
        notes.append("计划校验: DAG 结构通过。")
    return notes


def _validate_tool_actions(result: PlanningResult) -> list[str]:
    notes: list[str] = []
    unknown_actions = [
        f"{step.tool_name}.{step.action}"
        for step in result.task_steps
        if (step.tool_name, step.action) not in TOOL_ACTION_REGISTRY
    ]
    if unknown_actions:
        notes.append(f"计划校验: 存在未注册工具 action {unknown_actions}。")
    else:
        notes.append("计划校验: 工具 action 注册关系通过。")
    return notes


def _validate_action_params(result: PlanningResult) -> list[str]:
    notes: list[str] = []
    for step in result.task_steps:
        schema = TOOL_ACTION_REGISTRY.get((step.tool_name, step.action))
        if not schema:
            continue
        missing = [key for key in schema["required"] if _is_missing_param(step.params.get(key))]
        extra = sorted(set(step.params) - schema["allowed"])
        if missing:
            notes.append(f"计划校验: step {step.step_id} 缺少必填参数 {missing}。")
        if extra:
            notes.append(f"计划校验: step {step.step_id} 存在未声明参数 {extra}。")
        notes.extend(_validate_param_values(step.step_id, step.params))
    if not notes:
        notes.append("计划校验: 参数 schema 通过。")
    return notes


def _validate_intent_tool_contract(result: PlanningResult) -> list[str]:
    actions = {(step.tool_name, step.action) for step in result.task_steps}
    required_by_intent = {
        IntentType.METADATA_SEARCH: {
            ("tidb_metadata", "filter_tables"),
            ("milvus_rag", "semantic_search"),
            ("result_ranker", "merge_and_rank"),
        },
        IntentType.LINEAGE_SEARCH: {
            ("tidb_metadata", "resolve_table"),
            ("neo4j_lineage", "lineage_search"),
        },
        IntentType.IMPACT_ANALYSIS: {
            ("tidb_metadata", "resolve_table"),
            ("neo4j_lineage", "lineage_search"),
        },
    }
    required = required_by_intent.get(result.intent, set())
    missing = sorted(f"{tool}.{action}" for tool, action in required - actions)
    if missing:
        return [f"计划校验: intent={result.intent.value} 缺少必要工具组合 {missing}。"]
    return [f"计划校验: intent={result.intent.value} 工具组合契约通过。"]


def _validate_metadata_resolution_status(state: PlannerState) -> list[str]:
    result = state["result"]
    candidates = state.get("metadata_candidates", {})
    notes: list[str] = []
    if result.intent in {IntentType.LINEAGE_SEARCH, IntentType.IMPACT_ANALYSIS}:
        table_candidates = candidates.get("table", [])
        if not table_candidates:
            notes.append("计划校验: 血缘/影响分析缺少表元数据候选。")
        elif len(table_candidates) > 1:
            notes.append("计划校验: 表元数据候选不唯一，生产环境应先澄清或消歧。")
        else:
            notes.append("计划校验: 表元数据候选状态通过。")

    if not notes:
        notes.append("计划校验: 当前计划无需额外元数据候选校验。")
    return notes


def _validate_execution_policy(result: PlanningResult) -> list[str]:
    notes: list[str] = []
    for step in result.task_steps:
        depth = step.params.get("depth")
        top_k = step.params.get("top_k")
        if isinstance(depth, int) and not 1 <= depth <= 5:
            notes.append(f"计划校验: step {step.step_id} depth={depth} 超出允许范围 1-5。")
        if isinstance(top_k, int) and not 1 <= top_k <= 50:
            notes.append(f"计划校验: step {step.step_id} top_k={top_k} 超出允许范围 1-50。")
    if not notes:
        notes.append("计划校验: 执行策略边界通过。")
    return notes


def _validate_param_values(step_id: int, params: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    direction = params.get("direction")
    if direction is not None and direction not in {"upstream", "downstream", "both"}:
        notes.append(f"计划校验: step {step_id} direction={direction} 非法。")
    rank_by = params.get("rank_by")
    if rank_by is not None and not isinstance(rank_by, list):
        notes.append(f"计划校验: step {step_id} rank_by 必须是 list。")
    return notes


def _is_missing_param(value: Any) -> bool:
    return value is None or value == "" or value == []


def _has_cycle(result: PlanningResult) -> bool:
    graph = {step.step_id: set(step.depends_on) for step in result.task_steps}
    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(step_id: int) -> bool:
        if step_id in visiting:
            return True
        if step_id in visited:
            return False
        visiting.add(step_id)
        for dependency in graph.get(step_id, set()):
            if visit(dependency):
                return True
        visiting.remove(step_id)
        visited.add(step_id)
        return False

    return any(visit(step_id) for step_id in graph)


def _mock_table_candidates(table: TableIdentifier) -> list[str]:
    """Generate mock table candidates for local study and tests.

    一段式表名天然不唯一，例如 userInfo 可能同时存在于 dwd 和 dim；
    两段式/三段式表名已经携带 schema/catalog，当前 mock 直接作为候选返回。
    """
    if table.parts_count == 1:
        return [f"dwd.{table.table_name}", f"dim.{table.table_name}"]
    return [table.raw]


def _filter_table_candidates_by_context(candidates: list[str], entities: ExtractedEntities) -> list[str]:
    """Use available table-level context to narrow mock metadata candidates.

    当前先用 data_layer 做最小可解释消歧：`userInfo + DWD` 会优先收敛到
    `dwd.userInfo`。真实生产里这里会改成 TiDB/DataCatalog 的评分和排序结果。

    设计原则：
    - 能用明确上下文缩小候选就缩小，降低澄清率。
    - 如果上下文过滤不到结果，就保留原候选，交给 post_validate_slots 判断是否冲突。
    - 不在这里“拍脑袋选一个”，避免错误血缘分析。
    """
    if not entities.data_layer:
        return candidates
    layer_prefix = f"{entities.data_layer.value.lower()}."
    filtered = [candidate for candidate in candidates if candidate.lower().startswith(layer_prefix)]
    return filtered or candidates


MOCK_TABLE_PROFILES: dict[str, dict[str, str]] = {
    "dwd.orderInfo": {"domain": DomainType.TRANSACTION.value, "data_layer": DataLayer.DWD.value, "biz_line": "安逸花"},
    "dwd.order_info": {"domain": DomainType.TRANSACTION.value, "data_layer": DataLayer.DWD.value, "biz_line": "安逸花"},
    "dwd.payment_detail": {"domain": DomainType.MARKETING.value, "data_layer": DataLayer.DWD.value, "biz_line": "安逸花"},
    "dwd.pay_record": {"domain": DomainType.MARKETING.value, "data_layer": DataLayer.DWD.value, "biz_line": "安逸花"},
    "dwd.userInfo": {"domain": DomainType.MARKETING.value, "data_layer": DataLayer.DWD.value, "biz_line": "安逸花"},
    "dim.userInfo": {"domain": DomainType.USER.value, "data_layer": DataLayer.DIM.value, "biz_line": "安逸花"},
}


def _candidate_names(candidates: list[MetadataCandidate]) -> list[str]:
    """Convert repository candidates to the table-name list used by PlannerState.

    原子职责：
    PlannerState 里保留轻量的 `metadata_candidates["table"] = list[str]`，
    便于 post_validate_slots 判断候选数量，也便于最终 notes 展示。
    """
    return _merge_preserve_order([candidate.full_table_name for candidate in candidates])


def _candidate_profiles(candidates: list[MetadataCandidate]) -> dict[str, dict[str, str | None]]:
    """Convert repository candidates to profile map for post slot validation.

    原子职责：
    候选名只解决“有哪些表”，profile 解决“这些表属于哪个域/分层/业务线”。
    post_validate_slots 会用 profile 做跨槽位一致性校验。
    """
    # python中的字典推导 返回一个dict  key是遍历的candidate中的full_table_name, value是candidate.profile()
    return {candidate.full_table_name: candidate.profile() for candidate in candidates}


def _record_candidate_evidence(
    evidence_by_table: dict[str, MetadataCandidateEvidence],
    candidates: list[MetadataCandidate],
    source: MetadataCandidateSource,
    status: MetadataValidationStatus,
) -> None:
    """Record authoritative repository evidence without replacing a stronger existing source."""
    for candidate in candidates:
        _upsert_candidate_evidence(
            evidence_by_table,
            MetadataCandidateEvidence(
                full_table_name=candidate.full_table_name,
                source=source,
                validation_status=status,
            ),
        )


def _record_fallback_evidence(
    evidence_by_table: dict[str, MetadataCandidateEvidence],
    table_names: list[str],
) -> None:
    """Mark mock/config candidates as non-authoritative so post validation can fail closed."""
    for table_name in table_names:
        _upsert_candidate_evidence(
            evidence_by_table,
            MetadataCandidateEvidence(
                full_table_name=table_name,
                source=MetadataCandidateSource.MOCK_FALLBACK,
                validation_status=MetadataValidationStatus.FALLBACK,
            ),
        )


def _record_milvus_validated_evidence(
    evidence_by_table: dict[str, MetadataCandidateEvidence],
    response: Any,
    validated_candidates: list[MetadataCandidate],
) -> None:
    """Attach Milvus rank/score only to candidates that were validated again by MySQL."""
    recalled = response.candidates
    recalled_by_name = {candidate.full_table_name: candidate for candidate in recalled}
    rank_by_name = {candidate.full_table_name: rank for rank, candidate in enumerate(recalled, start=1)}
    gap_by_name: dict[str, float | None] = {}
    for index, candidate in enumerate(recalled):
        next_score = recalled[index + 1].score if index + 1 < len(recalled) else None
        gap_by_name[candidate.full_table_name] = (
            abs(candidate.score - next_score) if next_score is not None else None
        )

    for candidate in validated_candidates:
        recalled_candidate = recalled_by_name.get(candidate.full_table_name)
        if recalled_candidate is None:
            continue
        _upsert_candidate_evidence(
            evidence_by_table,
            MetadataCandidateEvidence(
                full_table_name=candidate.full_table_name,
                source=MetadataCandidateSource.MILVUS_MYSQL_VALIDATED,
                validation_status=MetadataValidationStatus.VALIDATED,
                score=recalled_candidate.score,
                rank=rank_by_name[candidate.full_table_name],
                score_gap_to_next=gap_by_name[candidate.full_table_name],
                retrieval_mode=response.retrieval_mode,
            ),
        )


def _upsert_candidate_evidence(
    evidence_by_table: dict[str, MetadataCandidateEvidence],
    incoming: MetadataCandidateEvidence,
) -> None:
    """Keep the most trustworthy evidence when multiple recall paths hit the same table."""
    existing = evidence_by_table.get(incoming.full_table_name)
    if existing is None or _evidence_priority(incoming) > _evidence_priority(existing):
        evidence_by_table[incoming.full_table_name] = incoming


def _evidence_priority(evidence: MetadataCandidateEvidence) -> tuple[int, int]:
    status_priority = {
        MetadataValidationStatus.FALLBACK: 0,
        MetadataValidationStatus.UNVERIFIED: 1,
        MetadataValidationStatus.VALIDATED: 2,
    }
    source_priority = {
        MetadataCandidateSource.MOCK_FALLBACK: 0,
        MetadataCandidateSource.MILVUS_MYSQL_VALIDATED: 1,
        MetadataCandidateSource.MYSQL_TABLE_TERM: 2,
        MetadataCandidateSource.MYSQL_IDENTIFIER: 3,
    }
    return status_priority[evidence.validation_status], source_priority[evidence.source]


def _validated_candidate_names(
    evidence_by_table: dict[str, MetadataCandidateEvidence],
) -> list[str]:
    """Return only candidates proven to exist in the authoritative metadata store."""
    return [
        table_name
        for table_name, evidence in evidence_by_table.items()
        if evidence.validation_status == MetadataValidationStatus.VALIDATED
    ]


def _mock_candidate_profiles(candidates: list[str]) -> dict[str, dict[str, str | None]]:
    """Build profile map for fallback mock candidates.

    原子职责：
    当 MySQL 不可用时，mock 候选也要能参与一致性校验，否则测试和本地学习会丢失
    domain/data_layer/biz_line 冲突判断。
    """
    return {candidate: MOCK_TABLE_PROFILES[candidate] for candidate in candidates if candidate in MOCK_TABLE_PROFILES}


def _table_term_lookup_values(state: PlannerState) -> list[str]:
    """Build MySQL lookup values from normalized table terms.

    meta_table_ext supports both normalized_term and term_value, so this method passes both
    canonical terms such as order_info and raw user terms such as 订单信息表.

    原子职责：
    normalize_entities 会产出 `订单信息表 -> order_info`。真实查询时不能只查 canonical，
    因为元数据字典里可能只维护了原始别名；也不能只查原词，因为生产词典可能只存标准术语。
    """
    values = list(state.get("table_term_candidates", {}).keys())
    for term in state.get("normalized_terms", []):
        if term.term_type == NormalizedTermType.TABLE_TERM:
            values.extend([term.text, term.canonical])
    return _merge_preserve_order(values)


def _build_semantic_table_query(
    question: str,
    entities: ExtractedEntities,
    table_term_candidates: dict[str, list[str]],
) -> str | None:
    """Build a Milvus query only when the question contains a usable table signal.

    核心职责：
    - 明确的两段式/三段式技术表名由 MySQL 事实查询处理，不额外构造语义检索请求。
    - 一段式表名、表级业务术语或“广告投放转化率相关表”可以形成语义检索请求。
    - 只有“查询下游依赖”但没有表资产或业务描述时返回 None，防止 Milvus 猜表。

    这里仅判断“有没有可检索的业务语义”，不决定是否真正调用 Milvus；最终路由由
    `_should_use_semantic_recall` 结合 MySQL 候选结果决定。
    """
    if entities.table and entities.table.parts_count >= 2:
        return None

    has_table_context = "表" in question or bool(table_term_candidates)
    has_business_signal = bool(
        table_term_candidates
        or entities.topic_keywords
        or entities.domain
        or entities.data_layer
        or re.search(
            r"[\u4e00-\u9fff]{2,}(?:指标|转化率|明细|汇总|画像|投放|订单|支付|用户)",
            question,
        )
    )
    if not has_table_context or not has_business_signal:
        return None
    return question.strip() or None


def _should_use_semantic_recall(
    entities: ExtractedEntities,
    table_candidates: list[str],
    semantic_query: str | None,
) -> bool:
    """Decide whether metadata resolution should invoke Milvus semantic recall.

    核心职责：按照表实体确定性和 MySQL 查询结果进行成本可控的检索路由。

    决策规则：
    1. 没有有效 semantic_query：不调用 Milvus。
    2. 没有技术表名、只有业务描述或业务术语：调用 Milvus 主动召回候选。
    3. 两段式/三段式技术表名：无论 MySQL 是否命中，都不调用 Milvus；事实库的
       “不存在”不能被向量库的近似结果覆盖。
    4. 一段式技术表名：MySQL 已经找到候选时不调用；MySQL 零结果时才用 Milvus 补召回。

    `table_candidates` 必须是经过 MySQL/meta_table 或结构化术语查询得到的当前候选，
    不能传入尚未经过事实校验的 Milvus 原始结果。
    """
    if not semantic_query:
        return False

    table = entities.table
    if table is None:
        return True
    if table.parts_count >= 2:
        return False

    # 兜底 啥都没查出来 & 是语义查询 走mivlus
    return not table_candidates


def _validate_candidate_trust(
    table_name: str,
    evidence: MetadataCandidateEvidence | None,
) -> list[SlotIssue]:
    """Require authoritative metadata evidence before lineage or impact execution."""
    if evidence is None:
        return [
            _slot_issue(
                slot_name="table",
                issue_type=SlotIssueType.INVALID,
                message=f"候选表 {table_name} 缺少来源和事实校验证据，不能进入执行链路。",
            )
        ]
    if evidence.validation_status != MetadataValidationStatus.VALIDATED:
        return [
            _slot_issue(
                slot_name="table",
                issue_type=SlotIssueType.INVALID,
                message=(
                    f"候选表 {table_name} 仅来自 {evidence.source.value}，"
                    f"状态为 {evidence.validation_status.value}，尚未通过 MySQL 元数据事实校验。"
                ),
            )
        ]
    return []


def _validate_semantic_candidate_confidence(
    table_name: str,
    evidence: MetadataCandidateEvidence | None,
) -> list[SlotIssue]:
    """Apply configurable score and score-gap policies to Milvus-origin candidates.

    不同检索模式的原始分数尺度不同，因此默认阈值为 0，生产上线前应通过离线评测集
    标定 DATA_AGENT_MILVUS_MIN_SCORE 和 DATA_AGENT_MILVUS_MIN_SCORE_GAP。
    """
    if evidence is None or evidence.source != MetadataCandidateSource.MILVUS_MYSQL_VALIDATED:
        return []
    if evidence.score is None:
        return [
            _slot_issue(
                slot_name="table",
                issue_type=SlotIssueType.LOW_CONFIDENCE,
                message=f"语义候选表 {table_name} 缺少召回分数，不能自动选择。",
            )
        ]

    min_score = float(os.getenv("DATA_AGENT_MILVUS_MIN_SCORE", "0"))
    min_gap = float(os.getenv("DATA_AGENT_MILVUS_MIN_SCORE_GAP", "0"))
    issues: list[SlotIssue] = []
    if evidence.score < min_score:
        issues.append(
            _slot_issue(
                slot_name="table",
                issue_type=SlotIssueType.LOW_CONFIDENCE,
                message=(
                    f"语义候选表 {table_name} 的 score={evidence.score:.4f} "
                    f"低于自动选择阈值 {min_score:.4f}。"
                ),
            )
        )
    if evidence.score_gap_to_next is not None and evidence.score_gap_to_next < min_gap:
        issues.append(
            _slot_issue(
                slot_name="table",
                issue_type=SlotIssueType.LOW_CONFIDENCE,
                message=(
                    f"语义候选表 {table_name} 与下一候选分差 {evidence.score_gap_to_next:.4f} "
                    f"低于自动选择阈值 {min_gap:.4f}。"
                ),
            )
        )
    return issues


def _validate_executable_table_identity(
    entities: ExtractedEntities,
    candidate_name: str,
) -> list[SlotIssue]:
    """Ensure downstream tools receive the unique canonical candidate, not a one-part alias."""
    table = entities.table
    if table is None or table.parts_count < 2 or table.raw.casefold() != candidate_name.casefold():
        actual = table.raw if table else None
        return [
            _slot_issue(
                slot_name="table",
                issue_type=SlotIssueType.INVALID,
                message=(
                    f"唯一候选为 {candidate_name}，但可执行表实体为 {actual}；"
                    "必须先回填规范化 db.table 后才能调用血缘工具。"
                ),
            )
        ]
    return []


def _validate_requested_table_identity(
    requested_table: TableIdentifier | None,
    candidate_name: str,
    profile: dict[str, str | None] | None,
) -> list[SlotIssue]:
    """Prevent an exact two/three-part identifier from resolving to another physical asset."""
    if requested_table is None or requested_table.parts_count < 2:
        return []

    candidate = TableIdentifier.parse(candidate_name)
    actual_catalog = (profile or {}).get("catalog_name") or candidate.catalog
    actual_schema = (profile or {}).get("db_name") or candidate.schema_name
    actual_table = (profile or {}).get("table_name") or candidate.table_name
    mismatches: list[str] = []
    if requested_table.catalog and _casefold(requested_table.catalog) != _casefold(actual_catalog):
        mismatches.append("catalog")
    if requested_table.schema_name and _casefold(requested_table.schema_name) != _casefold(actual_schema):
        mismatches.append("schema")
    if _casefold(requested_table.table_name) != _casefold(actual_table):
        mismatches.append("table_name")
    if not mismatches:
        return []
    return [
        _slot_issue(
            slot_name="table",
            issue_type=SlotIssueType.CONFLICT,
            message=(
                f"用户请求表 {requested_table.raw} 与事实候选 {candidate_name} "
                f"在 {mismatches} 上不一致。"
            ),
        )
    ]


def _validate_candidate_profile_completeness(
    entities: ExtractedEntities,
    table_name: str,
    profile: dict[str, str | None] | None,
) -> list[SlotIssue]:
    """Require identity fields and any user-specified governance dimensions in the profile."""
    if profile is None:
        return [
            _slot_issue(
                slot_name="table_profile",
                issue_type=SlotIssueType.INVALID,
                message=f"候选表 {table_name} 缺少权威元数据画像。",
            )
        ]

    required_fields = ["db_name", "table_name"]
    if entities.biz_line:
        required_fields.append("biz_line")
    if entities.domain:
        required_fields.append("domain")
    if entities.data_layer:
        required_fields.append("data_layer")
    missing_fields = [field_name for field_name in required_fields if not profile.get(field_name)]
    if not missing_fields:
        return []
    return [
        _slot_issue(
            slot_name="table_profile",
            issue_type=SlotIssueType.INVALID,
            message=f"候选表 {table_name} 的权威画像缺少字段 {missing_fields}。",
        )
    ]


def _casefold(value: str | None) -> str | None:
    return value.casefold() if value is not None else None


def _deduplicate_slot_issues(issues: list[SlotIssue]) -> list[SlotIssue]:
    """Collapse repeated generic/specialized findings while keeping the later, clearer message."""
    deduplicated: dict[tuple[str, SlotIssueType, str], SlotIssue] = {}
    for issue in issues:
        # MISSING 经常同时来自 required_any 和 intent 专项校验，只保留后者更明确的提示。
        message_key = "" if issue.issue_type == SlotIssueType.MISSING else issue.message
        deduplicated[(issue.slot_name, issue.issue_type, message_key)] = issue
    return list(deduplicated.values())


def _validate_cross_slot_consistency(
    entities: ExtractedEntities,
    table_candidates: list[str],
    candidate_profiles: dict[str, dict[str, str | None]],
) -> list[SlotIssue]:
    """Validate consistency between user slots and resolved table profile.

    真实企业里，这一步会读取元数据服务返回的表画像：
    - 表所属主题域
    - 表所在数仓分层
    - 表所属业务线或权限域

    当前用 mock profile 表达同样的生产逻辑：如果用户说“营销域”，但唯一候选表画像是
    “交易域”，就不能继续生成计划，必须先澄清或修正。

    注意：
    只有唯一候选表时才做一致性校验。多候选情况下，每个候选可能属于不同域/层级，
    这时优先让用户消歧，而不是提前判断冲突。
    """
    if len(table_candidates) != 1:
        return []

    table_name = table_candidates[0]
    profile = candidate_profiles.get(table_name)
    if not profile:
        return []

    issues: list[SlotIssue] = []
    if entities.domain and profile.get("domain") and entities.domain.value != profile["domain"]:
        issues.append(
            _slot_issue(
                slot_name="domain",
                issue_type=SlotIssueType.CONFLICT,
                message=f"用户输入主题域 {entities.domain.value} 与候选表 {table_name} 的主题域 {profile['domain']} 不一致。",
            )
        )
    if entities.data_layer and profile.get("data_layer") and entities.data_layer.value != profile["data_layer"]:
        issues.append(
            _slot_issue(
                slot_name="data_layer",
                issue_type=SlotIssueType.CONFLICT,
                message=f"用户输入数仓分层 {entities.data_layer.value} 与候选表 {table_name} 的分层 {profile['data_layer']} 不一致。",
            )
        )
    if entities.biz_line and profile.get("biz_line") and entities.biz_line != profile["biz_line"]:
        issues.append(
            _slot_issue(
                slot_name="biz_line",
                issue_type=SlotIssueType.CONFLICT,
                message=f"用户输入业务线 {entities.biz_line} 与候选表 {table_name} 的业务线 {profile['biz_line']} 不一致。",
            )
        )
    return issues


def _slot_issue(slot_name: str, issue_type: SlotIssueType, message: str) -> SlotIssue:
    """Create a slot issue using configured blocking policy.

    原子职责：
    把“发现了什么问题”统一包装成结构化 SlotIssue。是否阻断不写死在调用方，
    而是读取 slot_rules.yml，方便生产里按业务线调整策略。
    """
    return SlotIssue(
        slot_name=slot_name,
        issue_type=issue_type,
        message=message,
        blocking=load_slot_rule_config().is_blocking(issue_type),
    )


def _has_any_slot(slot_names: list[str], entities: ExtractedEntities, state: PlannerState) -> bool:
    """Check whether at least one configured slot has a usable value.

    原子职责：
    支持 required_any 语义。比如 lineage_search 的前置要求是 table 或 table_term
    二选一，只要其中一个存在，就允许进入元数据候选解析。
    """
    return any(_slot_has_value(slot_name, entities, state) for slot_name in slot_names)


def _slot_has_value(slot_name: str, entities: ExtractedEntities, state: PlannerState) -> bool:
    """Return whether a single slot is present and usable.

    原子职责：
    把配置里的字符串槽位名映射到真实数据来源：
    - intent/table/topic_keywords 来自 entities 或 state。
    - table_term 来自 normalize_entities 产出的 table_term_candidates。
    - table 在 post 阶段可以来自 metadata_candidates，表示已经解析出候选表。
    """
    if slot_name == "intent":
        return state.get("intent") not in {None, IntentType.UNKNOWN}
    if slot_name == "table_term":
        return bool(state.get("table_term_candidates"))
    if slot_name == "metadata_table_candidate":
        return bool(state.get("metadata_candidates", {}).get("table"))
    if slot_name == "table":
        return entities.table is not None or bool(state.get("metadata_candidates", {}).get("table"))
    value = getattr(entities, slot_name, None)
    return not _is_missing_param(value)


def _missing_slot_message(intent: IntentType) -> str:
    """Build user-facing missing-slot message by intent.

    原子职责：
    不同 intent 缺槽位时，应该问的问题不一样。血缘查询重点补表名；
    元数据搜索可以补主题域、分层、表名或业务关键词。
    """
    messages = {
        IntentType.METADATA_SEARCH: "缺少主题域、数仓分层、表名、表级业务术语或业务关键词，无法执行元数据搜索。",
        IntentType.LINEAGE_SEARCH: "缺少表名或表级业务术语，血缘查询无法定位数据资产。",
        IntentType.IMPACT_ANALYSIS: "缺少表名或表级业务术语，影响分析无法定位数据资产。",
        IntentType.UNKNOWN: "请补充你想查询元数据、血缘关系，还是表变更影响。",
    }
    return messages.get(intent, "缺少关键槽位，无法继续规划。")


def _slot_issue_notes(prefix: str, issues: list[SlotIssue]) -> list[str]:
    """Convert structured slot issues into readable notes.

    原子职责：
    SlotIssue 给系统做决策，notes 给人阅读和面试演示。这里把 issue_type、
    slot_name、blocking、message 都展开，方便追踪为什么进入澄清分支。
    """
    if not issues:
        return [f"{prefix}: 通过。"]
    return [
        f"{prefix}: {issue.issue_type.value} slot={issue.slot_name}, blocking={issue.blocking}, message={issue.message}"
        for issue in issues
    ]


def _slot_validation_notes(state: PlannerState) -> list[str]:
    """Collect pre/post slot validation notes in execution order.

    原子职责：
    最终 PlanningResult.notes 需要按流程展示，所以这里统一收集前置校验和后置校验的说明。
    """
    notes: list[str] = []
    for key in ["pre_slot_validation", "post_slot_validation"]:
        validation = state.get(key)
        if validation:
            notes.extend(validation.notes)
    return notes


def _all_slot_issues(state: PlannerState) -> list[SlotIssue]:
    """从前后置槽位校验中收集全部阻断问题，并返回确定性排序结果。

    原子职责：
    只读取 blocking_issues，非阻断告警继续保留在 notes，但不应触发人工澄清。
    pre/post 可能重复报告同一个缺失槽位，因此把原始问题统一交给
    _prioritize_clarification_issues 做跨阶段去重和排序。

    输入：PlannerState 中的 pre_slot_validation、post_slot_validation。
    输出：完整、有序的 SlotIssue 列表，第一项为本轮主问题，其余项进入 pending issues。

    面试总结：校验节点负责“发现问题”，该方法负责“汇总问题”，避免每个校验节点
    各自决定如何向用户提问。
    """
    issues = [
        issue
        for key in ["pre_slot_validation", "post_slot_validation"]
        if (validation := state.get(key))
        for issue in validation.blocking_issues
    ]
    return _prioritize_clarification_issues(issues)


def _prioritize_clarification_issues(issues: list[SlotIssue]) -> list[SlotIssue]:
    """按照槽位依赖和问题类型选择信息增益最高的问题，同时保留剩余问题。

    原子职责：
    1. 对同一个 missing 槽位跨 pre/post 去重，保留后产生的、更接近事实源的描述。
    2. 先按 slot dependency 排序，再按 issue type 排序。
    3. 不删除不同原因的 invalid/conflict，保证待处理问题和审计证据完整。

    排序依据：
    intent 决定整条工具路由，所以最先确认；table 可以通过元数据画像反向补齐
    domain/data_layer/biz_line，因此排在治理维度之前；FORBIDDEN 不属于澄清，正常情况下
    已由 forbidden 分支提前处理，所以在这里放到最后作为防御性设计。

    面试总结：每轮只问一个问题不等于简单取 issues[0]；先通过确定性优先级提升信息增益，
    可以减少澄清轮次，也避免让 LLM 自由决定关键业务流程。
    """
    deduplicated: dict[tuple[str, SlotIssueType, str], SlotIssue] = {}
    for issue in issues:
        message_key = "" if issue.issue_type == SlotIssueType.MISSING else issue.message
        deduplicated[(issue.slot_name, issue.issue_type, message_key)] = issue

    slot_priority = {
        "intent": 0,
        "table": 10,
        "table_profile": 15,
        "domain": 20,
        "data_layer": 21,
        "biz_line": 22,
        "lineage_direction": 30,
        "operation": 31,
    }
    issue_priority = {
        SlotIssueType.AMBIGUOUS: 0,
        SlotIssueType.MISSING: 1,
        SlotIssueType.INVALID: 2,
        SlotIssueType.CONFLICT: 3,
        SlotIssueType.LOW_CONFIDENCE: 4,
        SlotIssueType.FORBIDDEN: 99,
    }
    return sorted(
        deduplicated.values(),
        key=lambda issue: (
            slot_priority.get(issue.slot_name, 50),
            issue_priority.get(issue.issue_type, 50),
        ),
    )


def _build_clarification_request(state: PlannerState, issues: list[SlotIssue]) -> ClarificationRequest:
    """把最高优先级问题转换成带会话控制字段的结构化澄清卡片。

    原子职责：
    - 生成用户可读 question。
    - 选择 primary issue，并构建与它匹配的候选 options 和输入控件类型。
    - 写入 thread_id、clarification_id、state_version、当前轮次和最大轮次。
    - 只记录 pending_issue_count；完整剩余问题由 PlanningResult 单独保存。

    无 issues 时仍返回通用文本卡片，这是防御性 fallback，避免前端收到 need_clarification=true
    却没有可展示协议。

    面试总结：卡片既是 UI DTO，也是恢复协议；会话 ID 和版本字段保证后续回答能安全地
    定位到准确 checkpoint。
    """
    question = _build_clarification_question(issues)
    if not issues:
        return ClarificationRequest(
            clarification_id=f"clarify-{uuid4().hex[:12]}",
            thread_id=state.get("thread_id", "unbound-thread"),
            question=question,
            slot_name="data_asset",
            issue_type=SlotIssueType.MISSING,
            input_type=ClarificationInputType.TEXT,
            allow_custom_value=True,
            clarification_round=state.get("clarification_round", 0) + 1,
            max_clarification_rounds=state.get("max_clarification_rounds", 3),
        )

    primary_issue = issues[0]
    options = _build_clarification_options(state, primary_issue)
    return ClarificationRequest(
        clarification_id=f"clarify-{uuid4().hex[:12]}",
        thread_id=state.get("thread_id", "unbound-thread"),
        question=question,
        slot_name=primary_issue.slot_name,
        issue_type=primary_issue.issue_type,
        input_type=_clarification_input_type(primary_issue, options),
        options=options,
        allow_custom_value=primary_issue.slot_name == "table",
        pending_issue_count=max(len(issues) - 1, 0),
        state_version=state.get("state_version", 1),
        clarification_round=state.get("clarification_round", 0) + 1,
        max_clarification_rounds=state.get("max_clarification_rounds", 3),
    )


def _build_clarification_options(
    state: PlannerState,
    issue: SlotIssue,
) -> list[ClarificationOption]:
    """把已授权的表候选转换成前端可展示、服务端可验证的选项。

    原子职责：
    - 只为 table + ambiguous 场景生成单选候选，其他问题返回空列表。
    - 从候选画像补充主题域、分层、业务线和表说明，帮助用户做业务判断。
    - 从 evidence 补充来源、事实验证状态和匹配分数，支持解释与审计。
    - 使用物理表名哈希生成稳定 option_id，提交时再校验 option_id/value 绑定关系。

    安全边界：metadata_candidates 已经经过 authorize_context 过滤，本方法不得回查或补回
    未授权候选。

    面试总结：Milvus/MySQL 返回的是检索候选，本方法输出的是授权后的交互候选，两者不能
    直接等同。
    """
    if issue.slot_name != "table" or issue.issue_type != SlotIssueType.AMBIGUOUS:
        return []

    profiles = state.get("metadata_candidate_profiles", {})
    evidence_by_table = state.get("metadata_candidate_evidence", {})
    options: list[ClarificationOption] = []
    for table_name in state.get("metadata_candidates", {}).get("table", []):
        profile = profiles.get(table_name, {})
        evidence = evidence_by_table.get(table_name)
        metadata: dict[str, Any] = {
            key: profile.get(key)
            for key in ["domain", "data_layer", "biz_line"]
            if profile.get(key) is not None
        }
        if evidence:
            metadata.update(
                {
                    "candidate_source": evidence.source.value,
                    "validation_status": evidence.validation_status.value,
                    "match_score": evidence.score,
                }
            )
        options.append(
            ClarificationOption(
                option_id=f"table-{sha256(table_name.encode('utf-8')).hexdigest()[:10]}",
                label=table_name,
                value=table_name,
                description=profile.get("table_comment"),
                metadata=metadata,
            )
        )
    return options


def _clarification_input_type(
    issue: SlotIssue,
    options: list[ClarificationOption],
) -> ClarificationInputType:
    """根据问题语义确定前端控件类型，不让 LLM 猜测交互协议。

    有候选项使用 single_select；冲突确认使用 confirm；缺失值或低置信度问题使用 text。

    面试总结：控件类型属于稳定业务协议，适合规则映射而不是模型生成，可以降低前后端
    协议漂移和解析失败率。
    """
    if options:
        return ClarificationInputType.SINGLE_SELECT
    if issue.issue_type == SlotIssueType.CONFLICT:
        return ClarificationInputType.CONFIRM
    return ClarificationInputType.TEXT


def _build_clarification_question(issues: list[SlotIssue]) -> str:
    """把内部 SlotIssue 转换成用户可以直接回答的一句话。

    原子职责：
    ambiguous 明确要求选择唯一表；missing 给出符合系统解析规则的示例；conflict、invalid
    和 low_confidence 保留校验节点产生的具体原因。无问题时返回通用 fallback。

    输入约束：issues 必须已经由 _prioritize_clarification_issues 排序，因此这里读取第一项
    是明确的产品策略，而不是依赖 pre/post 校验的偶然执行顺序。

    面试总结：事实和候选由确定性代码产生，LLM 最多用于后续文案润色，不能改写 slot、
    option value 或权限过滤结果。
    """
    if not issues:
        return "请补充更明确的数据资产信息。"
    first_issue = issues[0]
    if first_issue.issue_type == SlotIssueType.AMBIGUOUS:
        return f"{first_issue.message} 请回复你要分析的唯一表名。"
    if first_issue.issue_type == SlotIssueType.MISSING:
        return f"{first_issue.message} 例如 dwd.orderInfo 或 userInfo。"
    return first_issue.message


def _normalize_text(value: str | None) -> str | None:
    """清洗普通文本实体。

    用于 biz_line 这类业务文本。这里只做轻量清洗：去掉首尾空白和常见中文标点。
    生产里更复杂的业务词归一不放在这里，而是交给 glossary/synonym 配置。
    """
    if value is None:
        return None
    cleaned = value.strip(" \t\r\n，,。；;：:")
    return cleaned or None


def _normalize_identifier_text(value: str | None) -> str | None:
    """清洗技术标识符。

    用于 table/catalog/schema 等工具参数。相比普通文本，会额外去掉反引号、
    单引号、双引号，避免用户输入 `dwd.orderInfo` 这类展示格式时污染工具调用参数。
    """
    if value is None:
        return None
    cleaned = value.strip(" \t\r\n`'\"，,。；;：:")
    return cleaned or None


def _normalize_table_identifier(table: TableIdentifier | None) -> TableIdentifier | None:
    """标准化表标识。

    当前策略：
    - catalog/schema 统一小写，便于后续元数据查询。
    - table_name 保留原样，避免破坏驼峰表名如 orderInfo。
    - 重新组装 raw，保证下游工具拿到一致格式。

    生产里这一步应读取 platform policy：Hive/Trino/PostgreSQL/MySQL 对大小写的规则不同。
    """
    if table is None:
        return None

    catalog = _normalize_identifier_text(table.catalog)
    schema_name = _normalize_identifier_text(table.schema_name)
    table_name = _normalize_identifier_text(table.table_name)
    if table_name is None:
        return None

    normalized_catalog = catalog.lower() if catalog else None
    normalized_schema = schema_name.lower() if schema_name else None
    raw_parts = [part for part in [normalized_catalog, normalized_schema, table_name] if part]
    return TableIdentifier(
        raw=".".join(raw_parts),
        catalog=normalized_catalog,
        schema_name=normalized_schema,
        table_name=table_name,
    )


def _normalize_topic_keywords(
    entities: ExtractedEntities,
    question: str,
) -> tuple[list[str], list[NormalizedTerm], list[NormalizationTrace]]:
    """标准化业务检索关键词。

    核心职责：
    - 从原始 question 中补充配置化 synonyms 命中的业务词。
    - 合并上游抽取出的 topic_keywords。
    - 去除 stopwords、主题域、数仓分层等已经结构化的词。
    - 将别名映射为 canonical term，例如 支付相关 -> 支付。
    - 生成 NormalizedTerm 和 NormalizationTrace，便于审计和评估。
    """
    config = load_normalization_config()
    stopwords = set(config.stopwords)
    if entities.domain:
        stopwords.add(entities.domain.value)
        stopwords.add(entities.domain.value.removesuffix("域"))
    if entities.data_layer:
        stopwords.add(entities.data_layer.value)
        stopwords.add(entities.data_layer.value.lower())

    normalized: list[str] = []
    terms: list[NormalizedTerm] = []
    traces: list[NormalizationTrace] = []
    candidate_keywords = [*_extract_synonym_aliases_from_question(question), *entities.topic_keywords]
    for keyword in candidate_keywords:
        cleaned = _normalize_text(keyword)
        if not cleaned or cleaned in stopwords:
            continue
        mapped = config.map_term(cleaned)
        final_keyword = mapped.canonical if mapped else cleaned
        if mapped and final_keyword not in normalized:
            terms.append(mapped)
            traces.append(
                NormalizationTrace(
                    field_name="topic_keywords",
                    before=cleaned,
                    after=final_keyword,
                    rule="term_synonym_mapping",
                    source=mapped.source,
                )
            )
        if final_keyword not in normalized:
            normalized.append(final_keyword)
    return normalized, terms, traces


def _extract_table_terms_from_question(question: str) -> tuple[list[NormalizedTerm], list[NormalizationTrace], dict[str, list[str]]]:
    """从原始问题中识别表级业务术语。

    当前阶段聚焦表级数据探查。这里把“订单信息表”“支付明细表”等业务说法映射
    到标准表级概念 table_term，再把 candidate_tables 交给元数据候选解析节点。
    """
    terms: list[NormalizedTerm] = []
    traces: list[NormalizationTrace] = []
    candidates: dict[str, list[str]] = {}
    for rule in load_normalization_config().table_terms:
        aliases = sorted([rule.display_name, rule.canonical, *rule.aliases], key=len, reverse=True)
        matched_alias = next((alias for alias in aliases if _alias_in_question(alias, question)), None)
        if not matched_alias:
            continue
        term = NormalizedTerm(
            text=matched_alias,
            canonical=rule.canonical,
            term_type=NormalizedTermType.TABLE_TERM,
            source="normalization_config",
            confidence=rule.confidence,
        )
        terms.append(term)
        candidates[rule.canonical] = rule.candidate_tables
        traces.append(
            NormalizationTrace(
                field_name="table",
                before=matched_alias,
                after=rule.canonical,
                rule="table_term_mapping",
                source="normalization_config",
            )
        )
    return terms, traces, candidates


def _extract_synonym_aliases_from_question(question: str) -> list[str]:
    """从原始问题中识别业务同义词。

    这里按长词优先扫描，避免“支付相关”先被“支付”截断。
    英文 alias 会走边界匹配，避免 orderInfo 中的 order 被误识别为订单。
    """
    aliases: list[str] = []
    for rule in load_normalization_config().synonyms:
        for alias in sorted([rule.canonical, *rule.aliases], key=len, reverse=True):
            if _alias_in_question(alias, question) and alias not in aliases:
                aliases.append(alias)
    return aliases


def _alias_in_question(alias: str, question: str) -> bool:
    """判断 alias 是否真实出现在用户问题中。

    中文 alias 使用包含判断；英文/数字/下划线 alias 使用词边界匹配，
    防止把 orderInfo 中的 order 误当成独立业务词。
    """
    if re.fullmatch(r"[A-Za-z0-9_]+", alias):
        return re.search(rf"(?<![A-Za-z0-9_.]){re.escape(alias)}(?![A-Za-z0-9_.])", question) is not None
    return alias in question


def _build_normalization_notes(
    before: ExtractedEntities,
    after: ExtractedEntities,
    terms: list[NormalizedTerm],
    traces: list[NormalizationTrace],
) -> list[str]:
    """生成给用户和调试者看的归一化备注。

    notes 是轻量可读解释；真正结构化审计信息在 normalization_traces 中。
    面试中可以讲：notes 面向人读，trace 面向系统审计和后续评估。
    """
    notes: list[str] = []
    if before.table != after.table and after.table:
        notes.append(f"实体标准化: 表名已归一化为 {after.table.raw}。")
    if before.topic_keywords != after.topic_keywords:
        notes.append(f"实体标准化: topic_keywords 已清洗为 {after.topic_keywords}。")
    if before.biz_line != after.biz_line and after.biz_line:
        notes.append(f"实体标准化: 业务线已归一化为 {after.biz_line}。")
    for term in terms:
        notes.append(f"实体标准化: {term.text} -> {term.canonical} ({term.term_type.value})。")
    if traces:
        notes.append(f"实体标准化: 已记录 {len(traces)} 条 normalization trace。")
    if not notes:
        notes.append("实体标准化: 实体已检查，无需额外归一化。")
    return notes


def _build_basic_normalization_traces(before: ExtractedEntities, after: ExtractedEntities) -> list[NormalizationTrace]:
    """生成基础字段变化 trace。

    记录 before/after/rule/source，解释每个实体为什么发生变化。
    生产中可把这些 trace 落日志或审计表，用于排查误归一化和优化词典。
    """
    traces: list[NormalizationTrace] = []
    if before.table != after.table:
        traces.append(
            NormalizationTrace(
                field_name="table",
                before=before.table.raw if before.table else None,
                after=after.table.raw if after.table else None,
                rule="table_identifier_format_normalization",
                source="planner_normalizer",
            )
        )
    if before.topic_keywords != after.topic_keywords:
        traces.append(
            NormalizationTrace(
                field_name="topic_keywords",
                before=before.topic_keywords,
                after=after.topic_keywords,
                rule="topic_keyword_cleanup",
                source="planner_normalizer",
            )
        )
    if before.biz_line != after.biz_line:
        traces.append(
            NormalizationTrace(
                field_name="biz_line",
                before=before.biz_line,
                after=after.biz_line,
                rule="basic_text_cleanup",
                source="planner_normalizer",
            )
        )
    return traces


def _merge_preserve_order(values: list[str]) -> list[str]:
    merged: list[str] = []
    for value in values:
        if value not in merged:
            merged.append(value)
    return merged
