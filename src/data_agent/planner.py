from __future__ import annotations

import os
import re
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph

from data_agent.hybrid_router import HybridRouteResult, HybridQuestionRouter
from data_agent.metadata_repository import MetadataCandidate, MetadataRepositoryError, MySQLMetadataRepository
from data_agent.milvus_repository import MilvusMetadataRepository, MilvusRepositoryError
from data_agent.models import (
    DataLayer,
    DomainType,
    ExtractedEntities,
    IntentType,
    MetadataCandidateEvidence,
    MetadataCandidateSource,
    MetadataValidationStatus,
    NormalizationTrace,
    NormalizedTerm,
    NormalizedTermType,
    PlanningResult,
    SlotIssue,
    SlotIssueType,
    SlotValidationResult,
    SlotValidationStage,
    TableIdentifier,
)
from data_agent.normalization import load_normalization_config
from data_agent.slot_rules import load_slot_rule_config
from data_agent.task_builder import build_task_plan


class PlannerState(TypedDict, total=False):
    question: str
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
    slot_errors: list[str]
    pre_slot_validation: SlotValidationResult
    post_slot_validation: SlotValidationResult
    planner_decision: str
    metadata_candidates: dict[str, list[str]]
    metadata_candidate_profiles: dict[str, dict[str, str | None]]
    metadata_candidate_evidence: dict[str, MetadataCandidateEvidence]
    table_term_candidates: dict[str, list[str]]
    semantic_table_query: str | None
    requested_table: TableIdentifier | None
    authorized: bool
    trace_id: str
    intent: IntentType
    confidence: float
    entities: ExtractedEntities
    result: PlanningResult


def create_planning_graph() -> Any:

    # 创建一个状态图 
    # add_node节点即不同的python函数 即声明功能
    # add_edge即声明节点之间的流程编排顺序
    graph = StateGraph(PlannerState)

    # step1 意图识别 state返回完整的意图识别结果和相关信息 都是从route_result中获
    graph.add_node("classify_intent", _classify_intent)

    # step2 实体抽取 意图识别的result中直接提取entities
    graph.add_node("extract_entities", _extract_entities_node)

    # step3 归一化实体 (让进入工具之前的实体完全符合工具要求)
    graph.add_node("normalize_entities", _normalize_entities)

    # step4 元数据解析前槽位校验: 只基于用户输入和实体抽取结果，判断是否具备最小可执行线索
    # 典型阻断: 用户只说“查下游”，但没有给表名或表级业务术语
    graph.add_node("validate_slots", _validate_slots)

    # step5 元数据候选解析: 优先查询 MySQL meta_table/meta_table_ext，失败时使用 mock fallback
    # 典型转换: userInfo / 订单信息表 -> dwd.userInfo / dim.userInfo / dwd.orderInfo
    graph.add_node("resolve_metadata_candidates", _resolve_metadata_candidates)

    # step6 📌 mock 权限和治理校验: 后续接权限系统 / 业务域隔离策略
    graph.add_node("authorize_context", _authorize_context)

    # step7 元数据解析后槽位校验: 判断候选表是否唯一、是否还缺关键槽位
    graph.add_node("post_validate_slots", _post_validate_slots)

    # step8 📌 澄清决策: 判断缺槽位、多候选、无权限等是否需要先问用户
    graph.add_node("decide_clarification_or_continue", _decide_clarification_or_continue)

    # step9 构建计划
    graph.add_node("build_task_plan", _build_task_plan)

    # step10 生成澄清结果: conditional edge 命中后不继续生成工具计划
    graph.add_node("return_clarification_result", _return_clarification_result)

    # step11 生成拒绝结果: conditional edge 命中后不继续生成工具计划
    graph.add_node("return_forbidden_result", _return_forbidden_result)

    # step12 📌校验任务计划: 工具名、参数、依赖关系等
    graph.add_node("validate_task_plan", _validate_task_plan)

    # step13 附加 trace: 记录路由、候选解析、权限、计划校验等备注
    graph.add_node("attach_trace", _attach_trace)

    # step14 返回计划结果
    graph.add_node("return_planning_result", _return_planning_result)

    # 设置整个图的first节点是什么 开始节点
    graph.set_entry_point("classify_intent")

    # 设置节点之间的编排流程
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
        },
    )
    graph.add_edge("build_task_plan", "validate_task_plan")
    graph.add_edge("validate_task_plan", "attach_trace")
    graph.add_edge("return_clarification_result", "attach_trace")
    graph.add_edge("return_forbidden_result", "attach_trace")
    graph.add_edge("attach_trace", "return_planning_result")
    graph.add_edge("return_planning_result", END)
    return graph.compile()


def plan_question(question: str) -> PlanningResult:
    app = create_planning_graph()
    final_state = app.invoke({"question": question})
    return final_state["result"]


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
    """Mock authorization and governance check.

    生产中这里应接权限系统、业务域隔离和审计策略。
    """
    notes = ["权限校验: mock 通过，后续接入真实权限、业务域隔离和审计策略。"]
    return {"authorized": True, "authorization_notes": notes}


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
    elif blocking_issues:
        decision = "clarify"
        notes.extend(f"澄清决策: {issue.message}" for issue in blocking_issues)
    else:
        decision = "continue"
        notes.append("澄清决策: 关键信息充分，继续生成任务计划。")
    return {"planner_decision": decision, "clarification_notes": notes}


def _route_after_clarification_decision(state: PlannerState) -> str:
    return state.get("planner_decision", "continue")


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
    return {"result": result}


def _return_clarification_result(state: PlannerState) -> PlannerState:
    """Build a PlanningResult for clarification branches.

    生产中这里通常会返回可交互的澄清卡片，例如候选表列表；当前先输出文本问题和 notes。
    """
    issues = _all_slot_issues(state)
    clarification_question = _build_clarification_question(issues)
    result = PlanningResult(
        question=state["question"],
        intent=state.get("intent", IntentType.UNKNOWN),
        confidence=min(state.get("confidence", 0.0), 0.62),
        entities=state["entities"],
        task_steps=[],
        need_clarification=True,
        clarification_question=clarification_question,
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
    return {"result": result}


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
    """Attach trace notes for auditability."""
    trace_id = f"plan-{uuid4().hex[:12]}"
    result = state["result"]
    trace_notes = [
        f"Trace: trace_id={trace_id}",
        f"Trace: intent={state.get('intent', IntentType.UNKNOWN).value}, confidence={state.get('confidence', 0.0):.2f}",
        "Trace: planner_version=v1-enterprise-mock",
    ]
    result.notes = [*result.notes, *state.get("plan_validation_notes", []), *trace_notes]
    return {"trace_id": trace_id, "trace_notes": trace_notes, "result": result}


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
    """Collect blocking slot issues from pre/post validation.

    原子职责：
    澄清节点只关心阻断型问题。非阻断问题可以留在 notes 中提示，但不影响继续规划。
    """
    return [
        issue
        for key in ["pre_slot_validation", "post_slot_validation"]
        if (validation := state.get(key))
        for issue in validation.blocking_issues
    ]


def _build_clarification_question(issues: list[SlotIssue]) -> str:
    """Build the clarification question shown to the user.

    原子职责：
    把系统内部的 SlotIssue 转成用户能直接回复的问题。ambiguous 要求用户选唯一表，
    missing 要求用户补表名或业务术语，conflict 直接说明冲突点。
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
