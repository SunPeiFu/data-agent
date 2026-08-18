from __future__ import annotations

import re
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph

from data_agent.hybrid_router import HybridRouteResult, HybridQuestionRouter
from data_agent.models import (
    DataLayer,
    DomainType,
    ExtractedEntities,
    IntentType,
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
    table_term_candidates: dict[str, list[str]]
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

    # step4 元数据解析前槽位校验: 判断用户是否给了足够的表级定位线索
    graph.add_node("validate_slots", _validate_slots)

    # step5 📌mock 元数据候选解析: 后续接 TiDB / 数据目录服务
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
    }


def _validate_slots(state: PlannerState) -> PlannerState:
    """Pre-metadata slot validation.

    核心职责：
    - 根据配置化 intent slot rule 判断用户是否提供最低可执行线索。
    - 这里只校验“有没有线索”，不判断候选是否唯一，因为真实候选要等元数据解析后才知道。
    - 输出结构化 SlotValidationResult，后续节点可以按 issue_type 做澄清、拒绝或继续执行。
    """
    intent = state.get("intent", IntentType.UNKNOWN)
    entities = state["entities"]
    config = load_slot_rule_config()
    rule = config.rule_for(intent)
    issues: list[SlotIssue] = []
    notes = [f"槽位预校验: intent={intent.value} 使用配置化 required_any={rule.pre_required_any}。"]

    if intent == IntentType.UNKNOWN:
        issues.append(
            _slot_issue(
                slot_name="intent",
                issue_type=SlotIssueType.MISSING,
                message="无法识别用户想查询元数据、血缘关系还是表变更影响。",
            )
        )
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
    - 当前 mock 会用数仓分层先做一次候选过滤，仍无法唯一时才进入澄清分支。
    """
    intent = state.get("intent", IntentType.UNKNOWN)
    entities = state["entities"]
    config = load_slot_rule_config()
    rule = config.rule_for(intent)
    candidates = state.get("metadata_candidates", {})
    table_candidates = candidates.get("table", [])
    issues: list[SlotIssue] = []
    notes = [f"槽位后校验: intent={intent.value} 使用配置化 post_required_any={rule.post_required_any}。"]

    if rule.post_required_any and not _has_any_slot(rule.post_required_any, entities, state):
        issues.append(
            _slot_issue(
                slot_name=",".join(rule.post_required_any),
                issue_type=SlotIssueType.MISSING,
                message="元数据解析后仍缺少可执行的表名。",
            )
        )

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

    issues.extend(_validate_cross_slot_consistency(entities, table_candidates))

    result = SlotValidationResult(
        stage=SlotValidationStage.POST_METADATA,
        passed=not any(issue.blocking for issue in issues),
        issues=issues,
        notes=[*notes, *_slot_issue_notes("槽位后校验", issues)],
    )
    return {"post_slot_validation": result}


def _resolve_metadata_candidates(state: PlannerState) -> PlannerState:
    """Mock metadata resolution.

    生产中这里应接 TiDB / DataHub / OpenMetadata：
    - 一段式表名解析成候选 fully qualified name。
    - 两段式/三段式表名校验存在性。
    - 表级业务术语映射到候选物理表。
    """
    entities = state["entities"]
    candidates: dict[str, list[str]] = {}
    notes: list[str] = []

    if entities.table:
        table_candidates = _mock_table_candidates(entities.table)
        table_candidates = _filter_table_candidates_by_context(table_candidates, entities)
        candidates["table"] = table_candidates
        if entities.table.parts_count == 1:
            notes.append(f"元数据候选解析: 一段式表名 {entities.table.raw} 已生成候选 {table_candidates}。")
        else:
            notes.append(f"元数据候选解析: 表名 {entities.table.raw} 已作为待校验候选。")

    table_term_candidates = state.get("table_term_candidates", {})
    if table_term_candidates:
        term_candidates = [table for tables in table_term_candidates.values() for table in tables]
        existing = candidates.get("table", [])
        merged_candidates = _filter_table_candidates_by_context(
            _merge_preserve_order([*existing, *term_candidates]),
            entities,
        )
        candidates["table"] = merged_candidates
        notes.append(f"元数据候选解析: table_terms 命中候选表 {table_term_candidates}。")
        if entities.table is None and len(merged_candidates) == 1:
            entities = entities.model_copy(update={"table": TableIdentifier.parse(merged_candidates[0])})
            notes.append(f"元数据候选解析: 唯一候选表 {merged_candidates[0]} 已回填到实体。")
        elif entities.table is None and len(merged_candidates) > 1:
            notes.append("元数据候选解析: 多个候选表未自动选择，等待澄清或真实元数据排序。")

    if not notes:
        notes.append("元数据候选解析: 当前问题无需表候选解析。")

    return {"entities": entities, "metadata_candidates": candidates, "metadata_notes": notes}


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
    if table.parts_count == 1:
        return [f"dwd.{table.table_name}", f"dim.{table.table_name}"]
    return [table.raw]


def _filter_table_candidates_by_context(candidates: list[str], entities: ExtractedEntities) -> list[str]:
    """Use available table-level context to narrow mock metadata candidates.

    当前先用 data_layer 做最小可解释消歧：`userInfo + DWD` 会优先收敛到
    `dwd.userInfo`。真实生产里这里会改成 TiDB/DataCatalog 的评分和排序结果。
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


def _validate_cross_slot_consistency(entities: ExtractedEntities, table_candidates: list[str]) -> list[SlotIssue]:
    """Validate consistency between user slots and resolved table profile.

    真实企业里，这一步会读取元数据服务返回的表画像：
    - 表所属主题域
    - 表所在数仓分层
    - 表所属业务线或权限域

    当前用 mock profile 表达同样的生产逻辑：如果用户说“营销域”，但唯一候选表画像是
    “交易域”，就不能继续生成计划，必须先澄清或修正。
    """
    if len(table_candidates) != 1:
        return []

    table_name = table_candidates[0]
    profile = MOCK_TABLE_PROFILES.get(table_name)
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
    """Create a slot issue using configured blocking policy."""
    return SlotIssue(
        slot_name=slot_name,
        issue_type=issue_type,
        message=message,
        blocking=load_slot_rule_config().is_blocking(issue_type),
    )


def _has_any_slot(slot_names: list[str], entities: ExtractedEntities, state: PlannerState) -> bool:
    """Check whether at least one configured slot has a usable value."""
    return any(_slot_has_value(slot_name, entities, state) for slot_name in slot_names)


def _slot_has_value(slot_name: str, entities: ExtractedEntities, state: PlannerState) -> bool:
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
    messages = {
        IntentType.METADATA_SEARCH: "缺少主题域、数仓分层、表名、表级业务术语或业务关键词，无法执行元数据搜索。",
        IntentType.LINEAGE_SEARCH: "缺少表名或表级业务术语，血缘查询无法定位数据资产。",
        IntentType.IMPACT_ANALYSIS: "缺少表名或表级业务术语，影响分析无法定位数据资产。",
        IntentType.UNKNOWN: "请补充你想查询元数据、血缘关系，还是表变更影响。",
    }
    return messages.get(intent, "缺少关键槽位，无法继续规划。")


def _slot_issue_notes(prefix: str, issues: list[SlotIssue]) -> list[str]:
    if not issues:
        return [f"{prefix}: 通过。"]
    return [
        f"{prefix}: {issue.issue_type.value} slot={issue.slot_name}, blocking={issue.blocking}, message={issue.message}"
        for issue in issues
    ]


def _slot_validation_notes(state: PlannerState) -> list[str]:
    notes: list[str] = []
    for key in ["pre_slot_validation", "post_slot_validation"]:
        validation = state.get(key)
        if validation:
            notes.extend(validation.notes)
    return notes


def _all_slot_issues(state: PlannerState) -> list[SlotIssue]:
    return [
        issue
        for key in ["pre_slot_validation", "post_slot_validation"]
        if (validation := state.get(key))
        for issue in validation.blocking_issues
    ]


def _build_clarification_question(issues: list[SlotIssue]) -> str:
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
