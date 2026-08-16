from __future__ import annotations

from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph

from data_agent.hybrid_router import HybridRouteResult, HybridQuestionRouter
from data_agent.models import ExtractedEntities, IntentType, PlanningResult, TableIdentifier
from data_agent.task_builder import build_task_plan


class PlannerState(TypedDict, total=False):
    question: str
    route_result: HybridRouteResult
    routing_notes: list[str]
    normalization_notes: list[str]
    metadata_notes: list[str]
    authorization_notes: list[str]
    clarification_notes: list[str] # 澄清
    plan_validation_notes: list[str]
    trace_notes: list[str]
    slot_errors: list[str]
    metadata_candidates: dict[str, list[str]]
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

    # step3 标准化实体 (让进入工具之前的实体完全符合工具要求)
    graph.add_node("normalize_entities", _normalize_entities)

    # step4 校验槽位 某系必输信息 关键信息是否补全(即数据的合法性与准确性 比如工具调用的必要信息)
    graph.add_node("validate_slots", _validate_slots)

    # step5 📌mock 元数据候选解析: 后续接 TiDB / 数据目录服务
    graph.add_node("resolve_metadata_candidates", _resolve_metadata_candidates)

    # step6 📌 mock 权限和治理校验: 后续接权限系统 / 敏感字段策略
    graph.add_node("authorize_context", _authorize_context)

    # step7 📌 澄清决策: 判断缺槽位、多候选、无权限等是否需要先问用户
    graph.add_node("decide_clarification_or_continue", _decide_clarification_or_continue)

    # step8  构建计划 
    graph.add_node("build_task_plan", _build_task_plan)

    # step9 📌校验任务计划: 工具名、参数、依赖关系等
    graph.add_node("validate_task_plan", _validate_task_plan)

    # step10 附加 trace: 记录路由、候选解析、权限、计划校验等备注
    graph.add_node("attach_trace", _attach_trace)

    # step11 返回计划结果
    graph.add_node("return_planning_result", _return_planning_result)

    # 设置整个图的first节点是什么 开始节点
    graph.set_entry_point("classify_intent")

    # 设置节点之间的编排流程
    graph.add_edge("classify_intent", "extract_entities")
    graph.add_edge("extract_entities", "normalize_entities")
    graph.add_edge("normalize_entities", "validate_slots")
    graph.add_edge("validate_slots", "resolve_metadata_candidates")
    graph.add_edge("resolve_metadata_candidates", "authorize_context")
    graph.add_edge("authorize_context", "decide_clarification_or_continue")
    graph.add_edge("decide_clarification_or_continue", "build_task_plan")
    graph.add_edge("build_task_plan", "validate_task_plan")
    graph.add_edge("validate_task_plan", "attach_trace")
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
    - 字段名要去掉反引号/引号等展示字符。
    - topic_keywords 要去重、去噪，避免把主题域/数仓分层当成检索词。
    - 归一化动作要写入 notes，方便排查“为什么工具参数变成这样”。
    """
    entities = state["entities"]
    normalized = ExtractedEntities(
        biz_line=_normalize_text(entities.biz_line),
        domain=entities.domain,
        data_layer=entities.data_layer,
        table=_normalize_table_identifier(entities.table),
        field_name=_normalize_identifier_text(entities.field_name),
        operation=entities.operation,
        topic_keywords=_normalize_topic_keywords(entities),
        lineage_direction=entities.lineage_direction,
    )

    notes = _build_normalization_notes(before=entities, after=normalized)
    return {"entities": normalized, "normalization_notes": notes}


def _validate_slots(state: PlannerState) -> PlannerState:
    errors: list[str] = []
    intent = state.get("intent", IntentType.UNKNOWN)
    entities = state["entities"]
    if intent in {IntentType.LINEAGE_SEARCH, IntentType.IMPACT_ANALYSIS} and entities.table is None:
        errors.append("缺少表名，血缘查询或影响分析无法定位数据资产。")
    if intent == IntentType.METADATA_SEARCH and not any(
        [entities.domain, entities.data_layer, entities.topic_keywords, entities.table]
    ):
        errors.append("缺少主题域、数仓分层、表名或业务关键词，无法执行元数据搜索。")
    return {"slot_errors": errors}


def _resolve_metadata_candidates(state: PlannerState) -> PlannerState:
    """Mock metadata resolution.

    生产中这里应接 TiDB / DataHub / OpenMetadata：
    - 一段式表名解析成候选 fully qualified name。
    - 两段式/三段式表名校验存在性。
    - 字段名校验是否属于表。
    """
    entities = state["entities"]
    candidates: dict[str, list[str]] = {}
    notes: list[str] = []

    if entities.table:
        table_candidates = _mock_table_candidates(entities.table)
        candidates["table"] = table_candidates
        if entities.table.parts_count == 1:
            notes.append(f"元数据候选解析: 一段式表名 {entities.table.raw} 已生成候选 {table_candidates}。")
        else:
            notes.append(f"元数据候选解析: 表名 {entities.table.raw} 已作为待校验候选。")

    if entities.field_name:
        candidates["field_name"] = [entities.field_name]
        notes.append(f"元数据候选解析: 字段 {entities.field_name} 已作为待校验候选。")

    if not notes:
        notes.append("元数据候选解析: 当前问题无需表/字段候选解析。")

    return {"metadata_candidates": candidates, "metadata_notes": notes}


def _authorize_context(state: PlannerState) -> PlannerState:
    """Mock authorization and governance check.

    生产中这里应接权限系统、敏感字段标签、业务域隔离和审计策略。
    """
    entities = state["entities"]
    notes = ["权限校验: mock 通过，后续接入真实权限和敏感字段治理策略。"]
    if entities.field_name:
        notes.append(f"权限校验: 字段级访问 {entities.field_name} 当前按 mock 策略允许。")
    return {"authorized": True, "authorization_notes": notes}


def _decide_clarification_or_continue(state: PlannerState) -> PlannerState:
    """Decide whether the planner should ask the user for more information.

    v1 仍继续生成计划，但会把澄清原因写入 notes；生产中可在这里条件分支到澄清节点。
    """
    notes: list[str] = []
    if state.get("slot_errors"):
        notes.extend(f"澄清决策: {error}" for error in state["slot_errors"])
    if not state.get("authorized", True):
        notes.append("澄清决策: 当前用户无权限，需要拒绝或发起权限申请。")

    table_candidates = state.get("metadata_candidates", {}).get("table", [])
    if len(table_candidates) > 1:
        notes.append("澄清决策: 表名存在多个元数据候选，生产环境应让用户选择唯一表。")
    elif table_candidates:
        notes.append("澄清决策: 元数据候选唯一或可继续进入计划生成。")
    elif not notes:
        notes.append("澄清决策: 关键信息充分，继续生成任务计划。")
    return {"clarification_notes": notes}


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
        *state.get("metadata_notes", []),
        *state.get("authorization_notes", []),
        *state.get("clarification_notes", []),
        *result.notes,
    ]
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
        "allowed": {"table", "field_name", "direction", "depth", "lineage_granularity"},
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

    lineage_steps = [
        step for step in result.task_steps if step.tool_name == "neo4j_lineage" and step.action == "lineage_search"
    ]
    for step in lineage_steps:
        if step.params.get("lineage_granularity") == "field" and not candidates.get("field_name"):
            notes.append("计划校验: 字段级血缘缺少字段元数据候选。")
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


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip(" \t\r\n，,。；;：:")
    return cleaned or None


def _normalize_identifier_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip(" \t\r\n`'\"，,。；;：:")
    return cleaned or None


def _normalize_table_identifier(table: TableIdentifier | None) -> TableIdentifier | None:
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


def _normalize_topic_keywords(entities: ExtractedEntities) -> list[str]:
    stopwords = {
        "表",
        "字段",
        "字段说明",
        "表说明",
        "相关表",
        "血缘",
        "上游",
        "下游",
        "影响",
    }
    if entities.domain:
        stopwords.add(entities.domain.value)
        stopwords.add(entities.domain.value.removesuffix("域"))
    if entities.data_layer:
        stopwords.add(entities.data_layer.value)
        stopwords.add(entities.data_layer.value.lower())

    normalized: list[str] = []
    for keyword in entities.topic_keywords:
        cleaned = _normalize_text(keyword)
        if not cleaned or cleaned in stopwords:
            continue
        if cleaned not in normalized:
            normalized.append(cleaned)
    return normalized


def _build_normalization_notes(before: ExtractedEntities, after: ExtractedEntities) -> list[str]:
    notes: list[str] = []
    if before.table != after.table and after.table:
        notes.append(f"实体标准化: 表名已归一化为 {after.table.raw}。")
    if before.field_name != after.field_name and after.field_name:
        notes.append(f"实体标准化: 字段名已归一化为 {after.field_name}。")
    if before.topic_keywords != after.topic_keywords:
        notes.append(f"实体标准化: topic_keywords 已清洗为 {after.topic_keywords}。")
    if before.biz_line != after.biz_line and after.biz_line:
        notes.append(f"实体标准化: 业务线已归一化为 {after.biz_line}。")
    if not notes:
        notes.append("实体标准化: 实体已检查，无需额外归一化。")
    return notes
