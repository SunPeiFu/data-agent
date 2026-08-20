from data_agent.models import IntentType
from data_agent.planner import plan_question
from data_agent.slot_rules import load_slot_rule_config


def test_enterprise_planner_notes_include_metadata_auth_validation_and_trace() -> None:
    result = plan_question("dwd.orderInfo 表修改，对下游报表和任务有什么影响")
    notes = "\n".join(result.notes)

    assert "元数据候选解析" in notes
    assert "权限校验" in notes
    assert "澄清决策" in notes
    assert "计划校验: DAG 结构通过。" in notes
    assert "计划校验: 工具 action 注册关系通过。" in notes
    assert "计划校验: 参数 schema 通过。" in notes
    assert "计划校验: intent=impact_analysis 工具组合契约通过。" in notes
    assert "计划校验: 表元数据候选状态通过。" in notes
    assert "计划校验: 执行策略边界通过。" in notes
    assert "Trace: trace_id=trace-" in notes
    assert ", run_id=run-" in notes
    assert "run_status=completed" in notes
    assert "planner_version=v1-enterprise-planner" in notes


def test_enterprise_planner_one_part_table_uses_layer_to_narrow_candidates() -> None:
    result = plan_question("营销域 DWD 层 userInfo 表的下游血缘有哪些")
    notes = "\n".join(result.notes)

    assert "一段式表名 userInfo 已生成候选" in notes
    assert result.need_clarification is False
    assert result.entities.table is not None
    assert result.entities.table.raw == "dwd.userInfo"
    lineage_step = next(step for step in result.task_steps if step.action == "lineage_search")
    assert lineage_step.params["table"] == "dwd.userInfo"
    assert "槽位后校验: 通过。" in notes


def test_enterprise_planner_normalizes_table_schema_and_records_notes() -> None:
    result = plan_question("查询 DWD.orderInfo 表的下游血缘")
    notes = "\n".join(result.notes)

    assert result.entities.table is not None
    assert result.entities.table.raw == "dwd.orderInfo"
    assert result.entities.table.schema_name == "dwd"
    assert "实体标准化: 表名已归一化为 dwd.orderInfo。" in notes


def test_enterprise_planner_cleans_topic_keywords_before_search_plan() -> None:
    result = plan_question("营销域 DWD 层搜索支付相关表和表说明")
    notes = "\n".join(result.notes)

    assert result.entities.topic_keywords == ["支付"]
    assert "实体标准化: topic_keywords 已清洗为 ['支付']。" in notes


def test_enterprise_planner_maps_table_term_and_keeps_normalization_trace() -> None:
    result = plan_question("订单信息表修改，对下游报表和任务有什么影响")
    notes = "\n".join(result.notes)

    assert result.need_clarification is True
    assert result.task_steps == []
    assert any(term.text == "订单信息表" and term.canonical == "order_info" for term in result.normalized_terms)
    assert any(trace.rule == "table_term_mapping" for trace in result.normalization_traces)
    assert "实体标准化: 订单信息表 -> order_info (table_term)。" in notes
    assert "table_terms 命中候选表" in notes
    assert "多个候选表未自动选择" in notes
    assert "需要用户选择唯一表" in notes


def test_enterprise_planner_maps_topic_synonym_to_canonical_term() -> None:
    result = plan_question("营销域 DWD 层搜索支付相关表和表说明")

    assert result.entities.topic_keywords == ["支付"]
    assert any(term.text == "支付相关" and term.canonical == "支付" for term in result.normalized_terms)
    assert any(trace.rule == "term_synonym_mapping" for trace in result.normalization_traces)


def test_enterprise_planner_uses_configured_slot_rules() -> None:
    config = load_slot_rule_config()

    assert config.rule_for(IntentType.IMPACT_ANALYSIS).pre_required_any == [
        "table",
        "table_term",
        "semantic_table_query",
    ]
    assert config.rule_for(IntentType.IMPACT_ANALYSIS).post_required_any == ["table"]


def test_enterprise_planner_missing_table_routes_to_clarification_without_plan() -> None:
    result = plan_question("查询下游依赖关系")
    notes = "\n".join(result.notes)

    assert result.need_clarification is True
    assert result.task_steps == []
    assert "槽位预校验" in notes
    assert "澄清决策" in notes


def test_enterprise_planner_ambiguous_table_term_routes_to_clarification_without_plan() -> None:
    result = plan_question("订单信息表修改，对下游报表和任务有什么影响")
    notes = "\n".join(result.notes)

    assert result.need_clarification is True
    assert result.task_steps == []
    assert "槽位后校验: ambiguous slot=table" in notes
    assert result.clarification_question is not None


def test_enterprise_planner_domain_conflict_routes_to_clarification_without_plan() -> None:
    result = plan_question("营销域 DWD 层 dwd.orderInfo 表的下游血缘有哪些")
    notes = "\n".join(result.notes)

    assert result.need_clarification is True
    assert result.task_steps == []
    assert "槽位后校验: conflict slot=domain" in notes
    assert "营销域 与候选表 dwd.orderInfo 的主题域 交易域 不一致" in notes


def test_enterprise_planner_layer_conflict_routes_to_clarification_without_plan() -> None:
    result = plan_question("DIM 层 dwd.payment_detail 表的下游血缘有哪些")
    notes = "\n".join(result.notes)

    assert result.need_clarification is True
    assert result.task_steps == []
    assert "槽位后校验: conflict slot=data_layer" in notes
    assert "DIM 与候选表 dwd.payment_detail 的分层 DWD 不一致" in notes
