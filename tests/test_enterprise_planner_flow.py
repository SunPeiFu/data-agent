from data_agent.planner import plan_question


def test_enterprise_planner_notes_include_metadata_auth_validation_and_trace() -> None:
    result = plan_question("dwd.orderInfo 表字段 pay_amount 修改，对下游报表和任务有什么影响")
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
    assert "Trace: trace_id=plan-" in notes
    assert "planner_version=v1-enterprise-mock" in notes


def test_enterprise_planner_one_part_table_notes_multiple_candidates() -> None:
    result = plan_question("营销域 DWD 层 userInfo 表的下游血缘有哪些")
    notes = "\n".join(result.notes)

    assert "一段式表名 userInfo 已生成候选" in notes
    assert "表名存在多个元数据候选" in notes
    assert "表元数据候选不唯一" in notes


def test_enterprise_planner_normalizes_table_schema_and_records_notes() -> None:
    result = plan_question("查询 DWD.orderInfo 表的下游血缘")
    notes = "\n".join(result.notes)

    assert result.entities.table is not None
    assert result.entities.table.raw == "dwd.orderInfo"
    assert result.entities.table.schema_name == "dwd"
    assert "实体标准化: 表名已归一化为 dwd.orderInfo。" in notes


def test_enterprise_planner_cleans_topic_keywords_before_search_plan() -> None:
    result = plan_question("营销域 DWD 层搜索支付相关表和字段说明")
    notes = "\n".join(result.notes)

    assert result.entities.topic_keywords == ["支付"]
    assert "实体标准化: topic_keywords 已清洗为 ['支付']。" in notes
