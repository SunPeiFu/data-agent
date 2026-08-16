# 智能数据探查 v1 设计说明

## 目标

v1 目标是实现一个可面试、可学习、可扩展的 Agent Planner。它不直接回答业务问题，而是把问题转成结构化执行计划。

## 数据流

```text
用户问题
  -> classify_intent
  -> extract_entities
  -> normalize_entities
  -> validate_slots
  -> resolve_metadata_candidates
  -> authorize_context
  -> decide_clarification_or_continue
  -> build_task_plan
  -> validate_task_plan
  -> attach_trace
  -> PlanningResult JSON
```

## LangGraph 节点

- `classify_intent`：通过 Hybrid Router 先做规则预分析，再调用 OpenAI-compatible LLM 做结构化解析，最后用策略合并结果。
- `extract_entities`：直接消费 Hybrid Router 合并后的实体结果；无模型或模型输出不合法时使用规则预分析兜底。
- `normalize_entities`：标准化 LLM/规则抽取出的实体，包括表名 schema/catalog 大小写、字段名展示字符、topic_keywords 去重去噪、主题域/数仓分层词剔除，并记录归一化 notes。
- `validate_slots`：检查关键槽位是否缺失。
- `resolve_metadata_candidates`：mock 解析表名/字段名候选；后续接 TiDB / DataHub / OpenMetadata 等元数据服务。
- `authorize_context`：mock 权限和治理校验；后续接权限系统、敏感字段策略、业务域隔离。
- `decide_clarification_or_continue`：根据缺槽位、多候选、权限状态决定是否需要澄清；v1 先写入 notes 并继续生成计划。
- `build_task_plan`：生成 TiDB、Milvus、Neo4j、impact analyzer 的工具调用计划。
- `validate_task_plan`：mock 校验工具名、步骤依赖和计划结构。
- `attach_trace`：附加 trace_id、planner_version、intent/confidence 等审计信息。
- `return_planning_result`：返回结构化结果。

## 工具规划

- `tidb_metadata.filter_tables`：按业务线、主题域、分层、关键词过滤表。
- `tidb_metadata.resolve_table`：解析一段式、两段式、三段式表名，定位候选表。
- `milvus_rag.semantic_search`：召回表说明、字段含义、业务语义。
- `neo4j_lineage.lineage_search`：通过 `direction` 参数控制上游、下游、双向血缘。
- `impact_analyzer.classify_impact`：对单向影响范围分级。
- `impact_analyzer.merge_lineage_and_metadata`：融合血缘和元数据语义。

## 当前实现边界

当前所有工具都是“计划中的工具”，不会真正访问 TiDB、Milvus、Neo4j。这个边界是刻意设计的，方便先把 Agent 的规划层讲清楚，再逐步补真实执行层。

## LLM 接入

生产路径使用 `HybridQuestionRouter`：

```text
RulePreAnalyzer
  -> LLMQuestionAnalyzer
  -> Pydantic 校验
  -> PolicyResolver
  -> HybridRouteResult
```

规则预分析负责识别高确定性约束，例如字段变更、上下游方向、枚举值和明显表名。LLM 负责理解自然语言和补充弱语义实体。`PolicyResolver` 不再使用简单的固定优先级，而是把规则和 LLM 都转成字段候选：

```text
RuleEvidence
  -> FieldCandidate
  -> ResolvedField
  -> EntityResolution
```

每个候选都包含 source、confidence、strength、reason、是否需要元数据校验。`domain`、`data_layer`、`operation`、`lineage_direction` 这类高确定性字段优先采用 hard rule；`table` 和 `field_name` 会被标记为候选值，后续必须通过 TiDB 元数据服务做 `resolve_table` / `check_field` 确认；冲突接近时会标记为 `needs_clarification`。

环境变量：

- `DATA_AGENT_USE_LLM`：是否启用 LLM，默认 true。
- `DATA_AGENT_LLM_BASE_URL`：OpenAI-compatible base URL，默认 `http://localhost:1234/v1`。
- `DATA_AGENT_LLM_API_KEY`：API key，LM Studio 可使用任意非空字符串。
- `DATA_AGENT_LLM_MODEL`：模型名；未配置时自动走规则兜底。
- `DATA_AGENT_LLM_TIMEOUT_SECONDS`：请求超时，默认 30。
- `DATA_AGENT_LLM_MAX_RETRIES`：重试次数，默认 2。
