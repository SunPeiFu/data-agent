# 智能数据探查 v1 设计说明

## 目标

v1 目标是实现一个可面试、可学习、可扩展的 Agent Planner。它不直接回答业务问题，而是把问题转成结构化执行计划。

## 数据流

```text
用户问题
  -> init_trace_context
  -> classify_intent
  -> extract_entities
  -> normalize_entities
  -> validate_slots
  -> resolve_metadata_candidates
  -> authorize_context
  -> post_validate_slots
  -> decide_clarification_or_continue
     | continue -> build_task_plan -> validate_task_plan
     | clarify -> return_clarification_result -> attach_trace -> await_clarification_response
     |             ^                                      |
     |             +-- normalize/metadata/auth/validate <-+
     | return_forbidden_result
     | handoff -> return_handoff_result
  -> attach_trace
  -> PlanningResult JSON
```

## LangGraph 节点

- `classify_intent`：通过 Hybrid Router 先做规则预分析，再调用 OpenAI-compatible LLM 做结构化解析，最后用策略合并结果。
- `init_trace_context`：在业务节点执行前创建 thread_id/trace_id/run_id，并通过 TraceRecorder 开启本次 Run。
- `extract_entities`：直接消费 Hybrid Router 合并后的实体结果；无模型或模型输出不合法时使用规则预分析兜底。
- `normalize_entities`：标准化 LLM/规则抽取出的实体，包括表名 schema/catalog 大小写、表级业务术语、topic_keywords 去重去噪、主题域/数仓分层词剔除，并记录归一化 notes。
- `validate_slots`：元数据解析前槽位校验，根据 `config/slot_rules.yml` 判断用户是否提供最低可执行线索。
- `resolve_metadata_candidates`：按实体确定性动态路由。两/三段式表名直接查 MySQL；一段式先查 MySQL、未命中再补 Milvus；业务描述走 Milvus Dense + BM25 + 标量过滤，召回结果必须回 MySQL 校验。
- `authorize_context`：基于 subject-action-resource 做表级 RBAC 和业务域隔离；当前使用 YAML Provider，生产可替换为 IAM/Ranger HTTP Provider。
- `post_validate_slots`：元数据解析后槽位校验，判断候选表是否存在、是否唯一、是否可继续规划。
- `decide_clarification_or_continue`：根据结构化 `SlotValidationResult` 和权限状态输出 `continue`、`clarify` 或 `forbidden`，由 LangGraph conditional edge 控制后续分支。
- `return_clarification_result`：对全部阻断问题跨阶段去重并按槽位依赖排序，每轮返回一个结构化 `ClarificationRequest`；其他问题保存在 `pending_clarification_issues`，不生成工具计划。
- `await_clarification_response`：使用 LangGraph `interrupt()` 和 SQLite checkpointer 暂停会话；通过 `Command(resume=...)` 恢复后，将用户确认值以 `source=user_confirmed, confidence=1.0` 写回实体，再重新执行标准化、元数据、权限和槽位校验。
- `return_handoff_result`：达到最大澄清轮数后停止自动追问，返回人工数据服务台转交结果。
- `return_forbidden_result`：无权限时返回拒绝结果，不生成工具计划。
- `build_task_plan`：生成 TiDB、Milvus、Neo4j、impact analyzer 的工具调用计划。
- `validate_task_plan`：mock 校验工具名、步骤依赖和计划结构。
- `attach_trace`：不再临时创建 Trace ID；根据 completed/interrupted/forbidden/handoff 分支关闭当前 Run，并将 thread_id/trace_id/run_id/parent_run_id 附加到 PlanningResult。
- `return_planning_result`：返回结构化结果。

## 澄清链路学习总结

```text
SlotValidationResult
  -> _all_slot_issues                  收集阻断问题
  -> _prioritize_clarification_issues  跨阶段去重并按信息增益排序
  -> _build_clarification_question     生成用户问题
  -> _build_clarification_options      生成授权后的候选项
  -> _build_clarification_request      组装版本化交互协议
  -> _return_clarification_result      生成无工具步骤的暂停态结果
  -> _await_clarification_response     interrupt 持久化暂停
  -> resume_clarification              幂等、安全地 Command(resume)
  -> _validate_clarification_response  校验会话、版本和候选绑定
  -> _confirmed_response_value         获取服务端标准值
  -> _merge_clarification_answer       写回强类型实体
  -> normalize/metadata/auth/validate  重新通过全部治理门禁
```

核心原则：每轮只询问信息增益最高的问题，但不丢弃其他问题；前端提交内容始终是不可信输入；
用户确认值具有高置信度，但不能绕过资产存在性、权限和一致性校验；达到轮数上限后必须转人工。

## 工具规划

- `tidb_metadata.filter_tables`：按业务线、主题域、分层、关键词过滤表。
- `tidb_metadata.resolve_table`：解析一段式、两段式、三段式表名，定位候选表。
- `milvus_rag.semantic_search`：召回表说明和业务语义。
- `neo4j_lineage.lineage_search`：通过 `direction` 参数控制上游、下游、双向血缘。
- `impact_analyzer.classify_impact`：对单向影响范围分级。
- `impact_analyzer.merge_lineage_and_metadata`：融合血缘和元数据语义。

## 当前实现边界

当前 `resolve_metadata_candidates` 已接入 MySQL 元数据候选解析和 Milvus 查询 Repository。Milvus Collection 数据同步、TiDB、Neo4j 和最终工具执行仍待后续补齐。详细路由与 Schema 见 `docs/metadata_candidate_resolution_design.md`。

## LLM 接入

生产路径使用 `HybridQuestionRouter`：

```text
RulePreAnalyzer
  -> LLMQuestionAnalyzer
  -> Pydantic 校验
  -> PolicyResolver
  -> HybridRouteResult
```

规则预分析负责识别高确定性约束，例如表变更、上下游方向、枚举值和明显表名。LLM 负责理解自然语言和补充弱语义实体。`PolicyResolver` 不再使用简单的固定优先级，而是把规则和 LLM 都转成字段候选：

```text
RuleEvidence
  -> FieldCandidate
  -> ResolvedField
  -> EntityResolution
```

每个候选都包含 source、confidence、strength、reason、是否需要元数据校验。`domain`、`data_layer`、`operation`、`lineage_direction` 这类高确定性字段优先采用 hard rule；`table` 会被标记为候选值，后续必须通过 TiDB 元数据服务做 `resolve_table` 确认；冲突接近时会标记为 `needs_clarification`。当前版本先聚焦表级探查，字段级血缘后续再扩展。

环境变量：

- `DATA_AGENT_USE_LLM`：是否启用 LLM，默认 true。
- `DATA_AGENT_LLM_BASE_URL`：OpenAI-compatible base URL，默认 `http://localhost:1234/v1`。
- `DATA_AGENT_LLM_API_KEY`：API key，LM Studio 可使用任意非空字符串。
- `DATA_AGENT_LLM_MODEL`：模型名；未配置时自动走规则兜底。
- `DATA_AGENT_LLM_TIMEOUT_SECONDS`：请求超时，默认 30。
- `DATA_AGENT_LLM_MAX_RETRIES`：重试次数，默认 2。
