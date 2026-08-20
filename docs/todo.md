# 智能数据探查 TODO List

这份 TODO 的目标是帮助你先整体串通项目骨架，再逐步把 Planner 层升级成真实生产 DataAgent。当前项目已经完成：Hybrid Router、实体标准化、结构化槽位校验、MySQL 元数据候选解析、Milvus 混合召回 Repository 与 Collection Schema、YAML RBAC 权限校验、条件澄清决策、任务计划生成、计划校验和 trace notes。

## P0：先串通现有骨架

- [ ] 跑通三个核心 CLI case，确认 `intent/entities/task_steps/notes` 都能看懂。
- [ ] 按顺序阅读 `planner.py`，理解企业级 DAG：
  `classify_intent -> extract_entities -> normalize_entities -> validate_slots -> resolve_metadata_candidates -> authorize_context -> post_validate_slots -> decide_clarification_or_continue -> build_task_plan/return_clarification_result/return_forbidden_result -> validate_task_plan -> attach_trace -> return_planning_result`
- [ ] 重点理解 `hybrid_router.py` 的候选合并模型：
  `RuleEvidence -> FieldCandidate -> ResolvedField -> EntityResolution`
- [ ] 重点理解 `task_builder.py` 的三类计划模板：
  `metadata_search`、`lineage_search`、`impact_analysis`
- [ ] 用 `pytest -q` 保证现有测试全部通过。
- [x] 给 `normalize_entities` 增加生产化基础能力：
  - [x] 配置化 stopwords / synonyms / table_terms。
  - [x] 同义词和表级业务术语映射。
  - [x] 业务术语类型化 `NormalizedTerm`。
  - [x] 归一化审计 `NormalizationTrace`。
- [x] 给 `validate_slots` 增加生产化基础能力：
  - [x] 新增结构化 `SlotValidationResult`。
  - [x] 拆分元数据解析前 `validate_slots` 和解析后 `post_validate_slots`。
  - [x] 基于 `config/slot_rules.yml` 配置 intent 槽位规则。
  - [x] 基于 LangGraph conditional edge 路由到继续、澄清或拒绝。

## P1：补真实工具接入

- [ ] 建立 `ToolRegistry`，注册工具名、action、输入 schema、输出 schema、超时、重试策略和权限 scope。
- [ ] 接入 TiDB 元数据工具：
  - [ ] `filter_tables`
  - [x] `resolve_table` 候选解析已先用 MySQL `meta_table` / `meta_table_ext` 落地。
  - [x] 支持一段式、两段式、三段式表名解析。
  - [x] 支持表级业务术语到候选物理表的解析。
- [ ] 完成 Milvus 表资产数据同步与执行闭环（Collection Schema 和查询 Repository 已完成）：
  - [x] 实现 `MilvusMetadataRepository.hybrid_search` 查询接口。
  - [ ] 将表说明、表级业务术语及 embedding 增量同步到 Collection。
  - [x] 增加 hybrid search：Dense + BM25 + 结构化过滤 + RRF。
  - [ ] 增加离线召回评测集、阈值标定和线上检索指标。
- [ ] 接入 Neo4j 血缘工具：
  - [ ] `lineage_search(direction=upstream/downstream/both)`
  - [ ] 支持 `depth`
  - [ ] 支持表级血缘。
- [ ] 后续把 MySQL 版 `resolve_metadata_candidates` 替换为真实 TiDB / 数据目录查询。
- [ ] 元数据模型补充 platform/environment/tenant 后，将三者加入候选身份一致性和权限域校验。
- [ ] 将当前 YAML `AuthorizationProvider` 替换为企业 IAM/Ranger/统一权限中心，并在 TiDB、Milvus、Neo4j 工具执行端二次鉴权。

## P2：补 LangGraph 执行闭环

- [ ] 继续增强 `validate_task_plan` 的生产校验：
  - [ ] 校验工具 output schema 和下游 input schema 是否匹配。
  - [ ] 校验权限 scope 是否满足每个工具 action。
  - [ ] 校验每个 step 是否具备 timeout、retry、fallback 策略。
  - [ ] 校验 cost budget、并发上限和查询深度是否符合业务策略。
  - [ ] 校验 trace_id、planner_version、rule evidence、entity resolution 是否完整落审计。
- [x] 给 `decide_clarification_or_continue` 增加 conditional edge：
  - [x] 缺槽位 -> `return_clarification_result`
  - [x] 多候选 -> `return_clarification_result`
  - [x] 无权限 -> `return_forbidden_result`
  - [x] 澄清问题跨 pre/post 去重并按槽位依赖排序。
  - [x] 返回结构化候选卡片并保留 pending issues。
  - [x] 信息充分 -> `build_task_plan`
- [ ] 增加 `execute_task_plan` 节点，根据 `TaskStep.depends_on` 执行工具。
- [ ] 增加并行执行能力：
  - [ ] 相同 `parallel_group` 的步骤并行执行。
  - [ ] 失败时能隔离单个工具错误。
- [ ] 增加 `collect_observations` 节点，统一收集 TiDB/Milvus/Neo4j 工具结果。
- [ ] 增加 `replan_or_continue` 节点：
  - [ ] 表名候选过多 -> 澄清
  - [ ] 查询无结果 -> 改写 query 或扩大检索范围
  - [ ] 工具失败 -> 重试或降级
- [ ] 增加失败重试、超时、限流和降级策略。

## P3：补结果生成和业务闭环

- [ ] 增加 `answer_generator`：
  - [ ] 元数据搜索输出表格。
  - [ ] 血缘查询输出上游/下游分层结果。
  - [ ] 影响分析输出直接影响、间接影响、风险等级。
- [ ] 增加引用和可解释性：
  - [ ] 表说明来源
  - [ ] 血缘路径来源
  - [ ] RAG 召回片段来源
- [ ] 后续再增加字段级影响分析：
  - [ ] 字段存在性校验
  - [ ] 字段级 lineage
  - [ ] 字段变更类型影响判断
  - [ ] 字段级无血缘时降级表级分析
- [ ] 增加结果融合：
  - [ ] TiDB 精确元数据
  - [ ] Milvus 语义召回
  - [ ] Neo4j 血缘路径
  - [ ] 影响等级排序

## P4：补生产治理和可观测性

### Trace 分阶段改造

- [x] 第一阶段：结构化 Trace 骨架。
  - [x] 增加 `TraceContext`、`NodeTrace`、`TraceEvent` 和 Run/Node 状态枚举。
  - [x] 增加 `init_trace_context`，在第一个业务节点前创建 trace_id/run_id。
  - [x] 增加 `TraceRecorder`、`LoggingTraceRecorder` 和 `InMemoryTraceRecorder`。
  - [x] 使用 `traced_node` 统一记录业务节点成功、失败、耗时和状态字段摘要。
  - [x] 节点异常时以 failed 关闭 Run，Recorder 故障不阻断 Agent 主流程。
  - [x] 将 `_attach_trace` 改造成 Run 最终状态收口节点。
  - [x] 澄清 Resume 创建新 trace_id/run_id，并通过 parent_run_id 关联上一轮。
- [ ] 第二阶段：接入核心业务证据和脱敏策略。
  - [ ] 为 `classify_intent` 记录 rule/LLM/final intent、置信度、来源和冲突处理结果。
  - [ ] 为 `extract_entities` 记录字段级 rule/LLM/selected value、selected_source 和 resolution。
  - [ ] 把 `normalization_traces` 转换为结构化 TraceEvent。
  - [ ] 为 MySQL/Milvus Repository 记录 operation、路由、耗时、候选数量、fallback 和错误码。
  - [ ] 为 `authorize_context` 记录 action/decision/reason_code/policy_id/resource_hash。
  - [ ] 在 LLM Client 层增加 model_call trace，记录模型、Prompt 版本、Token、耗时和解析状态。
  - [ ] 增加统一 Trace 脱敏器和字段 allowlist，禁止保存 API Key、完整 Prompt、Embedding 和无权资产名。
- [ ] 第三阶段：Trace 物理表和持久化 Recorder。
  - [ ] 初始化 `agent_run`、`agent_node_run`、`agent_model_call`、`agent_audit_event`。
  - [ ] 接入真实工具执行后初始化 `agent_tool_call`。
  - [ ] 实现 `MySQLTraceRecorder`，业务节点不依赖具体持久层。
  - [ ] 为 run_id/node_run_id/model_call_id/event_id 建唯一索引。
  - [ ] 为 thread/run/node/model/event 常用查询建立联合索引。
  - [ ] 增加批量写入、失败缓冲、可靠审计 outbox、归档和数据保留策略。
- [ ] 第四阶段：标准可观测性和 Trace 产品能力。
  - [ ] 对接 OpenTelemetry 或 LangSmith，映射 Run/Node/Model/Tool Span。
  - [ ] 建立节点耗时、失败率、LLM Token 成本、澄清轮数和工具成功率指标。
  - [ ] 增加 `/trace/{trace_id}` 和 `/thread/{thread_id}/runs` 查询接口。
  - [ ] 增加 Trace 瀑布流，展示父子 Run、节点耗时、模型调用和工具调用。
  - [ ] 增加采样率、错误全采样、权限事件全采样和告警策略。

### Agent 运行数据持久化

以下表只服务于 Agent 平台自身运行，不包含 TiDB/MySQL 元数据、Milvus 索引、Neo4j 血缘等业务或工具依赖数据。

#### 第一批：当前 Planner/HITL 阶段

- [ ] 设计并初始化 `agent_thread`：
  - [ ] 使用 `thread_id` 关联 LangGraph checkpoint 和多轮业务会话。
  - [ ] 保存 tenant/user/agent/status/current_run/last_active/expires_at。
  - [ ] 区分业务会话索引与 LangGraph 完整状态快照，避免重复保存 checkpoint state。
- [ ] 设计并初始化 `agent_message`：
  - [ ] 保存 user/assistant/system/tool 消息、message_type、sequence_no 和 run_id。
  - [ ] 文本与结构化 `content_json` 分开保存，大结果只保存对象存储引用。
- [ ] 设计并初始化 `agent_run`：
  - [ ] 一个 thread 支持 initial/resume/retry/replan 多次 run。
  - [ ] 保存 intent/confidence/status/planner_version/state_version/耗时和错误信息。
  - [ ] 通过 parent_run_id 记录恢复、重试和重规划关系。
- [ ] 设计并初始化 `agent_node_run`：
  - [ ] 记录 LangGraph node_name、执行顺序、状态、耗时、重试和错误。
  - [ ] input/output 只保存脱敏摘要，避免完整状态和敏感数据重复落库。
- [ ] 设计并初始化 `agent_clarification`：
  - [ ] 保存 clarification_id/thread_id/request_run_id/response_run_id。
  - [ ] 保存 slot/issue/question/options/selected_value/round/state_version/status。
  - [ ] 为 clarification_id 和 idempotency_key 建唯一索引。
  - [ ] 将当前 SQLite checkpoint 中的恢复状态与业务澄清审计明确分层。
- [ ] 设计并初始化 `agent_audit_event`：
  - [ ] 以 append-only 方式记录权限允许/拒绝、澄清、人工转交和计划校验事件。
  - [ ] 保存 event_type/action/resource_hash/decision/reason_code/policy_id。
  - [ ] 对资源名称和请求详情做脱敏，设置独立保留周期。

#### 第二批：工具和模型执行阶段

- [ ] 设计并初始化 `agent_tool_call`：
  - [ ] 保存 tool_call_id/run_id/node_run_id/step_id/tool/action/status/耗时。
  - [ ] 保存脱敏后的 request/response 摘要、超时、重试和错误信息。
  - [ ] tool_call_id 同时作为工具执行端幂等标识。
- [ ] 设计并初始化 `agent_model_call`：
  - [ ] 保存 provider/model/prompt_version/token 用量/耗时/状态。
  - [ ] Prompt 和响应默认只保存哈希、版本及脱敏摘要。
  - [ ] 支持统计模型成本、失败率、规则兜底率和不同模型效果。

#### 第三批：运营和人工闭环

- [ ] 设计并初始化 `agent_feedback`：
  - [ ] 保存 like/dislike/score/comment/corrected_value。
  - [ ] 用于沉淀意图、候选表、澄清和最终答案的评测样本。
- [ ] 设计并初始化 `agent_handoff`：
  - [ ] 保存 handoff_id/thread_id/run_id/reason/pending_issues/assignee/status。
  - [ ] 对接最大澄清轮数、权限申请、重试超限和人工数据服务台。

#### 建表边界

- [ ] 不自建 `agent_checkpoint`/`agent_state`，继续由 LangGraph Checkpointer 管理完整运行状态。
- [ ] 不单独建 `agent_idempotency`，幂等键分别落在 clarification/tool_call 等业务记录中。
- [ ] 不自建 Agent 用户和角色表，生产复用企业 IAM/Ranger/统一权限中心。
- [ ] 不单独建一张笼统的 `agent_trace`，由 run/node/model/tool/audit 记录共同组成完整 Trace。

- [ ] 增加完整 trace：
  - [ ] `trace_id`
  - [ ] rule evidence
  - [ ] LLM structured output
  - [ ] entity resolution
  - [ ] task plan
  - [ ] tool observations
  - [ ] final answer
- [ ] 增加日志分层：
  - [ ] planner log
  - [ ] tool log
  - [ ] audit log
  - [ ] error log
- [ ] 增加指标：
  - [ ] intent accuracy
  - [ ] entity resolution accuracy
  - [ ] tool success rate
  - [ ] clarification rate
  - [ ] answer acceptance rate
  - [ ] latency
  - [ ] token cost
- [ ] 增加安全治理：
  - [ ] 权限校验
  - [ ] 操作审计
  - [ ] 拒答策略

## P5：补服务化和面试 Demo

- [ ] 增加 FastAPI 服务层：
  - [ ] `/plan`
  - [ ] `/execute`
  - [ ] `/trace/{trace_id}`
- [ ] 增加本地 mock 数据集：
  - [ ] 表元数据
  - [ ] 表级血缘
  - [ ] RAG 文档片段
- [ ] 增加 README demo 脚本。
- [ ] 增加架构图：
  - [ ] Planner DAG
  - [ ] Tool execution DAG
  - [ ] DataAgent 总体架构
- [ ] 增加面试讲解材料：
  - [ ] 项目背景
  - [ ] 核心职责描述
  - [ ] 三个 case 的完整链路
  - [ ] 生产化差距和后续演进

## 建议推进顺序

1. 先完全吃透 P0。
2. 然后做 P1 的 TiDB mock/真实元数据工具，因为表名解析是一切血缘分析的前置。
3. 再做 P2 的执行闭环，让 `task_steps` 真正跑起来。
4. 最后做 P3/P4/P5，把系统从“可规划”升级成“可执行、可解释、可观测”。
