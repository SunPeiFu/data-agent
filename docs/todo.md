# 智能数据探查 TODO List

这份 TODO 的目标是帮助你先整体串通项目骨架，再逐步把 Planner 层升级成真实生产 DataAgent。当前项目已经完成：Hybrid Router、实体标准化、结构化槽位校验、MySQL 元数据候选解析、mock 权限校验、条件澄清决策、任务计划生成、计划校验和 trace notes。

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
  - [ ] 支持一段式、两段式、三段式表名解析。
  - [ ] 支持表级业务术语到候选物理表的解析。
- [ ] 接入 Milvus RAG 工具：
  - [ ] `semantic_search`
  - [ ] 支持表说明、表级业务术语召回。
  - [ ] 增加 hybrid search：结构化过滤 + 向量召回。
- [ ] 接入 Neo4j 血缘工具：
  - [ ] `lineage_search(direction=upstream/downstream/both)`
  - [ ] 支持 `depth`
  - [ ] 支持表级血缘。
- [ ] 后续把 MySQL 版 `resolve_metadata_candidates` 替换为真实 TiDB / 数据目录查询。
- [ ] 把 mock 的 `authorize_context` 替换为真实权限、业务域隔离和审计策略。

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
