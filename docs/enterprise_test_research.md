# 企业级 DataAgent 场景测试调研映射

本项目没有直接复制开源项目测试代码，因为 DataHub、OpenMetadata、Amundsen 的实现栈和接口与本项目不同。这里采用的是“能力映射式测试”：调研企业数据目录/血缘系统的核心能力后，抽象成智能数据探查 Agent 的输入输出验证场景。

## 参考能力

- DataHub：支持 table-level lineage、column-level lineage、upstream/downstream lineage、structured filters。
- OpenMetadata：支持表级和列级 lineage、上下游深度、pipeline/dashboard 等下游实体追踪。
- Amundsen：定位是 data discovery / metadata engine，核心能力是搜索表、列、描述、数据资产。

## 映射到本项目测试

| 企业级能力 | 本项目测试 |
|---|---|
| 数据目录搜索表、列、描述 | `test_data_catalog_search_uses_metadata_filters_and_semantic_recall` |
| 上游/下游血缘方向和深度 | `test_lineage_query_supports_configurable_upstream_downstream_direction` |
| 表级影响分析 | `test_table_level_impact_analysis_uses_table_granularity` |
| 一段式表名不能直接信任，需要 metadata resolution | `test_one_part_table_name_is_candidate_requiring_metadata_resolution` |
| 规则与 LLM 候选冲突 | `test_policy_resolver_requests_clarification_for_close_table_candidates` |
| 表名需要后续元数据校验 | `test_policy_resolver_marks_table_as_metadata_validation_candidate` |

## 测试原则

- 不只测 intent，还要测 `task_steps` 是否路由到正确工具。
- 表名只作为候选值，最终必须由元数据服务确认。
- 当前阶段聚焦 table-level 场景，血缘计划必须生成 `lineage_granularity=table`。
- 上游/下游不拆 intent，通过 `lineage_search(direction)` 控制。
- 模型不可用时仍可通过规则预分析兜底。
