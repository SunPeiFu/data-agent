# 实体归一化设计说明

## 核心结论

LangChain / LangGraph 不直接提供企业数据资产归一化组件。它们提供的是结构化输出、工具调用、图编排和状态管理；而 `订单信息表 -> order_info`、`支付明细表 -> payment_detail`、`支付相关 -> 支付` 这类映射属于企业数据语义层，必须由业务词典、表级业务术语、元数据服务或治理平台提供。

因此，本项目把归一化设计成独立节点 `_normalize_entities`，而不是藏在 prompt 或工具调用里。

## 当前实现能力

- 配置化 stopwords / synonyms / table_terms：见 `config/normalization.yml`。
- 表级业务术语映射：例如 `订单信息表 -> order_info`。
- 业务同义词映射：例如 `支付相关 -> 支付`。
- 业务术语类型化：使用 `NormalizedTerm` 区分 `business_term`、`metric`、`entity`、`table_term`。
- 归一化审计：使用 `NormalizationTrace` 记录 before、after、rule、source。
- 防误匹配：英文 alias 使用边界匹配，避免 `orderInfo` 中的 `order` 被误识别。

## 为什么要独立成节点

生产级 DataAgent 不能把 LLM/规则抽取结果直接交给工具层。原因：

- LLM 抽取可能不稳定。
- 用户会使用中文、英文、别名、简称、业务术语混合表达。
- 工具层需要的是标准参数，比如候选物理表、标准主题词、规范表名。
- 归一化过程必须可解释、可审计、可评估。

独立节点的价值：

```text
raw entities
  -> normalize_entities
  -> metadata resolution
  -> authorization
  -> task planning
```

## 面试表达

可以这样讲：

```text
LangChain 和 LangGraph 主要解决结构化输出、工具调用和流程编排问题，但不会内置企业数据资产的业务词典和表术语映射。像“订单信息表”映射到 order_info、“支付相关”映射到标准业务词“支付”，属于企业数据语义层能力。

所以我把 normalize_entities 设计成独立节点，通过配置化 stopwords、synonyms 和 table_terms 做基础归一化，并输出 NormalizedTerm 和 NormalizationTrace。这样既能保证下游工具参数稳定，又能让每次归一化都有 before/after/rule/source，方便排查、审计和后续接入 DataHub/OpenMetadata 或内部元数据平台。
```

## 简历表达

可写成：

```text
设计企业数据探查 Agent 的实体归一化层，基于配置化 stopwords、业务同义词和表级业务术语映射，将用户自然语言中的业务表达标准化为可被元数据、血缘和检索工具消费的结构化实体；引入 NormalizedTerm 和 NormalizationTrace，支持归一化过程的可解释、可审计和后续词典迭代。
```

更短版本：

```text
构建 DataAgent 实体归一化机制，支持表级业务术语、业务同义词、检索关键词的标准化映射，并保留 before/after/rule/source 级别审计 trace，提升工具调用参数稳定性和问题可排查性。
```

## 后续生产化方向

- 接入真实 glossary service。
- 接入真实 metadata catalog。
- 增加更完整的表别名和表业务术语映射。
- 增加多候选 disambiguation。
- 增加词典版本管理。
- 增加归一化命中率、误归一率、人工反馈闭环。
