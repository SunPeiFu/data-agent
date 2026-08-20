# 智能数据探查

面向真实面试和代码学习的 Agent 项目。当前已实现从意图理解、实体治理、任务拆解到注册工具执行和 Observation 收集的完整表级 DataAgent 链路。

## 技术选型

- Python 3.11.15
- LangGraph 1.1.x + LangChain 1.2.x
- Pydantic
- pytest

## 代码架构

代码按职责划分为 `application`、`domain`、`intelligence`、`infrastructure`、
`config` 和 `interfaces`。LangGraph 拓扑、状态、用例入口和节点实现已经分离，根包只保留
稳定公共 API 和 CLI 兼容入口。完整目录、依赖规则和面试表达见
[`docs/package_architecture.md`](docs/package_architecture.md)。

### 包的概念定义

判断代码属于哪个包时，核心标准是“这段代码因为什么原因发生变化”，而不是它使用了
什么框架。例如，使用 Pydantic 的 HTTP 请求模型属于 `interfaces`，表达表标识的
Pydantic 模型属于 `domain`。

| 包 | 概念定义 | 当前职责 | 后续典型内容 |
| --- | --- | --- | --- |
| `data_agent` | 项目的稳定公共边界 | 对外暴露 `plan_question`、`resume_clarification` 等公共 API；保留 CLI 模块入口 | 只增加经过确认的稳定 API，不放具体业务实现 |
| `application` | 应用用例编排层，负责协调领域能力完成一次用户任务 | 承载规划、执行、Observation 收集和恢复澄清等用例 | 答案生成用例、人工转交流程 |
| `application.planning` | 智能数据探查 Planner 的独立 Agent/工作流边界 | 分离 Graph 拓扑、State 契约、Node 实现、Service 入口和应用异常 | 将来可与 `lineage_agent`、`quality_agent` 等并列 |
| `application.execution` | 已校验任务计划的执行编排层 | 执行 TaskStep DAG、收集工具 Observation 和 TraceEvent | 重规划、结果生成和人工审批 |
| `domain` | 与框架和数据库无关的核心业务知识 | 定义意图、实体、表标识、槽位、任务步骤等模型，以及归一化规则和计划模板 | 血缘深度策略、影响等级规则、业务术语规则 |
| `intelligence` | 将非结构化自然语言转换成结构化业务语义的智能理解层 | 规则预分析、LLM 结构化解析、Hybrid Router、冲突合并与兜底 | Prompt 版本、模型路由、意图评测、实体解析策略 |
| `infrastructure` | 对数据库、向量库、权限中心、存储和观测系统的技术适配层 | 汇总所有外部系统 Adapter，实现上层需要的能力边界 | Neo4j、TiDB、Redis、IAM、OpenTelemetry Adapter |
| `infrastructure.repositories` | 数据访问适配包 | 封装 MySQL 元数据查询、Milvus 混合召回、Neo4j 表级血缘和 Collection 初始化 | 真实 TiDB/Data Catalog Repository |
| `tools` | Agent 可执行能力边界 | LangChain Tool 输入输出契约、实现、Registry、动态权限发现和 DAG Executor | MCP Provider、远程工具目录和熔断策略 |
| `infrastructure.security` | 安全与权限系统适配包 | 通过统一 Provider 执行 subject-action-resource 权限判定 | IAM、Apache Ranger 或公司权限中心 HTTP Provider |
| `infrastructure.persistence` | Agent 工作流状态持久化适配包 | 提供 LangGraph SQLite Checkpointer | Postgres Checkpointer、会话与 Run Repository |
| `infrastructure.observability` | Agent 可观测性适配包 | 记录 Run、Node 和 TraceEvent，隔离 Trace 后端故障 | MySQL Trace Recorder、OpenTelemetry、LangSmith |
| `config` | 进程启动配置和部署环境适配包 | 集中读取环境变量、模型配置、Embedding 配置和项目路径 | 多环境 Settings、Secret Manager、配置中心客户端 |
| `interfaces` | 系统入站适配层，把外部协议转换成应用用例调用 | 解析 CLI 参数并调用 Application Service | FastAPI Router、消息队列 Consumer、定时任务入口 |

### Planning 包内部职责

| 文件 | 核心职责 |
| --- | --- |
| `graph.py` | 注册 LangGraph 节点、普通 Edge 和 Conditional Edge，只描述工作流拓扑 |
| `state.py` | 定义所有节点共享的 `PlannerState`，是节点间数据交换契约 |
| `service.py` | 提供开始问题规划和恢复澄清的应用入口，管理 thread 与 checkpoint 调用 |
| `nodes.py` | 实现意图识别、归一化、槽位校验、候选解析、权限和计划校验节点 |
| `errors.py` | 定义澄清协议错误等应用层可识别异常 |

### 依赖方向

```text
interfaces -> application -> domain
                         -> intelligence
                         -> infrastructure

infrastructure -> domain models
```

- `domain` 不依赖 LangGraph、MySQL、Milvus 或具体模型厂商。
- `graph.py` 只负责流程编排，不直接编写 SQL、向量检索或权限规则。
- `interfaces` 不绕过 `application service` 直接拼装节点状态。
- `infrastructure` 可以替换技术实现，但不能改变领域模型的业务含义。

## 快速运行

```bash
conda activate python-agent
pip install -e ".[dev]"

python -m data_agent.cli plan "安逸花业务线营销域下DWD层关于支付的表有哪些"
python -m data_agent.cli plan "安逸花 dwd 层 dwd.orderInfo 中的字段修改，对下游哪些表产生影响"
python -m data_agent.cli plan "营销域下的 dwd层的userInfo表修改字段，关联影响的上游和下游表有哪些"

# 显式进入真实工具执行分支
python -m data_agent.cli run "查询 dwd.orderInfo 表说明和负责人"
python -m data_agent.cli run "安逸花业务线营销域下DWD层关于支付的表有哪些"
```

权限演示可以通过角色切换。`data_analyst` 只能读取营销域元数据和血缘，
`data_admin` 可执行全部动作：

```bash
python -m data_agent.cli plan \
  "营销域 DWD 层 dwd.payment_detail 表的下游血缘有哪些" \
  --user-id analyst-1 --role data_analyst
```

遇到多候选时，结果中的 `clarification_request` 会返回 `thread_id`、
`clarification_id`、`state_version` 和候选 `option_id`。提交选择后，LangGraph 从
SQLite checkpoint 恢复，并重新执行元数据和权限校验：

```bash
python -m data_agent.cli resume \
  --thread-id "卡片中的 thread_id" \
  --clarification-id "卡片中的 clarification_id" \
  --option-id "选项中的 option_id" \
  --value "dwd.orderInfo" \
  --state-version 1 \
  --idempotency-key "request-001"
```

checkpoint 默认写入 `.data-agent/checkpoints.sqlite3`，可通过
`DATA_AGENT_CHECKPOINT_DB` 修改路径。

## 接入本地 LM Studio

项目使用 OpenAI-compatible `/v1/chat/completions` 协议。LM Studio 启动本地服务后，配置：

```bash
export DATA_AGENT_USE_LLM=true
export DATA_AGENT_LLM_BASE_URL="http://localhost:1234/v1"
export DATA_AGENT_LLM_API_KEY="lm-studio"
export DATA_AGENT_LLM_MODEL="你的本地模型名"

python -m data_agent.cli plan "营销域下的 dwd层的userInfo表修改字段，关联影响的上游和下游表有哪些"
```

如果没有配置 `DATA_AGENT_LLM_MODEL`，系统会自动使用离线规则兜底，方便测试和演示。

## 启动 Milvus

项目使用独立容器和数据卷，复用本机已有的 Milvus 2.5.14 镜像：

```bash
docker compose -f docker-compose.milvus.yml up -d
conda run -n python-agent python -m data_agent.infrastructure.repositories.milvus_schema
```

本地 Embedding 使用 LM Studio 的 OpenAI-compatible `/v1/embeddings`：

```bash
export DATA_AGENT_EMBEDDING_BASE_URL="http://127.0.0.1:1234/v1"
export DATA_AGENT_EMBEDDING_API_KEY="lm-studio"
export DATA_AGENT_EMBEDDING_MODEL="你的 embedding 模型 ID"
export DATA_AGENT_EMBEDDING_DIM="1024"
```

Milvus SDK 地址为 `http://127.0.0.1:19531`，WebUI 为 `http://127.0.0.1:9092/webui/`。

## 测试

```bash
pytest -q
```

## 当前边界

`plan` 只返回经过结构化执行闸门批准的计划；闸门校验 DAG、Schema、数据流、权限和运行策略，并用 plan hash 防止校验后篡改。`run` 会执行 TiDB/MySQL 元数据、Milvus 混合召回、Neo4j 血缘、影响分析和结果融合工具，并返回 `tool_observations/final_output`。Neo4j 需要通过 `DATA_AGENT_NEO4J_*` 配置真实实例。最终自然语言答案生成、MCP 和 FastAPI 服务层仍在 TODO 中。
