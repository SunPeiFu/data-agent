# 智能数据探查

面向真实面试和代码学习的 Agent 项目。当前已实现意图识别、实体标准化、槽位校验、MySQL 元数据候选解析，以及 Milvus 表资产混合召回接入。

## 技术选型

- Python 3.11
- LangGraph + LangChain
- Pydantic
- pytest

## 快速运行

```bash
conda activate python-agent
pip install -e ".[dev]"

python -m data_agent.cli plan "安逸花业务线营销域下DWD层关于支付的表有哪些"
python -m data_agent.cli plan "安逸花 dwd 层 dwd.orderInfo 中的字段修改，对下游哪些表产生影响"
python -m data_agent.cli plan "营销域下的 dwd层的userInfo表修改字段，关联影响的上游和下游表有哪些"
```

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
conda run -n python-agent python -m data_agent.milvus_schema
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

当前输出的是结构化执行计划：

- TiDB 元数据查询计划
- Milvus RAG 语义召回计划
- Neo4j 血缘查询计划
- 影响分析与结果融合计划

真实工具调用、最终答案生成、FastAPI 服务层会在后续阶段落地。
