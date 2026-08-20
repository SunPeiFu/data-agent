# Data Agent 包架构

## 设计依据

本项目采用 `src` layout，并把 LangGraph 作为 application 层的工作流引擎，而不是让框架类型渗透到所有业务模块。

- LangGraph 官方项目模板使用 `src/<agent>/graph.py` 暴露可部署图入口，并将图定义与测试、配置分开：<https://github.com/langchain-ai/new-langgraph-project>
- LangGraph 官方 Retrieval Agent Template 将检索 Agent 放在独立包内，通过 `graph.py` 暴露工作流：<https://github.com/langchain-ai/retrieval-agent-template>
- OpenAI Agents Python SDK 按运行时、工具、guardrails、handoffs、tracing 等能力拆分核心包：<https://github.com/openai/openai-agents-python>
- Microsoft AutoGen 将 Agent、Team、Runtime、Memory、Model Client 分为独立包和子包：<https://github.com/microsoft/autogen>

这些项目的共同点不是某一套固定目录名，而是明确区分：公开入口、工作流编排、业务模型、模型理解、外部系统适配和可观测性。

## 当前目录

```text
src/data_agent/
├── application/execution/
│   └── nodes.py       # TaskStep DAG 执行与 Observation 收集节点
├── application/planning/
│   ├── graph.py       # LangGraph 节点注册、edge 和 conditional edge
│   ├── state.py       # PlannerState 状态契约
│   ├── service.py     # plan_question / resume_clarification 用例入口
│   ├── errors.py      # 澄清协议等应用层异常
│   └── nodes.py       # 节点实现与当前阶段的业务辅助函数
├── domain/
│   ├── models.py      # 领域枚举和 Pydantic 模型
│   ├── normalization.py
│   ├── slot_rules.py
│   └── task_builder.py
├── intelligence/
│   ├── classifier.py
│   ├── extractor.py
│   ├── hybrid_router.py
│   ├── llm_analyzer.py
│   └── llm_client.py
├── infrastructure/
│   ├── repositories/ # MySQL / Milvus / Neo4j Adapter
│   ├── security/     # Authorization Provider
│   ├── persistence/  # LangGraph Checkpointer
│   └── observability/# Trace Recorder
├── config/           # 环境配置和稳定项目路径
├── interfaces/       # CLI；未来 FastAPI 也放在这里
├── tools/            # LangChain Tool、强类型契约、Registry 和 Executor
├── __init__.py       # 稳定的 Python 公共 API
└── cli.py            # python -m data_agent.cli 兼容入口
```

## 依赖规则

```text
interfaces
    -> application
        -> domain
        -> intelligence
        -> infrastructure interfaces/protocols

infrastructure
    -> domain models
```

1. `domain` 不依赖 LangGraph、MySQL、Milvus 或具体 LLM Provider。
2. `graph.py` 只描述拓扑，不实现候选解析或权限算法。
3. `service.py` 负责一次用例的输入、恢复和返回，不维护节点细节。
4. `infrastructure` 实现外部系统访问；Planner 通过稳定类或 Protocol 使用这些能力。
5. 根包只暴露稳定公共 API，测试内部细节时从实际所属模块导入。
6. `tools` 依赖 Repository Protocol/实现完成真实调用；Planner 只产生 TaskStep，不直接持有 Tool。

## 面试表达

这不是为了增加目录数量，而是把不同变化频率隔离开：新增 LangGraph edge 主要改 `graph.py`，增加状态字段改 `state.py`，调整业务规则改 `domain`，更换 MySQL/Milvus/IAM/Trace 后端改 `infrastructure`。这样可以降低循环依赖和回归范围，也便于后续把单体 Planner 扩展为元数据、血缘、质量等多个 Agent。
