# 智能数据探查 1 天速通学习手册

这份计划只安排 1 天，目标是：能跑通项目、能讲清楚核心设计、能应对面试追问。不要追求每行代码都背下来，优先掌握“用户问题如何变成工具调用计划”这条主线。

## 总目标

一天结束后，你要能讲清楚：

- 这个项目为什么不是普通 RAG，而是 Agent Planner。
- TiDB、Milvus、Neo4j 三类工具分别负责什么。
- 意图识别、实体抽取、任务拆解分别解决什么问题。
- 为什么血缘查询统一是 `lineage_search`，上下游用 `direction` 参数控制。
- 为什么一段式表名必须先 `resolve_table`。
- 三个面试 case 最终会生成什么执行计划。

## 第 0 小时：准备环境和跑通项目

预计时间：30 分钟

先运行：

```bash
conda activate python-agent
cd /Users/sunpeifualiyun.com/Desktop/agent_study_workspace/codex_workspace/data-agent
pytest -q
```

再跑三个核心 case：

```bash
python -m data_agent.cli plan "安逸花业务线营销域下DWD层关于支付的表有哪些"

python -m data_agent.cli plan "安逸花 dwd 层 dwd.orderInfo 表修改，对下游哪些表产生影响"

python -m data_agent.cli plan "营销域下的 dwd层的userInfo表修改，关联影响的上游和下游表有哪些"
```

你先不要看源码，只观察输出里的三个字段：

- `intent`
- `entities`
- `task_steps`

这一小时的目标：

```text
知道这个项目最终输出的是 PlanningResult JSON，而不是直接生成自然语言答案。
```

## 第 1 小时：先理解项目定位

预计时间：45 分钟

先看：

```text
README.md
docs/interview_notes.md
docs/design.md
```

只抓 3 个核心判断：

1. 这是智能数据探查，不是知识库问答。
2. RAG 只负责语义补充，不能代替血缘图查询。
3. v1 只做规划层，不接真实 TiDB / Milvus / Neo4j。

必须背下来的面试表达：

```text
这个项目不是把用户问题直接丢给大模型回答，而是先通过 Agent Planner 识别意图、抽取实体、生成工具调用计划。TiDB 用来查结构化元数据，Milvus 用来补充表说明和业务语义，Neo4j 用来做确定性的血缘关系查询。
```

这一小时不要深挖代码，先确保能讲清楚“为什么这样做”。

## 第 2 小时：看数据结构，理解系统输入输出

预计时间：60 分钟

重点看：

```text
src/data_agent/models.py
```

优先级从高到低：

1. `PlanningResult`
2. `TaskStep`
3. `ExtractedEntities`
4. `IntentType`
5. `LineageDirection`
6. `TableIdentifier`
7. `DomainType` / `DataLayer` / `OperationType`

你只需要记住：

```text
PlanningResult = intent + entities + task_steps + clarification + notes
```

重点理解：

- `IntentType` 只有三类核心意图：
  - `metadata_search`
  - `lineage_search`
  - `impact_analysis`
- 上游、下游不做成两个意图，而是放在：
  - `LineageDirection.UPSTREAM`
  - `LineageDirection.DOWNSTREAM`
  - `LineageDirection.BOTH`
- `TableIdentifier` 支持：
  - 三段式：`catalog.dwd.orderInfo`
  - 两段式：`dwd.orderInfo`
  - 一段式：`userInfo`

快速练习：

```bash
python - <<'PY'
from data_agent.models import TableIdentifier

for name in ["catalog.dwd.orderInfo", "dwd.orderInfo", "userInfo"]:
    print(TableIdentifier.parse(name).model_dump())
PY
```

这一小时的目标：

```text
看到任意 CLI 输出时，你能说清楚每个字段是什么意思。
```

## 第 3 小时：看意图识别和实体抽取

预计时间：75 分钟

先看：

```text
src/data_agent/classifier.py
```

只记三条规则：

- “有哪些表 / 表说明 / 业务含义” -> `metadata_search`
- “上游 / 下游 / 依赖 / 血缘” -> `lineage_search`
- “修改字段 / 变更 / 影响” -> `impact_analysis`

再看：

```text
src/data_agent/extractor.py
```

重点看这些函数：

- `_extract_domain`
- `_extract_layer`
- `_extract_table`
- `_extract_operation`
- `_extract_lineage_direction`

你要讲清楚：

```text
意图识别负责判断用户想干什么，实体抽取负责把问题里的业务线、主题域、数仓分层、表名、操作类型和血缘方向提取出来。当前学习阶段先聚焦表级数据探查。
```

快速练习：

```bash
python - <<'PY'
from data_agent.classifier import RuleBasedIntentClassifier
from data_agent.extractor import extract_entities

q = "营销域下的 dwd层的userInfo表修改，关联影响的上游和下游表有哪些"
print(RuleBasedIntentClassifier().classify(q))
print(extract_entities(q).model_dump())
PY
```

这一小时的目标：

```text
能解释自然语言问题如何被转成结构化 entities。
```

## 第 4 小时：重点看任务拆解，这是最重要部分

预计时间：90 分钟

重点看：

```text
src/data_agent/task_builder.py
```

这是整个项目最值得面试展开讲的文件。

学习顺序：

1. `build_task_plan`
2. `_metadata_steps`
3. `_lineage_steps`
4. `_impact_steps`
5. `_resolve_table_step`
6. `_validate_slots`

你要背住三种计划模板。

### 模板 1：元数据查询

用户问：

```text
安逸花业务线营销域下DWD层关于支付的表有哪些
```

计划：

```text
tidb_metadata.filter_tables
milvus_rag.semantic_search
result_ranker.merge_and_rank
```

讲法：

```text
TiDB 负责结构化过滤业务线、主题域、分层；Milvus 负责语义召回支付相关表说明；最后做融合排序。
```

### 模板 2：单向下游影响分析

用户问：

```text
安逸花 dwd 层 dwd.orderInfo 表修改，对下游哪些表产生影响
```

计划：

```text
tidb_metadata.resolve_table
neo4j_lineage.lineage_search(direction=downstream)
impact_analyzer.classify_impact
```

讲法：

```text
先 resolve 表，避免表名不规范导致查错；再用 Neo4j 查下游血缘；最后按直接影响和间接影响做分级。
```

### 模板 3：上下游双向影响分析

用户问：

```text
营销域下的 dwd层的userInfo表修改，关联影响的上游和下游表有哪些
```

计划：

```text
tidb_metadata.resolve_table
neo4j_lineage.lineage_search(direction=both)
milvus_rag.semantic_search
impact_analyzer.merge_lineage_and_metadata
```

讲法：

```text
一段式表名 userInfo 不能直接判定唯一，所以先通过 TiDB 结合营销域和 DWD 分层定位候选表。表定位后，Neo4j 查询双向血缘，同时 Milvus 补充业务语义，最后融合成影响分析结果。
```

这一小时的目标：

```text
能不看代码讲清楚三个 case 的 task_steps。
```

## 第 5 小时：看 LangGraph 编排

预计时间：45 分钟

重点看：

```text
src/data_agent/planner.py
```

只记住这条图：

```text
classify_intent
  -> extract_entities
  -> normalize_entities
  -> validate_slots
  -> resolve_metadata_candidates
  -> authorize_context
  -> post_validate_slots
  -> decide_clarification_or_continue
  -> build_task_plan | return_clarification_result | return_forbidden_result
  -> return_planning_result
```

面试讲法：

```text
我用 LangGraph 把 Agent Planner 显式建模成状态图。当前版本已经把槽位校验、元数据候选解析、权限校验和澄清决策拆成独立节点，并通过 conditional edge 控制继续规划、澄清或拒绝；后续可以继续接真实工具执行、失败重试、trace 日志和并行节点。
```

不要在这一小时纠结 LangGraph 高级语法。面试里重点不是炫框架，而是说明你知道为什么用状态图组织 Agent 流程。

## 第 6 小时：看测试，反向理解需求

预计时间：45 分钟

重点看：

```text
tests/test_planner.py
```

测试就是项目需求的压缩版。重点看这些断言：

- Case 1 必须输出 `metadata_search`
- Case 2 必须输出 `impact_analysis + direction=downstream`
- Case 3 必须输出 `impact_analysis + direction=both`
- 表名支持三段式、两段式、一段式
- 缺表名时要触发澄清

运行：

```bash
pytest -q
```

这一小时的目标：

```text
能用测试证明你不是随便写 demo，而是把关键行为固化下来了。
```

## 第 7 小时：面试表达训练

预计时间：60 分钟

你要反复练这段：

```text
我做的是一个面向数据资产探查的 Agent Planner。用户输入自然语言后，系统先做意图识别，区分元数据查询、血缘查询和变更影响分析；然后做实体抽取，提取业务线、主题域、数仓分层、表名、操作类型和血缘方向；最后由任务拆解模块生成工具调用计划。

比如用户问“营销域下的 DWD 层 userInfo 表修改，关联影响的上游和下游表有哪些”，系统会识别为 impact_analysis，direction 是 both。因为 userInfo 是一段式表名，所以第一步必须调用 TiDB 元数据工具 resolve_table；表定位后调用 Neo4j 的 lineage_search(direction=both) 查双向血缘，同时调用 Milvus 做表说明和业务语义召回，最后由 impact_analyzer 合并血缘和元数据语义。
```

再准备 5 个追问答案。

### 追问 1：为什么不是纯 RAG？

```text
因为血缘关系是确定性图关系，不能靠向量召回猜。RAG 适合补充表说明和业务语义，Neo4j 才适合查上下游依赖。
```

### 追问 2：为什么上游下游不是两个 intent？

```text
上游和下游本质都是血缘查询，只是查询方向不同。所以意图层统一成 lineage_search，方向下沉到 LineageDirection 参数，这样意图空间更稳定，工具扩展也更简单。
```

### 追问 3：一段式表名为什么要 resolve？

```text
因为 userInfo 这种一段式表名在不同库、不同分层、不同主题域下可能重复。必须结合业务线、主题域、数仓分层先定位候选表，避免直接查错血缘。
```

### 追问 4：为什么当前先做表级？

```text
表级探查是数据地图最核心的入口，能先解决表定位、表说明、上下游血缘和表变更影响。字段级血缘依赖字段字典、字段映射和更细粒度的血缘数据，复杂度更高，所以当前版本先把表级链路做稳，字段级作为后续演进。
```

### 追问 5：为什么用 LangGraph？

```text
因为这个场景不是单轮问答，而是多阶段状态流转。LangGraph 可以把分类、抽取、校验、规划、后续工具执行都建模成图节点，方便扩展失败重试、人机澄清、trace 和并行执行。
```

## 当天最低通过标准

如果时间非常紧，至少完成这 4 件事：

1. 跑通三个 CLI case。
2. 看懂 `models.py` 里的 `PlanningResult`。
3. 看懂 `task_builder.py` 里的三种任务模板。
4. 背熟第 7 小时的面试表达。

## 文件阅读优先级

按优先级排序：

1. `docs/interview_notes.md`
2. `src/data_agent/models.py`
3. `src/data_agent/task_builder.py`
4. `src/data_agent/extractor.py`
5. `src/data_agent/classifier.py`
6. `src/data_agent/planner.py`
7. `tests/test_planner.py`
8. `src/data_agent/cli.py`
9. `docs/design.md`
10. `docs/todo.md`

真正面试最常用的是前 5 个。
