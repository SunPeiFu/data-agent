from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from data_agent.intelligence.llm_client import LLMClientError, OpenAICompatibleChatClient
from data_agent.domain.models import (
    DataLayer,
    DomainType,
    ExtractedEntities,
    IntentType,
    LineageDirection,
    OperationType,
    TableIdentifier,
)


class LLMAnalysis(BaseModel):
    intent: IntentType
    confidence: float = Field(ge=0.0, le=1.0)
    biz_line: str | None = None
    domain: DomainType | None = None
    data_layer: DataLayer | None = None
    table_name: str | None = None
    field_name: str | None = None
    operation: OperationType | None = None
    topic_keywords: list[str] = Field(default_factory=list)
    lineage_direction: LineageDirection | None = None

    def to_entities(self) -> ExtractedEntities:
        return ExtractedEntities(
            biz_line=self.biz_line,
            domain=self.domain,
            data_layer=self.data_layer,
            table=TableIdentifier.parse(self.table_name) if self.table_name else None,
            field_name=self.field_name,
            operation=self.operation,
            topic_keywords=self.topic_keywords,
            lineage_direction=self.lineage_direction,
        )


class LLMQuestionAnalyzer:
    """Production-style structured analyzer backed by an OpenAI-compatible model endpoint."""

    system_prompt = """你是企业数据治理场景的智能数据探查 Agent Planner。
你的任务是把用户自然语言问题解析为严格 JSON，不要回答业务问题。

只允许输出 JSON object，不要 Markdown，不要解释。

枚举约束：
- intent: metadata_search | lineage_search | impact_analysis | unknown
- domain: 营销域 | 风控域 | 交易域 | 用户域 | 财务域 | null
- data_layer: ODS | DWD | DWS | ADS | DIM | null
- operation: modify_field | add_field | delete_field | rename_field | unknown_change | null
- lineage_direction: upstream | downstream | both | null

判断准则：
- 查表、表说明、业务含义、某主题下有哪些表 => metadata_search
- 查上游、下游、依赖、血缘，且没有变更动作 => lineage_search
- 出现表修改、表变更、字段修改、影响范围 => impact_analysis；当前 planner 先统一生成表级影响计划
- 上游和下游同时出现 => lineage_direction=both
- 只出现上游/来源 => lineage_direction=upstream
- 只出现下游/影响哪些表/产生影响 => lineage_direction=downstream
- 表名必须尽量抽取原始表达，支持三段式、两段式、一段式，例如 catalog.dwd.orderInfo、dwd.orderInfo、userInfo
- field_name 为后续字段级扩展预留；当前表级探查阶段，用户只说“字段修改”时 field_name=null
- topic_keywords 只放业务主题词，例如 支付、订单、用户、营销、风控，不要放主题域或数仓分层
"""

    def __init__(self, client: OpenAICompatibleChatClient | None = None) -> None:
        self.client = client or OpenAICompatibleChatClient()

    def analyze(self, question: str) -> LLMAnalysis:
        user_prompt = f"""请解析下面的问题，输出符合约束的 JSON。

输出字段：
{{
  "intent": "...",
  "confidence": 0.0,
  "biz_line": null,
  "domain": null,
  "data_layer": null,
  "table_name": null,
  "field_name": null,
  "operation": null,
  "topic_keywords": [],
  "lineage_direction": null
}}

用户问题：{question}
"""
        # 用户问题为什么放最后 让模型建立任务和Json输出约束 在把待解析文本作为输入交给模型 有利于稳定输出结构化结果
        try:
            payload = self.client.chat_json(self.system_prompt, user_prompt)
            return LLMAnalysis.model_validate(payload)
        except (LLMClientError, ValidationError, ValueError) as exc:
            raise LLMClientError(f"LLM structured analysis failed: {exc}") from exc
