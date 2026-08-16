from __future__ import annotations

import re

from data_agent.llm_analyzer import LLMQuestionAnalyzer
from data_agent.llm_client import LLMClientError
from data_agent.models import (
    DataLayer,
    DomainType,
    ExtractedEntities,
    LineageDirection,
    OperationType,
    TableIdentifier,
)


BUSINESS_LINES = ("安逸花",)
TOPIC_KEYWORDS = ("支付", "订单", "用户", "营销", "风控", "交易", "财务")


def extract_entities(question: str) -> ExtractedEntities:
    """Production entity extraction.

    The primary path uses an OpenAI-compatible LLM with strict Pydantic validation.
    The deterministic extractor remains as a resilience fallback for offline tests,
    local demos without a model, and LLM timeout/error cases.
    """
    try:
        return LLMQuestionAnalyzer().analyze(question).to_entities()
    except LLMClientError:
        return extract_entities_by_rules(question)


def extract_entities_by_rules(question: str) -> ExtractedEntities:
    return ExtractedEntities(
        biz_line=_extract_biz_line(question),
        domain=_extract_domain(question),
        data_layer=_extract_layer(question),
        table=_extract_table(question),
        field_name=_extract_field(question),
        operation=_extract_operation(question),
        topic_keywords=_extract_topic_keywords(question),
        lineage_direction=_extract_lineage_direction(question),
    )


def _extract_biz_line(question: str) -> str | None:
    return next((line for line in BUSINESS_LINES if line in question), None)


def _extract_domain(question: str) -> DomainType | None:
    for domain in DomainType:
        if domain.value in question:
            return domain
    return None


def _extract_layer(question: str) -> DataLayer | None:
    upper_question = question.upper()
    for layer in DataLayer:
        if layer.value in upper_question:
            return layer
    return None


def _extract_table(question: str) -> TableIdentifier | None:
    three_or_two_part = re.search(r"\b([A-Za-z][\w]*\.[A-Za-z][\w]*(?:\.[A-Za-z][\w]*)?)\b", question)
    if three_or_two_part:
        return TableIdentifier.parse(three_or_two_part.group(1))

    one_part_patterns = [
        r"(?:层的|表名为|表是)([A-Za-z][A-Za-z0-9_]*)(?:表|中的|修改|字段|，|,|对|$)",
        r"([A-Za-z][A-Za-z0-9_]*)(?:表修改字段|表字段|表的|表中|表有哪些)",
        r"\b([A-Za-z][\w]*(?:Info|Order|User|Pay|Payment|Profile|Detail|Summary))\b",
    ]
    for pattern in one_part_patterns:
        match = re.search(pattern, question)
        if match:
            candidate = match.group(1)
            if candidate.lower() not in {"dwd", "ods", "dws", "ads", "dim"}:
                return TableIdentifier.parse(candidate)
    return None


def _extract_field(question: str) -> str | None:
    match = re.search(r"(?:字段|field)[：:\s]*([A-Za-z_][\w]*)", question)
    if match:
        return match.group(1)
    match = re.search(r"\b([A-Za-z_][\w]*)\s*字段(?:修改|变更|新增|删除|重命名)", question)
    if match:
        return match.group(1)
    return None


def _extract_operation(question: str) -> OperationType | None:
    if "新增字段" in question:
        return OperationType.ADD_FIELD
    if "删除字段" in question:
        return OperationType.DELETE_FIELD
    if "重命名字段" in question or "字段重命名" in question:
        return OperationType.RENAME_FIELD
    if "修改字段" in question or "字段修改" in question or re.search(r"字段\s*[A-Za-z_][\w]*\s*修改", question):
        return OperationType.MODIFY_FIELD
    if "变更" in question or "影响" in question:
        return OperationType.UNKNOWN_CHANGE
    return None


def _extract_topic_keywords(question: str) -> list[str]:
    return [keyword for keyword in TOPIC_KEYWORDS if keyword in question]


def _extract_lineage_direction(question: str) -> LineageDirection | None:
    has_upstream = "上游" in question or "来源" in question
    has_downstream = "下游" in question or "影响" in question
    if has_upstream and has_downstream:
        return LineageDirection.BOTH
    if has_upstream:
        return LineageDirection.UPSTREAM
    if has_downstream:
        return LineageDirection.DOWNSTREAM
    return None
