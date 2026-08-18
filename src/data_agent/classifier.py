from __future__ import annotations

from data_agent.llm_analyzer import LLMQuestionAnalyzer
from data_agent.llm_client import LLMClientError
from data_agent.models import IntentType


class RuleBasedIntentClassifier:
    """Offline fallback classifier for tests and local development without an LLM."""

    metadata_keywords = ("有哪些表", "表有哪些", "表说明", "元数据", "关于", "业务含义", "搜索", "相关表")
    lineage_keywords = ("上游", "下游", "依赖", "血缘", "来源", "影响哪些表")
    impact_keywords = ("修改字段", "字段修改", "新增字段", "删除字段", "重命名字段", "变更", "影响")

    def classify(self, question: str) -> tuple[IntentType, float]:
        normalized = question.strip()
        has_impact = any(keyword in normalized for keyword in self.impact_keywords)
        has_lineage = any(keyword in normalized for keyword in self.lineage_keywords)
        has_metadata = any(keyword in normalized for keyword in self.metadata_keywords)

        if has_impact:
            return IntentType.IMPACT_ANALYSIS, 0.92 if has_lineage else 0.86
        if has_lineage:
            return IntentType.LINEAGE_SEARCH, 0.88
        if has_metadata:
            return IntentType.METADATA_SEARCH, 0.86
        return IntentType.UNKNOWN, 0.35


class LLMIntentClassifier:
    """Production classifier backed by an OpenAI-compatible structured LLM call."""

    def __init__(self, analyzer: LLMQuestionAnalyzer | None = None) -> None:
        self.analyzer = analyzer or LLMQuestionAnalyzer()

    def classify(self, question: str) -> tuple[IntentType, float]:
        analysis = self.analyzer.analyze(question)
        return analysis.intent, analysis.confidence


class HybridIntentClassifier:
    """Prefer production LLM classification, then fall back to deterministic rules."""

    def __init__(self, llm_classifier: LLMIntentClassifier | None = None) -> None:
        self.llm_classifier = llm_classifier or LLMIntentClassifier()
        self.rule_classifier = RuleBasedIntentClassifier()

    def classify(self, question: str) -> tuple[IntentType, float]:
        try:
            return self.llm_classifier.classify(question)
        except LLMClientError:
            return self.rule_classifier.classify(question)
