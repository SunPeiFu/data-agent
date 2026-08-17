from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from data_agent.classifier import RuleBasedIntentClassifier
from data_agent.extractor import extract_entities_by_rules
from data_agent.llm_analyzer import LLMAnalysis, LLMQuestionAnalyzer
from data_agent.llm_client import LLMClientError
from data_agent.models import ExtractedEntities, IntentType, TableIdentifier


class EvidenceStrength(str, Enum):
    """候选证据强度：用于判断冲突时谁更应该被信任。"""

    HARD = "hard"
    STRONG = "strong"
    WEAK = "weak"


class CandidateSource(str, Enum):
    """候选值来源：生产中每个字段都要知道是谁给出的。"""

    RULE = "rule"
    LLM = "llm"
    METADATA = "metadata"


class ConflictResolution(str, Enum):
    """字段冲突处理结果：记录最终值是怎么被选出来的。"""

    MERGED = "merged"
    SELECTED_RULE = "selected_rule"
    SELECTED_LLM = "selected_llm"
    NEEDS_CLARIFICATION = "needs_clarification"
    NEEDS_METADATA_VALIDATION = "needs_metadata_validation"


class RuleEvidence(BaseModel):
    """规则预分析产生的证据。

    它不是最终字段值，而是“为什么规则认为某字段应该是某个值”的证据记录。
    企业级系统会保留 evidence 方便排查误判、做灰度和调权重。
    """

    field_name: str
    value: Any
    source: CandidateSource = CandidateSource.RULE
    confidence: float = Field(ge=0.0, le=1.0)
    strength: EvidenceStrength
    reason: str
    matched_text: str | None = None


class FieldCandidate(BaseModel):
    """字段候选值。

    同一个字段可能来自规则、LLM、元数据服务等多个来源。PolicyResolver
    不直接二选一，而是先把这些来源统一包装成候选，再按字段策略决策。
    """

    field_name: str
    value: Any
    source: CandidateSource
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    strength: EvidenceStrength = EvidenceStrength.WEAK
    requires_metadata_validation: bool = False


class ResolvedField(BaseModel):
    """单个字段的最终决策结果。

    保存最终 value、采用的 source、置信度、冲突处理方式以及所有候选。
    这比直接返回一个值更适合生产排障和面试讲解。
    """

    field_name: str
    value: Any = None
    selected_source: CandidateSource | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    resolution: ConflictResolution
    candidates: list[FieldCandidate] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class EntityResolution(BaseModel):
    """所有字段的决策结果集合。"""

    fields: dict[str, ResolvedField] = Field(default_factory=dict)

    def get_value(self, field_name: str) -> Any:
        """读取某个字段的最终值，用于组装 ExtractedEntities。"""
        resolved = self.fields.get(field_name)
        return resolved.value if resolved else None

    @property
    def notes(self) -> list[str]:
        """汇总字段级决策备注，最终透传到 PlanningResult.notes。"""
        return [note for field in self.fields.values() for note in field.notes]


class EntityMergePolicy(BaseModel):
    """字段级合并策略。

    真实生产中不同字段的可信来源不同：
    枚举/方向/操作更适合 hard rule，表名/字段名更适合作为候选交给元数据校验。
    """

    hard_rule_fields: set[str] = Field(
        default_factory=lambda: {
            "intent",
            "operation",
            "lineage_direction",
            "domain",
            "data_layer",
        }
    )
    metadata_validated_fields: set[str] = Field(default_factory=lambda: {"table", "field_name"})
    llm_semantic_fields: set[str] = Field(default_factory=lambda: {"table", "field_name", "topic_keywords"})
    conflict_margin: float = 0.12


class RulePreAnalysis(BaseModel):
    """规则预分析的完整输出。"""

    intent: IntentType
    confidence: float = Field(ge=0.0, le=1.0)
    entities: ExtractedEntities
    evidences: list[RuleEvidence] = Field(default_factory=list)


class HybridRouteResult(BaseModel):
    """Hybrid Router 的最终输出。

    Planner 后续只需要消费 intent、confidence、entities；调试和学习时可以看
    entity_resolution、llm_analysis、rule_analysis 理解决策过程。
    """

    intent: IntentType
    confidence: float = Field(ge=0.0, le=1.0)
    entities: ExtractedEntities
    entity_resolution: EntityResolution = Field(default_factory=EntityResolution)
    llm_analysis: LLMAnalysis | None = None
    rule_analysis: RulePreAnalysis
    notes: list[str] = Field(default_factory=list)


class RulePreAnalyzer:
    """Extract high-certainty routing signals before calling the LLM."""

    def analyze(self, question: str) -> RulePreAnalysis:
        """执行规则预分析。

        职责：
        1. 用轻量规则识别高确定性 intent。
        2. 用规则抽取能稳定识别的实体。
        3. 生成 RuleEvidence，给 PolicyResolver 后续决策提供证据。
        """
        intent, confidence = RuleBasedIntentClassifier().classify(question)
        entities = extract_entities_by_rules(question)
        return RulePreAnalysis(
            intent=intent,
            confidence=confidence,
            entities=entities,
            evidences=_build_rule_evidences(intent=intent, confidence=confidence, entities=entities),
        )


class PolicyResolver:
    """Merge deterministic rule signals, LLM candidates, and validation policy."""

    def __init__(self, policy: EntityMergePolicy | None = None) -> None:
        """初始化字段级合并策略。

        policy 可以后续从配置中心加载，比如不同业务线设置不同 hard rule 字段、
        冲突阈值、是否强制人工澄清等。
        """
        self.policy = policy or EntityMergePolicy()

    def resolve(self, rule: RulePreAnalysis, llm: LLMAnalysis | None) -> HybridRouteResult:
        """合并规则预分析和 LLM 结构化解析。

        这是 Hybrid Router 的核心入口：
        - LLM 成功：规则候选 + LLM 候选一起进入字段级决策。
        - LLM 失败：规则候选仍能生成可执行计划。
        - 返回结果同时包含最终 entities 和完整 entity_resolution。
        """
        llm_entities = llm.to_entities() if llm else None
        resolution = self._resolve_entities(rule=rule, llm_entities=llm_entities, llm=llm)
        intent_field = resolution.fields["intent"]
        intent = intent_field.value or IntentType.UNKNOWN

        notes = resolution.notes
        if llm is None:
            notes = ["LLM 不可用或结构化解析失败，已使用规则预分析兜底。", *notes]

        return HybridRouteResult(
            intent=intent,
            confidence=intent_field.confidence,
            entities=self._entities_from_resolution(resolution),
            entity_resolution=resolution,
            llm_analysis=llm,
            rule_analysis=rule,
            notes=notes,
        )

    # _resolve_scalar 是规则结果和 LLM 结果的“单字段仲裁器”：
    # 先看有没有候选，再看候选是否一致，不一致时优先 hard rule，然后看置信度差距，差距太小就要求澄清，否则选择置信度最高的候选。
    def _resolve_entities(
        self,
        rule: RulePreAnalysis,
        llm_entities: ExtractedEntities | None,
        llm: LLMAnalysis | None = None,
    ) -> EntityResolution:
        """逐字段生成候选并完成决策。

        企业级实体合并的关键点是“按字段决策”，而不是全局 rule 优先或 LLM 优先。
        例如 operation/direction 是 hard rule，table/field_name 是候选并需要元数据校验。
        """
        fields = {
            "intent": self._resolve_scalar(
                "intent",
                [
                    FieldCandidate(
                        field_name="intent",
                        value=rule.intent,
                        source=CandidateSource.RULE,
                        confidence=rule.confidence,
                        strength=_intent_strength(rule.intent),
                        reason="rule_intent_classifier",
                    ),
                    *(
                        [
                            FieldCandidate(
                                field_name="intent",
                                value=llm.intent,
                                source=CandidateSource.LLM,
                                confidence=llm.confidence,
                                strength=EvidenceStrength.STRONG,
                                reason="llm_structured_output",
                            )
                        ]
                        if llm
                        else []
                    ),
                ],
            ),
            "biz_line": self._resolve_scalar(
                "biz_line",
                self._field_candidates("biz_line", rule.entities.biz_line, _llm_value(llm_entities, "biz_line")),
            ),
            "domain": self._resolve_scalar(
                "domain",
                self._field_candidates("domain", rule.entities.domain, _llm_value(llm_entities, "domain")),
            ),
            "data_layer": self._resolve_scalar(
                "data_layer",
                self._field_candidates("data_layer", rule.entities.data_layer, _llm_value(llm_entities, "data_layer")),
            ),
            "table": self._resolve_scalar(
                "table",
                self._field_candidates(
                    "table",
                    rule.entities.table,
                    _llm_value(llm_entities, "table"),
                    requires_metadata_validation=True,
                ),
            ),
            "field_name": self._resolve_scalar(
                "field_name",
                self._field_candidates(
                    "field_name",
                    rule.entities.field_name,
                    _llm_value(llm_entities, "field_name"),
                    requires_metadata_validation=True,
                ),
            ),
            "operation": self._resolve_scalar(
                "operation",
                self._field_candidates("operation", rule.entities.operation, _llm_value(llm_entities, "operation")),
            ),
            "lineage_direction": self._resolve_scalar(
                "lineage_direction",
                self._field_candidates(
                    "lineage_direction",
                    rule.entities.lineage_direction,
                    _llm_value(llm_entities, "lineage_direction"),
                ),
            ),
            "topic_keywords": self._resolve_keywords(rule.entities, llm_entities),
        }
        return EntityResolution(fields=fields)

    def _field_candidates(
        self,
        field_name: str,
        rule_value: Any,
        llm_value: Any,
        requires_metadata_validation: bool = False,
    ) -> list[FieldCandidate]:
        """把规则值和 LLM 值统一包装成字段候选。

        每个候选都携带 source、confidence、strength、reason，以及是否需要元数据校验。
        后续 _resolve_scalar 只处理标准候选，不关心候选来自哪里。
        """
        candidates: list[FieldCandidate] = []
        if rule_value is not None:
            candidates.append(
                FieldCandidate(
                    field_name=field_name,
                    value=rule_value,
                    source=CandidateSource.RULE,
                    confidence=_rule_confidence(field_name),
                    strength=EvidenceStrength.HARD if field_name in self.policy.hard_rule_fields else EvidenceStrength.STRONG,
                    reason=f"rule_{field_name}_detected",
                    requires_metadata_validation=requires_metadata_validation,
                )
            )
        if llm_value is not None:
            candidates.append(
                FieldCandidate(
                    field_name=field_name,
                    value=llm_value,
                    source=CandidateSource.LLM,
                    confidence=_llm_field_confidence(field_name),
                    strength=EvidenceStrength.STRONG if field_name in self.policy.llm_semantic_fields else EvidenceStrength.WEAK,
                    reason="llm_structured_output",
                    requires_metadata_validation=requires_metadata_validation,
                )
            )
        return candidates

    def _resolve_scalar(self, field_name: str, candidates: list[FieldCandidate]) -> ResolvedField:
        """解析单值字段的最终值。

        决策顺序：
        1. 没候选：返回空字段。
        2. 候选值一致：选择置信度最高的候选。
        3. 有 hard rule 冲突：hard rule 覆盖。
        4. 候选置信度接近：标记 needs_clarification。
        5. 否则选择最高置信度候选。
        """
        if not candidates:
            return ResolvedField(field_name=field_name, resolution=ConflictResolution.MERGED)

        # 归一化后是一个 比如dwd DWD 归一化后都是dwd
        normalized_values = {_normalize_candidate_value(candidate.value) for candidate in candidates}
        if len(normalized_values) == 1:
            # 选取置信度最高的一个
            selected = max(candidates, key=lambda item: item.confidence)
            return ResolvedField(
                field_name=field_name,
                value=selected.value,
                selected_source=selected.source,
                confidence=selected.confidence,
                resolution=(
                    ConflictResolution.NEEDS_METADATA_VALIDATION
                    if selected.requires_metadata_validation
                    else ConflictResolution.MERGED
                ),
                candidates=candidates,
                notes=_metadata_validation_notes(field_name, selected),
            )

        # 如果候选值不一致 就进入冲突处理
        hard_rule = next(
            (
                candidate
                for candidate in candidates
                if candidate.source == CandidateSource.RULE
                and candidate.strength == EvidenceStrength.HARD
                and field_name in self.policy.hard_rule_fields
            ),
            None,
        )

        # 命中强规则 -> 强规则的覆盖llm
        if hard_rule is not None:
            return ResolvedField(
                field_name=field_name,
                value=hard_rule.value,
                selected_source=CandidateSource.RULE,
                confidence=hard_rule.confidence,
                resolution=ConflictResolution.SELECTED_RULE,
                candidates=candidates,
                notes=[f"{field_name} 存在冲突，已采用 hard rule 候选。"],
            )

        # 未命中强规则 按照置信度倒序排列取前
        selected, runner_up = sorted(candidates, key=lambda item: item.confidence, reverse=True)[:2]
        # 倒序前两个的二者之差太小 应该澄清或者由元数据校验 并返回重置的相对较低的置信度
        if selected.confidence - runner_up.confidence < self.policy.conflict_margin:
            return ResolvedField(
                field_name=field_name,
                value=selected.value,
                selected_source=selected.source,
                confidence=min(selected.confidence, 0.62),
                resolution=ConflictResolution.NEEDS_CLARIFICATION,
                candidates=candidates,
                notes=[f"{field_name} 候选冲突且置信度接近，后续应澄清或由元数据校验。"],
            )

        # 直接选高置信度的
        return ResolvedField(
            field_name=field_name,
            value=selected.value,
            selected_source=selected.source,
            confidence=selected.confidence,
            resolution=(
                ConflictResolution.SELECTED_LLM
                if selected.source == CandidateSource.LLM
                else ConflictResolution.SELECTED_RULE
            ),
            candidates=candidates,
            notes=_metadata_validation_notes(field_name, selected),
        )

    def _resolve_keywords(self, rule: ExtractedEntities, llm: ExtractedEntities | None) -> ResolvedField:
        """合并 topic_keywords。

        关键词不是强约束字段，通常用于检索召回和排序，因此采用合并去重策略，
        保留规则与 LLM 各自识别到的业务主题词。
        """
        merged = _merge_keywords(rule.topic_keywords, llm.topic_keywords if llm else [])
        candidates = [
            *[
                FieldCandidate(
                    field_name="topic_keywords",
                    value=keyword,
                    source=CandidateSource.RULE,
                    confidence=0.75,
                    reason="rule_topic_keyword_detected",
                )
                for keyword in rule.topic_keywords
            ],
            *[
                FieldCandidate(
                    field_name="topic_keywords",
                    value=keyword,
                    source=CandidateSource.LLM,
                    confidence=0.82,
                    reason="llm_topic_keyword_detected",
                )
                for keyword in (llm.topic_keywords if llm else [])
            ],
        ]
        return ResolvedField(
            field_name="topic_keywords",
            value=merged,
            selected_source=None,
            confidence=0.82 if merged else 0.0,
            resolution=ConflictResolution.MERGED,
            candidates=candidates,
        )

    def _entities_from_resolution(self, resolution: EntityResolution) -> ExtractedEntities:
        """把字段级决策结果还原成 Planner 使用的 ExtractedEntities。

        EntityResolution 用于解释和审计，ExtractedEntities 用于后续任务拆解。
        """
        return ExtractedEntities(
            biz_line=resolution.get_value("biz_line"),
            domain=resolution.get_value("domain"),
            data_layer=resolution.get_value("data_layer"),
            table=resolution.get_value("table"),
            field_name=resolution.get_value("field_name"),
            operation=resolution.get_value("operation"),
            topic_keywords=resolution.get_value("topic_keywords") or [],
            lineage_direction=resolution.get_value("lineage_direction"),
        )


class HybridQuestionRouter:
    """Production router: rule pre-analysis + LLM analysis + policy resolution."""

    def __init__(
        self,
        rule_analyzer: RulePreAnalyzer | None = None,
        llm_analyzer: LLMQuestionAnalyzer | None = None,
        resolver: PolicyResolver | None = None,
    ) -> None:
        """组装 Hybrid Router 的三个组件。

        依赖可注入，方便单测里替换 LLM 或 resolver，也方便生产里按业务线定制策略。
        """
        self.rule_analyzer = rule_analyzer or RulePreAnalyzer()
        self.llm_analyzer = llm_analyzer or LLMQuestionAnalyzer()
        self.resolver = resolver or PolicyResolver()

    def route(self, question: str) -> HybridRouteResult:
        """执行完整路由流程。

        流程：
        1. 规则预分析先跑，保证即使 LLM 不可用也有结果。
        2. 尝试调用 LLM 做结构化理解。
        3. 把 rule 和 llm 交给 PolicyResolver 合并。
        """
        rule = self.rule_analyzer.analyze(question)
        llm: LLMAnalysis | None = None
        try:
            llm = self.llm_analyzer.analyze(question)
        except LLMClientError:
            llm = None
        return self.resolver.resolve(rule=rule, llm=llm)


def _merge_keywords(rule_keywords: list[str], llm_keywords: list[str]) -> list[str]:
    """合并并去重业务主题词，保持原始发现顺序。"""
    merged: list[str] = []
    for keyword in [*rule_keywords, *llm_keywords]:
        if keyword and keyword not in merged:
            merged.append(keyword)
    return merged


def _build_rule_evidences(intent: IntentType, confidence: float, entities: ExtractedEntities) -> list[RuleEvidence]:
    """基于规则预分析结果生成证据列表。

    这些证据目前主要用于解释和后续扩展；未来可以参与更细粒度的权重打分。
    """
    evidences = [
        RuleEvidence(
            field_name="intent",
            value=intent,
            confidence=confidence,
            strength=_intent_strength(intent),
            reason="rule_intent_classifier",
        )
    ]
    for field_name in ["domain", "data_layer", "operation", "lineage_direction"]:
        value = getattr(entities, field_name)
        if value is not None:
            evidences.append(
                RuleEvidence(
                    field_name=field_name,
                    value=value,
                    confidence=_rule_confidence(field_name),
                    strength=EvidenceStrength.HARD,
                    reason=f"rule_{field_name}_detected",
                )
            )
    return evidences


def _intent_strength(intent: IntentType) -> EvidenceStrength:
    """给规则 intent 分配证据强度。"""
    if intent in {IntentType.IMPACT_ANALYSIS, IntentType.LINEAGE_SEARCH}:
        return EvidenceStrength.HARD
    if intent == IntentType.METADATA_SEARCH:
        return EvidenceStrength.STRONG
    return EvidenceStrength.WEAK


def _rule_confidence(field_name: str) -> float:
    """规则候选的默认字段置信度。

    枚举、操作和方向较稳定，置信度更高；表名和字段名只是候选，置信度较低。
    """
    return {
        "biz_line": 0.9,
        "domain": 0.98,
        "data_layer": 0.99,
        "operation": 0.98,
        "lineage_direction": 0.98,
        "table": 0.76,
        "field_name": 0.72,
    }.get(field_name, 0.7)


def _llm_field_confidence(field_name: str) -> float:
    """LLM 候选的默认字段置信度。

    LLM 更擅长抽取自然语言里的表名、字段名和语义词，但枚举类字段仍需策略约束。
    """
    return {
        "biz_line": 0.82,
        "domain": 0.88,
        "data_layer": 0.88,
        "operation": 0.86,
        "lineage_direction": 0.86,
        "table": 0.86,
        "field_name": 0.84,
    }.get(field_name, 0.8)


def _normalize_candidate_value(value: Any) -> str:
    """归一化候选值，用于判断不同来源的候选是否表达同一个值。"""
    if isinstance(value, TableIdentifier):
        return value.raw.lower()
    return str(getattr(value, "value", value)).lower()


def _metadata_validation_notes(field_name: str, selected: FieldCandidate) -> list[str]:
    """为需要元数据确认的字段生成备注。

    table 和 field_name 即使被选中，也只是候选值，后续必须由 TiDB 元数据服务确认。
    """
    if selected.requires_metadata_validation:
        return [f"{field_name} 已作为候选值选中，后续必须通过元数据服务校验。"]
    return []


def _llm_value(entities: ExtractedEntities | None, field_name: str) -> Any:
    """安全读取 LLM entities 中的字段值。"""
    return getattr(entities, field_name) if entities else None
