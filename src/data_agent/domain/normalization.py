from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml
from pydantic import BaseModel, Field

from data_agent.config.paths import CONFIG_DIR
from data_agent.domain.models import NormalizedTerm, NormalizedTermType


class SynonymRule(BaseModel):
    canonical: str
    term_type: NormalizedTermType
    aliases: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)


class TableTermRule(BaseModel):
    canonical: str
    display_name: str
    aliases: list[str] = Field(default_factory=list)
    candidate_tables: list[str] = Field(default_factory=list)
    domain: str | None = None
    preferred_layer: str | None = None
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)


class NormalizationConfig(BaseModel):
    stopwords: list[str] = Field(default_factory=list)
    synonyms: list[SynonymRule] = Field(default_factory=list)
    table_terms: list[TableTermRule] = Field(default_factory=list)

    def map_term(self, text: str) -> NormalizedTerm | None:
        for rule in self.synonyms:
            values = {rule.canonical, *rule.aliases}
            if text in values:
                return NormalizedTerm(
                    text=text,
                    canonical=rule.canonical,
                    term_type=rule.term_type,
                    source="normalization_config",
                    confidence=rule.confidence,
                )
        table_rule = self.get_table_term(text)
        if table_rule:
            return NormalizedTerm(
                text=text,
                canonical=table_rule.canonical,
                term_type=NormalizedTermType.TABLE_TERM,
                source="normalization_config",
                confidence=table_rule.confidence,
            )
        return None

    def get_table_term(self, text: str) -> TableTermRule | None:
        for rule in self.table_terms:
            values = {rule.canonical, rule.display_name, *rule.aliases}
            if text in values:
                return rule
        return None


@lru_cache(maxsize=1)
def load_normalization_config() -> NormalizationConfig:
    config_path = CONFIG_DIR / "normalization.yml"
    if not config_path.exists():
        return NormalizationConfig()
    with config_path.open("r", encoding="utf-8") as file:
        payload: dict[str, Any] = yaml.safe_load(file) or {}
    return NormalizationConfig.model_validate(payload)
