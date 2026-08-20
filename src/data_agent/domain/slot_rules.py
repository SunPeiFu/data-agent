from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml
from pydantic import BaseModel, Field

from data_agent.config.paths import CONFIG_DIR
from data_agent.domain.models import IntentType, SlotIssueType


class IntentSlotRule(BaseModel):
    pre_required_any: list[str] = Field(default_factory=list)
    post_required_any: list[str] = Field(default_factory=list)


class SlotRuleConfig(BaseModel):
    intents: dict[IntentType, IntentSlotRule] = Field(default_factory=dict)
    blocking_issue_types: set[SlotIssueType] = Field(
        default_factory=lambda: {
            SlotIssueType.MISSING,
            SlotIssueType.AMBIGUOUS,
            SlotIssueType.INVALID,
            SlotIssueType.CONFLICT,
            SlotIssueType.FORBIDDEN,
            SlotIssueType.LOW_CONFIDENCE,
        }
    )

    def rule_for(self, intent: IntentType) -> IntentSlotRule:
        return self.intents.get(intent, IntentSlotRule())

    def is_blocking(self, issue_type: SlotIssueType) -> bool:
        return issue_type in self.blocking_issue_types


@lru_cache(maxsize=1)
def load_slot_rule_config() -> SlotRuleConfig:
    config_path = CONFIG_DIR / "slot_rules.yml"
    if not config_path.exists():
        return SlotRuleConfig()
    with config_path.open("r", encoding="utf-8") as file:
        payload: dict[str, Any] = yaml.safe_load(file) or {}
    return SlotRuleConfig.model_validate(payload)
