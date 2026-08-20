"""Single source of truth for executable tools and their production policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from data_agent.domain.models import AccessContext, ExtractedEntities, IntentType
from data_agent.infrastructure.security.authorization import AuthorizationProvider, YamlAuthorizationProvider


INTENT_SCOPES = {
    IntentType.METADATA_SEARCH: "metadata:read",
    IntentType.LINEAGE_SEARCH: "lineage:read",
    IntentType.IMPACT_ANALYSIS: "impact:analyze",
}


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 1
    backoff_seconds: float = 0.0


@dataclass(frozen=True)
class RegisteredTool:
    service: str
    action: str
    tool: BaseTool
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    intents: frozenset[IntentType]
    required_scope: str
    timeout_seconds: float = 10.0
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    idempotent: bool = True
    side_effect_level: str = "read_only"
    allow_failed_dependencies: bool = False
    version: str = "v1"

    @property
    def key(self) -> tuple[str, str]:
        return self.service, self.action


class ToolRegistry:
    """Register, discover, validate, and authorize tools through one stable API."""

    def __init__(
        self,
        authorization_provider: AuthorizationProvider | None = None,
        *,
        version: str = "local-tools-v1",
        policy_version: str = "yaml-rbac-v1",
    ) -> None:
        self._tools: dict[tuple[str, str], RegisteredTool] = {}
        self.authorization_provider = authorization_provider or YamlAuthorizationProvider()
        self.version = version
        self.policy_version = policy_version

    def register(self, definition: RegisteredTool) -> None:
        if definition.key in self._tools:
            raise ValueError(f"工具已注册: {definition.service}.{definition.action}")
        self._tools[definition.key] = definition

    def register_many(self, definitions: Iterable[RegisteredTool]) -> None:
        for definition in definitions:
            self.register(definition)

    def get(self, service: str, action: str) -> RegisteredTool:
        try:
            return self._tools[(service, action)]
        except KeyError as exc:
            raise KeyError(f"工具未注册: {service}.{action}") from exc

    def all(self) -> list[RegisteredTool]:
        return list(self._tools.values())

    def available_for(
        self,
        intent: IntentType,
        access_context: AccessContext,
        entities: ExtractedEntities,
    ) -> list[RegisteredTool]:
        if not self.is_intent_authorized(intent, access_context, entities):
            return []
        return [
            definition
            for definition in self._tools.values()
            if intent in definition.intents and self.is_authorized(definition, access_context, entities)
        ]

    def is_authorized(
        self,
        definition: RegisteredTool,
        access_context: AccessContext,
        entities: ExtractedEntities,
    ) -> bool:
        resource = {
            "full_table_name": entities.table.raw if entities.table else None,
            "domain": entities.domain.value if entities.domain else None,
            "biz_line": entities.biz_line,
        }
        decision = self.authorization_provider.authorize(access_context, definition.required_scope, resource)
        return decision.allowed

    def is_intent_authorized(
        self,
        intent: IntentType,
        access_context: AccessContext,
        entities: ExtractedEntities,
    ) -> bool:
        required_scope = INTENT_SCOPES.get(intent)
        if required_scope is None:
            return False
        resource = {
            "full_table_name": entities.table.raw if entities.table else None,
            "domain": entities.domain.value if entities.domain else None,
            "biz_line": entities.biz_line,
        }
        return self.authorization_provider.authorize(access_context, required_scope, resource).allowed
