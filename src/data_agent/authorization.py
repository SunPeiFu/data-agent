from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import yaml
from pydantic import BaseModel, Field

from data_agent.models import AccessContext, AuthorizationDecision


class AuthorizationProvider(Protocol):
    """Boundary used to replace local policies with IAM or Ranger without changing the planner."""

    def authorize(
        self,
        context: AccessContext,
        action: str,
        resource: dict[str, Any],
    ) -> AuthorizationDecision:
        """Evaluate whether a subject may perform an action on a data resource."""


class RolePolicy(BaseModel):
    """Minimal RBAC policy with table-level business scopes."""

    allowed_actions: list[str] = Field(default_factory=list)
    allowed_domains: list[str] = Field(default_factory=list)
    allowed_biz_lines: list[str] = Field(default_factory=list)


class AuthorizationConfig(BaseModel):
    """Validated representation of the local authorization policy file."""

    roles: dict[str, RolePolicy] = Field(default_factory=dict)


class YamlAuthorizationProvider:
    """Local RBAC provider for demos; production can implement the same protocol via HTTP."""

    def __init__(self, config: AuthorizationConfig | None = None) -> None:
        self.config = config or load_authorization_config()

    def authorize(
        self,
        context: AccessContext,
        action: str,
        resource: dict[str, Any],
    ) -> AuthorizationDecision:
        """Combine role grants and direct grants, then evaluate action and data scope."""
        actions, domains, biz_lines, policy_ids = self._effective_grants(context)
        resource_name = _resource_name(resource)

        if not _matches(actions, action):
            return _decision(False, action, resource_name, policy_ids, "AUTH_ACTION_DENIED")

        domain = resource.get("domain")
        if not _scope_allowed(domains, domain):
            reason = "AUTH_SCOPE_REQUIRED" if domain is None else "AUTH_DOMAIN_DENIED"
            return _decision(False, action, resource_name, policy_ids, reason)

        biz_line = resource.get("biz_line")
        if biz_lines and not _scope_allowed(biz_lines, biz_line):
            reason = "AUTH_SCOPE_REQUIRED" if biz_line is None else "AUTH_BIZ_LINE_DENIED"
            return _decision(False, action, resource_name, policy_ids, reason)

        return _decision(True, action, resource_name, policy_ids, "AUTH_ALLOWED")

    def _effective_grants(self, context: AccessContext) -> tuple[set[str], set[str], set[str], list[str]]:
        """Union grants from all subject roles, matching common additive RBAC semantics."""
        actions = set(context.allowed_actions)
        domains = set(context.allowed_domains)
        biz_lines: set[str] = set()
        policy_ids: list[str] = []
        for role_name in context.roles:
            policy = self.config.roles.get(role_name)
            if not policy:
                continue
            actions.update(policy.allowed_actions)
            domains.update(policy.allowed_domains)
            biz_lines.update(policy.allowed_biz_lines)
            policy_ids.append(f"role:{role_name}")
        return actions, domains, biz_lines, policy_ids


@lru_cache(maxsize=1)
def load_authorization_config() -> AuthorizationConfig:
    """Load and validate demo policies once per process."""
    config_path = Path(__file__).resolve().parents[2] / "config" / "access_policies.yml"
    if not config_path.exists():
        return AuthorizationConfig()
    with config_path.open("r", encoding="utf-8") as file:
        payload: dict[str, Any] = yaml.safe_load(file) or {}
    return AuthorizationConfig.model_validate(payload)


def _scope_allowed(grants: set[str], value: str | None) -> bool:
    """Require an explicit scope unless the subject has a wildcard grant."""
    if "*" in grants:
        return True
    return value is not None and value in grants


def _matches(grants: set[str], value: str) -> bool:
    return "*" in grants or value in grants


def _resource_name(resource: dict[str, Any]) -> str | None:
    value = resource.get("full_table_name")
    return str(value) if value else None


def _decision(
    allowed: bool,
    action: str,
    resource: str | None,
    policy_ids: list[str],
    reason_code: str,
) -> AuthorizationDecision:
    return AuthorizationDecision(
        allowed=allowed,
        action=action,
        resource=resource,
        policy_id=",".join(policy_ids) or None,
        reason_code=reason_code,
        audit_id=f"auth-{uuid4().hex[:12]}",
    )
