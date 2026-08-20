"""Stable helpers shared by plan compilation and pre-execution enforcement."""

from __future__ import annotations

import json
from hashlib import sha256

from data_agent.domain.models import PlanningResult


def compute_plan_hash(result: PlanningResult) -> str:
    """Fingerprint the executable plan so later mutation cannot bypass validation."""
    payload = {
        "intent": result.intent.value,
        "metadata_query_mode": result.metadata_query_mode.value if result.metadata_query_mode else None,
        "task_steps": [step.model_dump(mode="json") for step in result.task_steps],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()
