from __future__ import annotations

import os
import sqlite3
from enum import Enum
from functools import lru_cache
from inspect import isclass
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel


@lru_cache(maxsize=1)
def get_planner_checkpointer() -> SqliteSaver:
    """Create one process-wide SQLite checkpointer for resumable local HITL workflows.

    SQLite makes the interview demo survive process restarts. Production deployments should replace
    this boundary with PostgresSaver or the company's durable workflow-state service.
    """
    configured_path = os.getenv("DATA_AGENT_CHECKPOINT_DB", ".data-agent/checkpoints.sqlite3")
    path = Path(configured_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    checkpointer = SqliteSaver(connection, serde=JsonPlusSerializer(allowed_msgpack_modules=_trusted_state_types()))
    checkpointer.setup()
    return checkpointer


def _trusted_state_types() -> list[type]:
    """Allow only project-owned Pydantic models and enums during checkpoint deserialization."""
    from data_agent.domain import models
    from data_agent.intelligence import hybrid_router, llm_analyzer

    trusted: list[type] = []
    for module in [models, hybrid_router, llm_analyzer]:
        for value in vars(module).values():
            if not isclass(value) or value.__module__ != module.__name__:
                continue
            if issubclass(value, (BaseModel, Enum)):
                trusted.append(value)
    return trusted
