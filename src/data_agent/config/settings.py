from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMSettings:
    model: str | None
    api_key: str
    base_url: str
    timeout_seconds: float
    max_retries: int
    enabled: bool

    @classmethod
    def from_env(cls) -> "LLMSettings":
        model = os.getenv("DATA_AGENT_LLM_MODEL")
        return cls(
            model=model,
            api_key=os.getenv("DATA_AGENT_LLM_API_KEY", "lm-studio"),
            base_url=os.getenv("DATA_AGENT_LLM_BASE_URL", "http://localhost:1234/v1"),
            timeout_seconds=float(os.getenv("DATA_AGENT_LLM_TIMEOUT_SECONDS", "30")),
            max_retries=int(os.getenv("DATA_AGENT_LLM_MAX_RETRIES", "2")),
            enabled=os.getenv("DATA_AGENT_USE_LLM", "true").lower() not in {"0", "false", "no"} and bool(model),
        )


@dataclass(frozen=True)
class EmbeddingSettings:
    """OpenAI-compatible embedding settings used by Milvus dense retrieval."""

    model: str | None
    api_key: str
    base_url: str
    dimension: int
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> "EmbeddingSettings":
        return cls(
            model=os.getenv("DATA_AGENT_EMBEDDING_MODEL"),
            api_key=os.getenv("DATA_AGENT_EMBEDDING_API_KEY", "lm-studio"),
            base_url=os.getenv("DATA_AGENT_EMBEDDING_BASE_URL", "http://localhost:1234/v1"),
            dimension=int(os.getenv("DATA_AGENT_EMBEDDING_DIM", "1024")),
            timeout_seconds=float(os.getenv("DATA_AGENT_EMBEDDING_TIMEOUT_SECONDS", "30")),
        )
