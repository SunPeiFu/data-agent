from __future__ import annotations

import json
import re
from typing import Any

import httpx

from data_agent.settings import LLMSettings


class LLMClientError(RuntimeError):
    pass


class OpenAICompatibleChatClient:
    """OpenAI-compatible chat client for LM Studio, vLLM, Ollama proxies, or cloud models."""

    def __init__(self, settings: LLMSettings | None = None) -> None:
        self.settings = settings or LLMSettings.from_env()

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if not self.settings.enabled or not self.settings.model:
            raise LLMClientError("LLM is not configured. Set DATA_AGENT_LLM_MODEL to enable it.")

        payload = {
            "model": self.settings.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        url = self.settings.base_url.rstrip("/") + "/chat/completions"

        last_error: Exception | None = None
        for _ in range(self.settings.max_retries + 1):
            try:
                with httpx.Client(timeout=self.settings.timeout_seconds) as client:
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return _parse_json_object(content)
            except Exception as exc:  # noqa: BLE001 - errors are normalized for caller fallback.
                last_error = exc
        raise LLMClientError(f"LLM request failed: {last_error}") from last_error


def _parse_json_object(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        raise LLMClientError("LLM response did not contain a JSON object.")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise LLMClientError("LLM response JSON must be an object.")
    return parsed
