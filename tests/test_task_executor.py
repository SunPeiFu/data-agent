from __future__ import annotations

import asyncio

from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from typing import cast

from data_agent.domain.models import (
    AccessContext,
    ExtractedEntities,
    IntentType,
    TaskExecutionStatus,
    TaskStep,
    ToolCallStatus,
)
from data_agent.tools.executor import TaskExecutor
from data_agent.tools.registry import RegisteredTool, RetryPolicy, ToolRegistry


class ValueInput(BaseModel):
    value: int


class BoundInput(BaseModel):
    source: dict[str, int]
    multiplier: int


class ValueOutput(BaseModel):
    value: int


def _definition(service: str, action: str, tool: StructuredTool, **kwargs: object) -> RegisteredTool:
    return RegisteredTool(
        service=service,
        action=action,
        tool=tool,
        input_schema=cast(type[BaseModel], tool.args_schema),
        output_schema=ValueOutput,
        intents=frozenset({IntentType.METADATA_SEARCH}),
        required_scope="metadata:read",
        **kwargs,
    )


def test_executor_resolves_step_output_bindings() -> None:
    first = StructuredTool.from_function(
        func=lambda value: {"value": value},
        name="test_first",
        description="Return one value.",
        args_schema=ValueInput,
    )
    second = StructuredTool.from_function(
        func=lambda source, multiplier: {"value": source["value"] * multiplier},
        name="test_second",
        description="Multiply a bound value.",
        args_schema=BoundInput,
    )
    registry = ToolRegistry()
    registry.register(_definition("test", "first", first))
    registry.register(_definition("test", "second", second))
    steps = [
        TaskStep(step_id=1, tool_name="test", action="first", params={"value": 2}),
        TaskStep(
            step_id=2,
            tool_name="test",
            action="second",
            params={"multiplier": 3},
            input_bindings={"source": 1},
            depends_on=[1],
        ),
    ]

    result = TaskExecutor(registry).execute(
        steps,
        intent=IntentType.METADATA_SEARCH,
        access_context=AccessContext(user_id="admin", roles=["data_admin"]),
        entities=ExtractedEntities(),
        run_id="run-test",
    )

    assert result.status == TaskExecutionStatus.SUCCESS
    assert result.terminal_outputs == {2: {"value": 6}}
    assert all(item.status == ToolCallStatus.SUCCESS for item in result.observations)
    assert all(item.run_id == "run-test" for item in result.observations)


def test_executor_retries_transient_tool_failure() -> None:
    attempts = 0

    def flaky(value: int) -> dict[str, int]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("temporary")
        return {"value": value}

    tool = StructuredTool.from_function(
        func=flaky,
        name="test_flaky",
        description="Fail once.",
        args_schema=ValueInput,
    )
    registry = ToolRegistry()
    registry.register(
        _definition("test", "flaky", tool, retry_policy=RetryPolicy(max_attempts=2))
    )

    result = TaskExecutor(registry).execute(
        [TaskStep(step_id=1, tool_name="test", action="flaky", params={"value": 7})],
        intent=IntentType.METADATA_SEARCH,
        access_context=AccessContext(user_id="admin", roles=["data_admin"]),
        entities=ExtractedEntities(),
    )

    assert result.status == TaskExecutionStatus.SUCCESS
    assert result.observations[0].attempts == 2


def test_executor_enforces_tool_timeout() -> None:
    async def slow(value: int) -> dict[str, int]:
        await asyncio.sleep(0.05)
        return {"value": value}

    tool = StructuredTool.from_function(
        coroutine=slow,
        name="test_slow",
        description="Complete too slowly.",
        args_schema=ValueInput,
    )
    registry = ToolRegistry()
    registry.register(_definition("test", "slow", tool, timeout_seconds=0.01))

    result = TaskExecutor(registry).execute(
        [TaskStep(step_id=1, tool_name="test", action="slow", params={"value": 1})],
        intent=IntentType.METADATA_SEARCH,
        access_context=AccessContext(user_id="admin", roles=["data_admin"]),
        entities=ExtractedEntities(),
    )

    assert result.status == TaskExecutionStatus.FAILED
    assert result.observations[0].status == ToolCallStatus.TIMEOUT
    assert result.observations[0].error_code == "TOOL_TIMEOUT"
