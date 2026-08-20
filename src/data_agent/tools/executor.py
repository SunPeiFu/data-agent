"""Dependency-aware executor for deterministic TaskStep DAGs."""

from __future__ import annotations

import asyncio
from time import perf_counter
from uuid import uuid4

from pydantic import ValidationError

from data_agent.domain.models import (
    AccessContext,
    ExtractedEntities,
    IntentType,
    TaskExecutionResult,
    TaskExecutionStatus,
    TaskStep,
    ToolCallStatus,
    ToolObservation,
)
from data_agent.tools.factory import get_default_tool_registry
from data_agent.tools.registry import RegisteredTool, ToolRegistry


class TaskExecutor:
    """Execute ready DAG steps concurrently while preserving policy and typed boundaries."""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or get_default_tool_registry()

    def execute(
        self,
        steps: list[TaskStep],
        *,
        intent: IntentType,
        access_context: AccessContext,
        entities: ExtractedEntities,
        run_id: str | None = None,
    ) -> TaskExecutionResult:
        return asyncio.run(
            self.aexecute(
                steps,
                intent=intent,
                access_context=access_context,
                entities=entities,
                run_id=run_id,
            )
        )

    async def aexecute(
        self,
        steps: list[TaskStep],
        *,
        intent: IntentType,
        access_context: AccessContext,
        entities: ExtractedEntities,
        run_id: str | None = None,
    ) -> TaskExecutionResult:
        pending = {step.step_id: step for step in steps}
        observations: dict[int, ToolObservation] = {}

        while pending:
            ready = [
                step
                for step in pending.values()
                if all(dependency in observations for dependency in step.depends_on)
            ]
            if not ready:
                for step in pending.values():
                    observations[step.step_id] = _observation(
                        step, run_id, ToolCallStatus.SKIPPED, error_code="UNRESOLVED_DEPENDENCY",
                        error_message="任务依赖无法解析，计划可能包含循环或非法依赖。",
                    )
                break

            wave = await asyncio.gather(
                *[
                    self._execute_step(step, observations, intent, access_context, entities, run_id)
                    for step in ready
                ]
            )
            for observation in wave:
                observations[observation.step_id] = observation
                pending.pop(observation.step_id, None)

        ordered = [observations[step.step_id] for step in steps]
        terminal_ids = set(observations) - {dependency for step in steps for dependency in step.depends_on}
        terminal_outputs = {
            step_id: observations[step_id].output
            for step_id in terminal_ids
            if observations[step_id].status == ToolCallStatus.SUCCESS
        }
        return TaskExecutionResult(
            status=_execution_status(ordered),
            observations=ordered,
            terminal_outputs=terminal_outputs,
        )

    async def _execute_step(
        self,
        step: TaskStep,
        observations: dict[int, ToolObservation],
        intent: IntentType,
        access_context: AccessContext,
        entities: ExtractedEntities,
        run_id: str | None,
    ) -> ToolObservation:
        try:
            definition = self.registry.get(step.tool_name, step.action)
        except KeyError as exc:
            return _observation(step, run_id, ToolCallStatus.FAILED, error_code="TOOL_NOT_FOUND", error_message=str(exc))

        if intent not in definition.intents:
            return _observation(
                step, run_id, ToolCallStatus.FORBIDDEN, error_code="TOOL_INTENT_DENIED",
                error_message=f"工具不允许用于 intent={intent.value}。",
            )
        if not self.registry.is_intent_authorized(intent, access_context, entities):
            return _observation(
                step, run_id, ToolCallStatus.FORBIDDEN, error_code="INTENT_AUTH_DENIED",
                error_message=f"缺少意图权限 intent={intent.value}。",
            )
        if not self.registry.is_authorized(definition, access_context, entities):
            return _observation(
                step, run_id, ToolCallStatus.FORBIDDEN, error_code="TOOL_AUTH_DENIED",
                error_message=f"缺少工具权限 {definition.required_scope}。",
            )

        failed_dependencies = [
            dependency
            for dependency in step.depends_on
            if observations[dependency].status != ToolCallStatus.SUCCESS
        ]
        if failed_dependencies and not definition.allow_failed_dependencies:
            return _observation(
                step, run_id, ToolCallStatus.SKIPPED, error_code="DEPENDENCY_FAILED",
                error_message=f"依赖步骤失败: {failed_dependencies}。",
            )

        tool_input = dict(step.params)
        for input_name, dependency in step.input_bindings.items():
            dependency_observation = observations.get(dependency)
            tool_input[input_name] = (
                dependency_observation.output
                if dependency_observation and dependency_observation.status == ToolCallStatus.SUCCESS
                else {}
            )
        return await _invoke_with_policy(definition, step, tool_input, run_id)


async def _invoke_with_policy(
    definition: RegisteredTool,
    step: TaskStep,
    tool_input: dict[str, object],
    run_id: str | None,
) -> ToolObservation:
    started = perf_counter()
    last_error: Exception | None = None
    attempts = 0
    for attempts in range(1, definition.retry_policy.max_attempts + 1):
        try:
            raw_output = await asyncio.wait_for(
                definition.tool.ainvoke(tool_input),
                timeout=definition.timeout_seconds,
            )
            output = definition.output_schema.model_validate(raw_output).model_dump(mode="json")
            return _observation(
                step, run_id, ToolCallStatus.SUCCESS, attempts=attempts,
                duration_ms=(perf_counter() - started) * 1000, output=output,
            )
        except asyncio.TimeoutError as exc:
            last_error = exc
            if attempts == definition.retry_policy.max_attempts:
                return _observation(
                    step, run_id, ToolCallStatus.TIMEOUT, attempts=attempts,
                    duration_ms=(perf_counter() - started) * 1000,
                    error_code="TOOL_TIMEOUT",
                    error_message=f"工具执行超过 {definition.timeout_seconds} 秒。",
                )
        except ValidationError as exc:
            last_error = exc
            break
        except Exception as exc:
            last_error = exc
            if attempts == definition.retry_policy.max_attempts:
                break
        if definition.retry_policy.backoff_seconds:
            await asyncio.sleep(definition.retry_policy.backoff_seconds * attempts)

    return _observation(
        step, run_id, ToolCallStatus.FAILED, attempts=attempts,
        duration_ms=(perf_counter() - started) * 1000,
        error_code=type(last_error).__name__ if last_error else "TOOL_FAILED",
        error_message=str(last_error)[:500] if last_error else "工具执行失败。",
    )


def _observation(
    step: TaskStep,
    run_id: str | None,
    status: ToolCallStatus,
    *,
    attempts: int = 0,
    duration_ms: float = 0.0,
    output: object = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> ToolObservation:
    return ToolObservation(
        tool_call_id=f"tool-{uuid4().hex}",
        run_id=run_id,
        step_id=step.step_id,
        tool_name=step.tool_name,
        action=step.action,
        status=status,
        attempts=attempts,
        duration_ms=round(duration_ms, 3),
        output=output,
        error_code=error_code,
        error_message=error_message,
    )


def _execution_status(observations: list[ToolObservation]) -> TaskExecutionStatus:
    statuses = {observation.status for observation in observations}
    if statuses == {ToolCallStatus.SUCCESS} or not observations:
        return TaskExecutionStatus.SUCCESS
    if ToolCallStatus.SUCCESS in statuses:
        return TaskExecutionStatus.PARTIAL
    return TaskExecutionStatus.FAILED
