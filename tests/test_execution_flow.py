from __future__ import annotations

from data_agent.application.planning.service import run_question
from data_agent.domain.models import (
    TaskExecutionResult,
    TaskExecutionStatus,
    ToolCallStatus,
    ToolObservation,
)


def test_run_question_executes_plan_and_collects_observations(monkeypatch) -> None:
    class FakeExecutor:
        def execute(self, steps, **kwargs):
            terminal = steps[-1]
            observation = ToolObservation(
                tool_call_id="tool-test",
                run_id=kwargs["run_id"],
                step_id=terminal.step_id,
                tool_name=terminal.tool_name,
                action=terminal.action,
                status=ToolCallStatus.SUCCESS,
                attempts=1,
                output={"assets": [], "total": 0},
            )
            return TaskExecutionResult(
                status=TaskExecutionStatus.SUCCESS,
                observations=[observation],
                terminal_outputs={terminal.step_id: observation.output},
            )

    monkeypatch.setattr("data_agent.application.execution.nodes.TaskExecutor", FakeExecutor)

    result = run_question("营销域 DWD 层支付相关表有哪些")

    assert result.execution_status == TaskExecutionStatus.SUCCESS
    assert result.final_output == {"assets": [], "total": 0}
    assert result.tool_observations[0].tool_call_id == "tool-test"
    assert any("工具执行: status=success" in note for note in result.notes)
