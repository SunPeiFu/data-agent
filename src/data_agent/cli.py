from __future__ import annotations

import argparse
import json
from uuid import uuid4

from data_agent.models import AccessContext, ClarificationResponse
from data_agent.planner import plan_question, resume_clarification


def main() -> None:
    parser = argparse.ArgumentParser(prog="data-agent", description="智能数据探查 Agent CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="生成意图识别与任务拆解计划")
    plan_parser.add_argument("question", help="用户自然语言问题")
    plan_parser.add_argument("--user-id", default="demo-user", help="鉴权用户标识")
    plan_parser.add_argument("--role", action="append", dest="roles", help="用户角色，可重复传入")

    resume_parser = subparsers.add_parser("resume", help="提交澄清回答并恢复原 LangGraph 会话")
    resume_parser.add_argument("--thread-id", required=True, help="澄清卡片中的 thread_id")
    resume_parser.add_argument("--clarification-id", required=True, help="澄清卡片 ID")
    resume_parser.add_argument("--value", required=True, help="用户确认的槽位值")
    resume_parser.add_argument("--option-id", help="选择候选卡片时提交的 option_id")
    resume_parser.add_argument("--state-version", required=True, type=int, help="澄清卡片状态版本")
    resume_parser.add_argument("--idempotency-key", help="客户端幂等键；不传时由 CLI 自动生成")

    args = parser.parse_args()
    if args.command == "plan":
        result = plan_question(
            args.question,
            access_context=AccessContext(
                user_id=args.user_id,
                roles=args.roles or ["data_admin"],
                tenant_id="demo",
            ),
        )
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    elif args.command == "resume":
        result = resume_clarification(
            ClarificationResponse(
                thread_id=args.thread_id,
                clarification_id=args.clarification_id,
                option_id=args.option_id,
                value=args.value,
                state_version=args.state_version,
                idempotency_key=args.idempotency_key or f"cli-{uuid4().hex}",
            )
        )
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
