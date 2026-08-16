from __future__ import annotations

import argparse
import json

from data_agent.planner import plan_question


def main() -> None:
    parser = argparse.ArgumentParser(prog="data-agent", description="智能数据探查 Agent CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="生成意图识别与任务拆解计划")
    plan_parser.add_argument("question", help="用户自然语言问题")

    args = parser.parse_args()
    if args.command == "plan":
        result = plan_question(args.question)
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
