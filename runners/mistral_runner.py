"""Mistral Agents API style runner with MCP connector parity."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from usb_agents.policy import PolicyConfig, load_policy
from usb_agents.runner_base import BaseRunner, GuardrailViolation, RunResult
from usb_agents.tasks import TaskSpec, load_tasks


class MistralRunner(BaseRunner):
    runtime_name = "mistral_agents"
    model_name = os.getenv("MISTRAL_AGENT_MODEL", "mistral-large-latest")

    def __init__(
        self,
        policy: PolicyConfig,
        traces_dir: Path,
        *,
        transport_mode: str = "embedded",
        base_url: Optional[str] = None,
        stdio_cmd: Optional[str] = None,
        tool_client=None,
        owns_tool_client: bool = True,
        loop=None,
        owns_loop: bool = True,
    ) -> None:
        super().__init__(
            policy,
            traces_dir,
            transport_mode=transport_mode,
            base_url=base_url,
            stdio_cmd=stdio_cmd,
            tool_client=tool_client,
            owns_tool_client=owns_tool_client,
            loop=loop,
            owns_loop=owns_loop,
        )
        self.transport = transport_mode

    def guard_output(self, output: Dict[str, str]) -> Dict[str, str]:
        if "error" in output:
            raise GuardrailViolation("Mistral response contained error field")
        return super().guard_output(output)

    def request_approval(
        self, tool_name: str, arguments: Dict, approvals: List[Dict]
    ) -> Dict[str, str]:
        decision = super().request_approval(tool_name, arguments, approvals)
        decision["reason"] = f"mistral-hitl:{tool_name}"
        approvals[-1]["reason"] = decision["reason"]
        return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mistral Agents simulator.")
    parser.add_argument("--task", help="Task ID to execute")
    parser.add_argument("--policy", default="mcp-server/policy.yaml")
    parser.add_argument("--traces-dir", default="traces")
    parser.add_argument(
        "--transport",
        choices=["embedded", "http", "stdio"],
        default="embedded",
        help="Transport mode for tool calls.",
    )
    parser.add_argument(
        "--http-url",
        default=os.getenv("MCP_SERVER_HTTP_URL", "http://127.0.0.1:8000"),
        help="HTTP base URL when using --transport http.",
    )
    parser.add_argument(
        "--stdio-cmd",
        default=os.getenv("MCP_SERVER_STDIO_CMD"),
        help="Command to launch the MCP server in stdio mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policy = load_policy(Path(args.policy))
    traces_dir = Path(args.traces_dir)
    base_url = args.http_url if args.transport == "http" else None
    runner = MistralRunner(
        policy,
        traces_dir,
        transport_mode=args.transport,
        base_url=base_url,
        stdio_cmd=args.stdio_cmd,
    )
    tasks = load_tasks(Path("tasks"))
    selected = [t for t in tasks if not args.task or t.task_id == args.task]
    if not selected:
        print("No matching tasks.")
        return 1
    results: List[RunResult] = []
    for task in selected:
        result = runner.run_task(task)
        print(json.dumps(result.to_row()))
        results.append(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
