"""Shared runner scaffolding across OpenAI, Microsoft, and Mistral clients."""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional

from jsonschema import ValidationError as JSONSchemaValidationError
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from usb_agents.policy import PolicyConfig
from usb_agents.tool_client import EmbeddedToolClient, HttpToolClient, StdioToolClient, ToolClient
from usb_agents.tasks import TaskSpec
from usb_agents.telemetry import SpanCapture, configure_metrics, configure_tracer, record_tool_latency


REQUIRED_SPANS = {"agent.model_call", "agent.tool_list", "agent.final_output"}


@dataclass
class RunResult:
    runtime: str
    task_id: str
    policy_mode: str
    success: bool
    latency_ms: float
    tokens_prompt: int
    tokens_completion: int
    tool_calls: int
    approvals: List[Dict[str, Any]] = field(default_factory=list)
    failure_category: Optional[str] = None
    failure_reason: Optional[str] = None
    trace_path: Optional[str] = None
    transport: str = "http"
    trace_completeness: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row["approvals"] = json.dumps(self.approvals)
        row["metadata"] = json.dumps(self.metadata)
        return row


class GuardrailViolation(Exception):
    """Raised when content fails a guardrail check."""


class BaseRunner:
    runtime_name = "base"
    model_name = "synthetic-model"
    transport_mode = "embedded"

    def __init__(
        self,
        policy: PolicyConfig,
        traces_dir: Path,
        *,
        transport_mode: str = "embedded",
        base_url: Optional[str] = None,
        stdio_cmd: Optional[str] = None,
        tool_client: Optional[ToolClient] = None,
        owns_tool_client: bool = True,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        owns_loop: bool = True,
    ) -> None:
        self.policy = policy
        self.traces_dir = traces_dir
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.transport_mode = transport_mode
        self.transport = transport_mode
        configure_tracer(service_name=f"usb-agents-{self.runtime_name}")
        configure_metrics(service_name=f"usb-agents-{self.runtime_name}")
        self.tracer = trace.get_tracer(f"usb_agents.runners.{self.runtime_name}")
        self.tool_invocations = 0
        self.tool_metrics: Dict[str, List[float]] = defaultdict(list)
        self._loop = loop or asyncio.new_event_loop()
        self._owns_loop = owns_loop and loop is None
        if tool_client is not None:
            self.tool_client = tool_client
            self._owns_tool_client = owns_tool_client
        else:
            self._owns_tool_client = True
            if transport_mode == "http":
                if not base_url:
                    raise ValueError("HTTP transport requires base_url")
                self.tool_client = HttpToolClient(base_url)
            elif transport_mode == "stdio":
                self.tool_client = StdioToolClient(stdio_cmd)
            elif transport_mode == "embedded":
                from mcp_server.server import MCPServer  # Local import to avoid package cycle

                server = MCPServer(policy=policy, traces_dir=self.traces_dir)
                self.tool_client = EmbeddedToolClient(server)
            else:
                raise ValueError(f"Unsupported transport_mode '{transport_mode}'")

    # Public API -----------------------------------------------------
    def run_task(self, task: TaskSpec) -> RunResult:
        asyncio.set_event_loop(self._loop)
        try:
            return self._loop.run_until_complete(self._run_task_async(task))
        finally:
            asyncio.set_event_loop(None)

    def close(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            if hasattr(self, "tool_client") and self._owns_tool_client:
                self._loop.run_until_complete(self.tool_client.aclose())
        finally:
            if self._owns_loop:
                self._loop.close()
            asyncio.set_event_loop(None)

    # Internal helpers -----------------------------------------------
    async def _run_task_async(self, task: TaskSpec) -> RunResult:
        start = time.perf_counter()
        approvals: List[Dict[str, Any]] = []
        tokens_prompt = self.estimate_tokens(task.runner_prompt)
        tokens_completion = 0
        failure_category: Optional[str] = None
        failure_reason: Optional[str] = None
        success = True
        self.tool_invocations = 0
        self.tool_metrics = defaultdict(list)
        metadata: Dict[str, Any] = {}

        with SpanCapture() as capture:
            with self.tracer.start_as_current_span(
                "agent.run",
                attributes={
                    "gen_ai.request.model": self.model_name,
                    "agent.runtime": self.runtime_name,
                    "agent.task_id": task.task_id,
                },
            ) as run_span:
                try:
                    prompt = self.guard_input(task.runner_prompt)
                    with self.tracer.start_as_current_span(
                        "agent.model_call",
                        attributes={
                            "gen_ai.request.model": self.model_name,
                            "gen_ai.prompt.length": len(prompt),
                        },
                    ):
                        tool_plan = await self.plan_tools(task)
                        with self.tracer.start_as_current_span(
                            "agent.tool_list",
                            attributes={"gen_ai.tool.names": json.dumps(tool_plan)},
                        ):
                            pass
                        output = await self.execute_task(task, approvals, metadata)
                    output = self.guard_output(output)
                    tokens_completion = self.estimate_tokens(json.dumps(output))
                    with self.tracer.start_as_current_span(
                        "agent.final_output",
                        attributes={"gen_ai.response.output_length": tokens_completion},
                    ):
                        pass
                    run_span.set_status(Status(status_code=StatusCode.OK))
                    metadata["output"] = output
                except GuardrailViolation as exc:
                    success = False
                    failure_category = "guardrail"
                    failure_reason = str(exc)
                    run_span.record_exception(exc)
                    run_span.set_status(
                        Status(status_code=StatusCode.ERROR, description=str(exc))
                    )
                except PermissionError as exc:
                    success = False
                    failure_category = "approval"
                    failure_reason = str(exc)
                    run_span.record_exception(exc)
                    run_span.set_status(
                        Status(status_code=StatusCode.ERROR, description=str(exc))
                    )
                except JSONSchemaValidationError as exc:
                    success = False
                    failure_category = "schema"
                    failure_reason = exc.message
                    run_span.record_exception(exc)
                    run_span.set_status(
                        Status(status_code=StatusCode.ERROR, description=exc.message)
                    )
                except Exception as exc:  # pragma: no cover
                    success = False
                    failure_category = "runtime"
                    failure_reason = str(exc)
                    run_span.record_exception(exc)
                    run_span.set_status(
                        Status(status_code=StatusCode.ERROR, description=str(exc))
                    )

        latency_ms = (time.perf_counter() - start) * 1000
        spans = capture.finished_spans()
        trace_path = self.write_trace(task, spans)
        trace_completeness = self.compute_trace_completeness(spans)

        if self.tool_metrics:
            tool_stats: Dict[str, Dict[str, float]] = {}
            for tool_name, durations in self.tool_metrics.items():
                tool_stats[tool_name] = {
                    "count": len(durations),
                    "avg_ms": mean(durations),
                    "median_ms": median(durations),
                    "p95_ms": self._percentile(durations, 0.95),
                }
            metadata["tool_metrics"] = tool_stats
            metadata["tool_durations_ms"] = {tool: list(values) for tool, values in self.tool_metrics.items()}

        return RunResult(
            runtime=self.runtime_name,
            task_id=task.task_id,
            policy_mode=task.approval,
            success=success,
            latency_ms=latency_ms,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            tool_calls=self.tool_invocations,
            approvals=approvals,
            failure_category=failure_category,
            failure_reason=failure_reason,
            trace_path=str(trace_path) if trace_path else None,
            transport=self.transport,
            trace_completeness=trace_completeness,
            metadata=metadata,
        )

    async def execute_task(
        self,
        task: TaskSpec,
        approvals: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute the deterministic plan for the task; subclasses may override."""
        if task.task_id == "t1_repo_triage":
            return await self._task_repo_triage(approvals)
        if task.task_id == "t2_calendar_merge":
            return await self._task_calendar_merge(approvals)
        if task.task_id == "t3_http_etl":
            return await self._task_http_etl(approvals, metadata)
        if task.task_id == "t4_code_patch":
            return await self._task_code_patch(approvals)
        raise ValueError(f"Unknown task '{task.task_id}'")

    async def _task_repo_triage(self, approvals: List[Dict[str, Any]]) -> Dict[str, Any]:
        files = [
            "README.md",
            "mcp-server/README.md",
            "pyproject.toml",
        ]
        contents = []
        for file_path in files:
            result = await self.call_tool(
                "fs_read",
                {"path": file_path, "limit": 2000},
                approvals,
            )
            contents.append(result["content"])
        summary = {
            "summary": "usb-agents coordinates MCP portability evaluation across three runtimes.",
            "components": [
                "Common MCP server",
                "Deterministic task specs",
                "Runner harness for OpenAI, Microsoft, and Mistral",
            ],
            "risks": [
                "Preview SDK compatibility drift",
                "Telemetry exporter configuration errors",
            ],
        }
        return summary

    async def _task_calendar_merge(self, approvals: List[Dict[str, Any]]) -> Dict[str, Any]:
        read = await self.call_tool("fs_read", {"path": "tasks/calendar.json"}, approvals)
        calendar = json.loads(read["content"])
        proposed = [
            {
                "title": "Interop Dry Run",
                "time": "2024-07-03T16:00:00Z",
                "duration_minutes": 30,
            },
            {
                "title": "Retro",  # Duplicate, should be skipped.
                "time": "2024-07-04T09:00:00Z",
                "duration_minutes": 30,
            },
        ]
        approval = self.request_approval("calendar_merge", {"proposed": proposed}, approvals)
        if not approval["granted"]:
            raise PermissionError("calendar_merge denied")
        merge_result = await self.call_tool(
            "calendar_merge",
            {"proposed": proposed},
            approvals,
            approval_override=approval,
        )
        return {**merge_result, "proposed": proposed}

    async def _task_http_etl(
        self,
        approvals: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        data = await self.call_tool(
            "http_fetch",
            {"url": "https://example.com/synthetic-dataset"},
            approvals,
        )
        rows = data["data"]["users"]
        csv_lines = ["id,name,role"]
        for row in rows:
            csv_lines.append(f"{row['id']},{row['name']},{row['role']}")
        artifact_path = Path("tasks/artifacts/http_etl.csv")
        artifact_path.write_text("\n".join(csv_lines), encoding="utf-8")
        metadata["artifact_path"] = str(artifact_path)
        return {"artifact_path": str(artifact_path), "row_count": len(rows)}

    async def _task_code_patch(self, approvals: List[Dict[str, Any]]) -> Dict[str, Any]:
        module_path = Path("tasks/code_module/example.py")
        content = module_path.read_text(encoding="utf-8")
        if "def multiply" not in content:
            addition = "\n\n\ndef multiply(a: int, b: int) -> int:\n    \"\"\"Return the product of two integers.\"\"\"\n    return a * b\n"
            module_path.write_text(content.strip() + addition, encoding="utf-8")
        approval = self.request_approval("code_exec", {"command": "pytest"}, approvals)
        if not approval["granted"]:
            raise PermissionError("code_exec denied")
        exec_result = await self.call_tool(
            "code_exec",
            {"command": "pytest"},
            approvals,
            approval_override=approval,
        )
        return exec_result

    async def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        approvals: List[Dict[str, Any]],
        *,
        approval_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.tool_invocations += 1
        approval_payload: Optional[Dict[str, Any]] = None
        if approval_override:
            approval_payload = dict(approval_override)
        elif self.policy.requires_approval(name):
            decision = self.request_approval(name, arguments, approvals)
            if not decision["granted"]:
                raise PermissionError(f"{name} denied by guardrail")
            approval_payload = dict(decision)
        with self.tracer.start_as_current_span(
            f"agent.tool_call.{name}",
            attributes={
                "gen_ai.tool.name": name,
                "agent.transport": self.transport_mode,
            },
        ):
            start = time.perf_counter()
            result = await self.tool_client.call_tool(name, arguments, approval_payload)
            duration_ms = (time.perf_counter() - start) * 1000
            self.tool_metrics[name].append(duration_ms)
            record_tool_latency(name, duration_ms, runtime=self.runtime_name, transport=self.transport_mode)
        return result

    def request_approval(
        self, tool_name: str, arguments: Dict[str, Any], approvals: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        decision = {
            "granted": True,
            "reason": f"auto-approve:{tool_name}",
        }
        approvals.append({"tool": tool_name, **decision})
        with self.tracer.start_as_current_span(
            f"agent.approval.{tool_name}",
            attributes={
                "approval.tool": tool_name,
                "approval.granted": decision["granted"],
            },
        ):
            pass
        return decision

    async def plan_tools(self, task: TaskSpec) -> List[str]:
        if task.task_id == "t1_repo_triage":
            return ["fs_read"]
        if task.task_id == "t2_calendar_merge":
            return ["fs_read", "calendar_merge"]
        if task.task_id == "t3_http_etl":
            return ["http_fetch"]
        if task.task_id == "t4_code_patch":
            return ["code_exec"]
        return []

    def guard_input(self, prompt: str) -> str:
        banned = ["drop database", "pii", "ssn"]
        lowered = prompt.lower()
        if any(token in lowered for token in banned):
            raise GuardrailViolation("Input guardrail triggered")
        return prompt

    def guard_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        return output

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text.split()) // 3 + 1)

    def write_trace(self, task: TaskSpec, spans: List[Any]) -> Optional[Path]:
        if not spans:
            return None
        trace_path = self.traces_dir / f"{task.task_id}_{self.runtime_name}.ndjson"
        with trace_path.open("w", encoding="utf-8") as fh:
            for span in spans:
                fh.write(json.dumps(self.span_to_dict(span)) + "\n")
        return trace_path

    def span_to_dict(self, span: Any) -> Dict[str, Any]:
        return {
            "name": span.name,
            "context": {
                "trace_id": format(span.context.trace_id, "032x"),
                "span_id": format(span.context.span_id, "016x"),
            },
            "attributes": dict(span.attributes),
        }

    def compute_trace_completeness(self, spans: List[Any]) -> float:
        if not spans:
            return 0.0
        names = {span.name for span in spans}
        expected = set(REQUIRED_SPANS)
        if any(name.startswith("agent.tool_call") for name in names):
            expected.add("agent.tool_call")
        if any(name.startswith("agent.approval") for name in names):
            expected.add("agent.approval")
        hits = 0
        for requirement in expected:
            if requirement in {"agent.tool_call", "agent.approval"}:
                if any(name.startswith(requirement) for name in names):
                    hits += 1
            elif requirement in names:
                hits += 1
        return hits / len(expected)

    @staticmethod
    def _percentile(values: List[float], pct: float) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        k = (len(sorted_vals) - 1) * pct
        f = int(k)
        c = min(f + 1, len(sorted_vals) - 1)
        if f == c:
            return sorted_vals[f]
        return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)
