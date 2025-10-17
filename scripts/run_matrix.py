"""Run the runtime x task matrix and compute aggregate metrics."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
import jsonschema

from eval.metrics import (
    MetricsBundle,
    bundle_to_dict,
    compare_to_baseline,
    compute_latency_alerts,
    compute_success_alerts,
    compute_stdio_wait_alerts,
    compute_metrics,
    compute_metrics_by_transport,
)
from runners.mistral_runner import MistralRunner
from runners.ms_runner import MicrosoftRunner
from runners.openai_runner import OpenAIRunner
from scripts.dashboard import build_json as build_dashboard_json
from scripts.dashboard import build_summary as build_dashboard_summary
from scripts.dashboard import render_markdown as render_dashboard_markdown
from scripts.report import render_markdown
from usb_agents.policy import load_policy
from usb_agents.runner_base import RunResult
from usb_agents.tool_client import StdioToolClient, ToolClient
from usb_agents.tasks import load_tasks

RUNNERS: List[Type] = [OpenAIRunner, MicrosoftRunner, MistralRunner]

METRICS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "threshold_ms": {"type": "number"},
        "success_threshold": {"type": "number"},
        "stdio_wait_threshold_ms": {"type": "number"},
        "transports": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "portability_success_rate": {"type": "number"},
                    "latency_p95": {"type": "number"},
                    "latency_p50": {"type": "number"},
                    "stdio_pool": {
                        "type": "object",
                        "properties": {
                            "concurrency": {"type": "number"},
                            "wait_ms": {"type": "object"},
                            "active_workers": {"type": "object"},
                            "available_workers": {"type": "object"},
                        },
                        "additionalProperties": True,
                    },
                },
                "required": ["portability_success_rate", "latency_p95"],
                "additionalProperties": True,
            },
        },
        "alerts": {"type": "object"},
    },
    "required": ["threshold_ms", "success_threshold", "transports"],
    "additionalProperties": True,
}


def validate_metrics_payload(payload: Dict[str, Any]) -> None:
    jsonschema.validate(instance=payload, schema=METRICS_SCHEMA)


def normalise_result(result: RunResult) -> Dict:
    return {
        "runtime": result.runtime,
        "task_id": result.task_id,
        "policy_mode": result.policy_mode,
        "success": result.success,
        "latency_ms": result.latency_ms,
        "tokens_prompt": result.tokens_prompt,
        "tokens_completion": result.tokens_completion,
        "tool_calls": result.tool_calls,
        "approvals": result.approvals,
        "failure_category": result.failure_category,
        "failure_reason": result.failure_reason,
        "trace_path": result.trace_path,
        "transport": result.transport,
        "trace_completeness": result.trace_completeness,
        "metadata": result.metadata,
    }


def write_csv(path: Path, rows: List[RunResult]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].to_row().keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_row())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run runtime x task matrix")
    parser.add_argument("--policy", default="mcp-server/policy.yaml")
    parser.add_argument("--results", default="results.csv")
    parser.add_argument("--markdown", default="results.md")
    parser.add_argument("--traces", default="traces")
    parser.add_argument(
        "--transport-json",
        default="transport_metrics.json",
        help="Path to write per-transport metrics JSON summary.",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        help="Directory to archive transport metrics per run (timestamped).",
    )
    parser.add_argument(
        "--dashboard-output",
        type=Path,
        help="Markdown file to write aggregated dashboard (defaults to archive dir/dashboard.md).",
    )
    parser.add_argument(
        "--dashboard-json",
        type=Path,
        help="JSON file to write aggregated dashboard (defaults to archive dir/dashboard.json).",
    )
    parser.add_argument(
        "--transport",
        choices=["embedded", "http", "stdio"],
        default=os.getenv("USB_AGENTS_TRANSPORT", "embedded"),
        help="Transport mode for runners (embedded=direct call, http=REST MCP server, stdio placeholder).",
    )
    parser.add_argument(
        "--http-url",
        default=os.getenv("MCP_SERVER_HTTP_URL", "http://127.0.0.1:8000"),
        help="HTTP MCP base URL when --transport http is used.",
    )
    parser.add_argument(
        "--server-host",
        default=os.getenv("MCP_SERVER_HOST", "127.0.0.1"),
        help="Host for auto-spawned MCP server when using HTTP transport.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=int(os.getenv("MCP_SERVER_PORT", 8000)),
        help="Port for auto-spawned MCP server when using HTTP transport.",
    )
    parser.add_argument(
        "--spawn-server",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Automatically spawn the MCP HTTP server (default when transport=http).",
    )
    parser.add_argument(
        "--compare-transports",
        action="store_true",
        help="Run the matrix for both embedded and HTTP transports and report deltas.",
    )
    parser.add_argument(
        "--transports",
        nargs="+",
        choices=["embedded", "http", "stdio"],
        help="Explicit list of transports to execute (overrides --transport/--compare-transports).",
    )
    parser.add_argument(
        "--stdio-cmd",
        default=os.getenv("MCP_SERVER_STDIO_CMD"),
        help="Command to launch the MCP server in stdio mode (default inferred if unset).",
    )
    parser.add_argument(
        "--stdio-concurrency",
        type=int,
        default=int(os.getenv("USB_AGENTS_STDIO_CONCURRENCY", "1")),
        help="Maximum number of concurrent stdio workers to maintain.",
    )
    parser.add_argument(
        "--stdio-warmup",
        type=int,
        default=int(os.getenv("USB_AGENTS_STDIO_WARMUP", "0")),
        help="Number of stdio workers to pre-warm (up to concurrency).",
    )
    parser.add_argument(
        "--stdio-wait-threshold-ms",
        type=float,
        default=float(os.getenv("USB_AGENTS_STDIO_WAIT_THRESHOLD_MS", 100.0)),
        help="Threshold for stdio worker wait-time p95 alerts (ms).",
    )
    parser.add_argument(
        "--latency-threshold-ms",
        type=float,
        default=float(os.getenv("USB_AGENTS_LATENCY_THRESHOLD_MS", 500.0)),
        help="Latency threshold (ms) for p95 alerts per transport.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=float(os.getenv("USB_AGENTS_SUCCESS_THRESHOLD", 0.75)),
        help="Minimum success rate per transport before raising an alert.",
    )
    parser.add_argument(
        "--fail-on-alert",
        action="store_true",
        help="Exit with non-zero status if latency or success alerts are triggered.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Optional baseline transport_metrics.json for regression comparison.",
    )
    parser.add_argument(
        "--max-latency-delta",
        type=float,
        default=float(os.getenv("USB_AGENTS_MAX_LATENCY_DELTA", 50.0)),
        help="Maximum allowed increase in latency p95 relative to baseline (ms).",
    )
    parser.add_argument(
        "--max-success-delta",
        type=float,
        default=float(os.getenv("USB_AGENTS_MAX_SUCCESS_DELTA", 0.05)),
        help="Maximum allowed decrease in success rate relative to baseline.",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="When provided with --baseline, overwrite the baseline file if no alerts are triggered.",
    )
    parser.add_argument(
        "--alerts-output",
        type=Path,
        default=Path("alerts_summary.md"),
        help="Path to write a human-readable alert summary.",
    )
    return parser.parse_args()


def start_http_server(policy_path: Path, host: str, port: int) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "mcp_server.server",
        "--transport",
        "http",
        "--host",
        host,
        "--port",
        str(port),
        "--policy",
        str(policy_path),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def wait_for_server(base_url: str, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    health_url = f"{base_url.rstrip('/')}/health"
    while time.time() < deadline:
        try:
            response = httpx.get(health_url, timeout=1.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.25)
        else:
            time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for MCP server at {health_url}")


def main() -> int:
    args = parse_args()
    policy = load_policy(Path(args.policy))
    tasks = load_tasks(Path("tasks"))
    traces_dir = Path(args.traces)
    traces_dir.mkdir(parents=True, exist_ok=True)

    csv_rows: List[RunResult] = []
    normalised: List[Dict] = []

    if args.transports:
        transports = args.transports
    elif args.compare_transports:
        transports = ["embedded", "http"]
    else:
        transports = [args.transport]

    transport_extras: Dict[str, Dict[str, Any]] = {}
    transports_dict: Dict[str, Dict[str, Any]] = {}
    stdio_wait_alerts: List[Dict[str, object]] = []
    baseline_transports: Dict[str, Any] = {}

    for transport_mode in transports:
        spawn_server = args.spawn_server
        if spawn_server is None:
            spawn_server = transport_mode == "http"

        server_process: subprocess.Popen | None = None
        shared_loop: Optional[asyncio.AbstractEventLoop] = None
        shared_tool_client: Optional[ToolClient] = None
        base_url = args.http_url if transport_mode == "http" else None
        if transport_mode == "http" and spawn_server:
            base_url = f"http://{args.server_host}:{args.server_port}"
            server_process = start_http_server(Path(args.policy), args.server_host, args.server_port)
            try:
                wait_for_server(base_url)
            except Exception as exc:
                if server_process.poll() is None:
                    server_process.terminate()
                    try:
                        stdout_data, stderr_data = server_process.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        server_process.kill()
                        stdout_data, stderr_data = server_process.communicate(timeout=5)
                else:
                    stdout_data, stderr_data = server_process.communicate()
                sys.stderr.write("Failed to start MCP HTTP server\n")
                if stdout_data:
                    sys.stderr.write(stdout_data.decode())
                if stderr_data:
                    sys.stderr.write(stderr_data.decode())
                raise exc
        if transport_mode == "stdio":
            shared_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(shared_loop)
            try:
                shared_tool_client = StdioToolClient(
                    args.stdio_cmd,
                    concurrency=args.stdio_concurrency,
                )
                warmup_count = min(args.stdio_warmup, args.stdio_concurrency)
                if warmup_count > 0:
                    shared_loop.run_until_complete(shared_tool_client.warmup(warmup_count))
            finally:
                asyncio.set_event_loop(None)

        try:
            for runner_cls in RUNNERS:
                try:
                    runner = runner_cls(
                        policy,
                        traces_dir,
                        transport_mode=transport_mode,
                        base_url=base_url,
                        stdio_cmd=args.stdio_cmd,
                        tool_client=shared_tool_client,
                        owns_tool_client=shared_tool_client is None,
                        loop=shared_loop,
                        owns_loop=shared_loop is None,
                    )
                except NotImplementedError as exc:
                    sys.stderr.write(f"Skipping transport '{transport_mode}': {exc}\n")
                    break
                try:
                    for task in tasks:
                        result = runner.run_task(task)
                        csv_rows.append(result)
                        normalised.append(normalise_result(result))
                finally:
                    runner.close()
        finally:
            if server_process:
                server_process.send_signal(signal.SIGTERM)
                try:
                    server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server_process.kill()
            if shared_tool_client and shared_loop:
                asyncio.set_event_loop(shared_loop)
                try:
                    stats = shared_loop.run_until_complete(shared_tool_client.collect_stats())
                    if stats["wait_ms"]["count"] or stats["active_workers"]["count"]:
                        transport_extras.setdefault(transport_mode, {})["stdio_pool"] = stats
                    shared_loop.run_until_complete(shared_tool_client.aclose())
                finally:
                    asyncio.set_event_loop(None)
            if shared_loop:
                shared_loop.close()

    transport_metrics = compute_metrics_by_transport(normalised)
    latency_alerts = compute_latency_alerts(transport_metrics, args.latency_threshold_ms)
    success_alerts = compute_success_alerts(normalised, args.success_threshold)
    baseline_alerts: List[Dict[str, object]] = []
    baseline_path = Path(args.baseline) if args.baseline else None
    if baseline_path and baseline_path.exists():
        baseline_data = json.loads(baseline_path.read_text(encoding="utf-8"))
        validate_metrics_payload(baseline_data)
        baseline_transports = baseline_data.get("transports", baseline_data)
        baseline_alerts = compare_to_baseline(
            transport_metrics,
            baseline_transports,
            args.max_latency_delta,
            args.max_success_delta,
        )
    stdio_wait_alerts = compute_stdio_wait_alerts(
        transports_dict,
        args.stdio_wait_threshold_ms,
    )

    transports_dict = {name: bundle_to_dict(bundle) for name, bundle in transport_metrics.items()}
    for transport, extras in transport_extras.items():
        transports_dict.setdefault(transport, {}).update(extras)

    stdio_wait_alerts = compute_stdio_wait_alerts(transports_dict, args.stdio_wait_threshold_ms)
    combined_alerts = {
        "latency": latency_alerts,
        "success": success_alerts,
        "baseline": baseline_alerts,
        "stdio_wait": stdio_wait_alerts,
    }
    write_csv(Path(args.results), csv_rows)
    metrics: MetricsBundle = compute_metrics(normalised)
    markdown = render_markdown(
        normalised,
        metrics,
        latency_alerts,
        success_alerts,
        baseline_alerts,
        transports_dict,
        stdio_wait_alerts,
        baseline_transports or None,
    )
    Path(args.markdown).write_text(markdown, encoding="utf-8")

    transport_payload = {
        "threshold_ms": args.latency_threshold_ms,
        "success_threshold": args.success_threshold,
        "stdio_wait_threshold_ms": args.stdio_wait_threshold_ms,
        "transports": transports_dict,
        "alerts": combined_alerts,
    }
    validate_metrics_payload(transport_payload)
    Path(args.transport_json).write_text(json.dumps(transport_payload, indent=2), encoding="utf-8")
    alerts_output_path = args.alerts_output
    alert_sections = [
        ("Latency Alerts", latency_alerts),
        ("Success Alerts", success_alerts),
        ("Baseline Alerts", baseline_alerts),
        ("Stdio Wait Alerts", stdio_wait_alerts),
    ]
    lines = ["# Alerts Summary", ""]
    any_alerts = any(section for _, section in alert_sections if section)
    if not any_alerts:
        lines.append("No alerts triggered.")
    else:
        for title, entries in alert_sections:
            if not entries:
                continue
            lines.append(f"## {title}")
            for entry in entries:
                formatted = ", ".join(f"{k}={v}" for k, v in entry.items())
                lines.append(f"- {formatted}")
            lines.append("")
    if args.update_baseline and baseline_path:
        lines.append("")
        if any_alerts:
            lines.append(f"Baseline NOT updated: {baseline_path}")
        else:
            lines.append(f"Baseline updated: {baseline_path}")
    alerts_output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    if args.archive_dir:
        archive_dir = Path(args.archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        archive_path = archive_dir / f"{timestamp}-transport_metrics.json"
        archive_path.write_text(json.dumps(transport_payload, indent=2), encoding="utf-8")
        summary = build_dashboard_summary(archive_dir)
        dashboard_md = render_dashboard_markdown(summary)
        dashboard_json = build_dashboard_json(summary)
        md_path = args.dashboard_output or (archive_dir / "dashboard.md")
        json_path = args.dashboard_json or (archive_dir / "dashboard.json")
        md_path.write_text(dashboard_md + "\n", encoding="utf-8")
        json_path.write_text(json.dumps(dashboard_json, indent=2), encoding="utf-8")
    if args.update_baseline and baseline_path:
        if any_alerts:
            sys.stderr.write("Skipping baseline update due to active alerts.\n")
        else:
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(json.dumps(transport_payload, indent=2), encoding="utf-8")
    if args.fail_on_alert and (latency_alerts or success_alerts or baseline_alerts or stdio_wait_alerts):
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
