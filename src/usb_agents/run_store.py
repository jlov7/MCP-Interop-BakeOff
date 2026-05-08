"""Local run artifact storage for the CLI and dashboard API."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from difflib import unified_diff
from pathlib import Path
from typing import Any

from eval.metrics import MetricsBundle, compute_metrics
from usb_agents.adapters import runtime_adapters as registry_runtime_adapters
from usb_agents.config import load_config
from usb_agents.models import (
    ApprovalDecision,
    ArtifactManifest,
    BenchmarkSuite,
    CaseResult,
    MetricSummary,
    Run,
    RuntimeAdapter,
    TaskCase,
    TraceEvent,
    TransportAdapter,
)
from usb_agents.redaction import redact
from usb_agents.tasks import load_tasks

RUNS_ROOT = Path(".usb-agents") / "runs"


def ensure_runs_root() -> Path:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    return RUNS_ROOT


def default_suite() -> BenchmarkSuite:
    config = load_config()
    cases = [
        TaskCase(
            id=task.task_id,
            name=task.name,
            description=task.description.strip(),
            approval=task.approval,
            mode=task.mode,
            expected_artifact=task.expected_artifact,
        )
        for task in load_tasks(config.suite_path)
    ]
    return BenchmarkSuite(cases=cases)


def runtime_adapters() -> list[RuntimeAdapter]:
    return registry_runtime_adapters(load_config().runtimes)


def transport_adapters() -> list[TransportAdapter]:
    return [
        TransportAdapter(id="embedded", name="Embedded", protocol="in-process simulator"),
        TransportAdapter(id="http", name="Streamable HTTP", protocol="MCP JSON-RPC over HTTP"),
        TransportAdapter(id="stdio", name="Stdio", protocol="MCP JSON-RPC over stdio"),
    ]


def result_to_case_result(row: Any) -> CaseResult:
    approvals = [
        ApprovalDecision(
            tool=item.get("tool", "unknown"),
            granted=bool(item.get("granted", False)),
            reason=str(item.get("reason", "")),
        )
        for item in row.approvals
    ]
    return CaseResult(
        runtime=row.runtime,
        task_id=row.task_id,
        transport=row.transport,
        success=row.success,
        latency_ms=row.latency_ms,
        approvals=approvals,
        tool_calls=row.tool_calls,
        failure_category=row.failure_category,
        failure_reason=row.failure_reason,
        trace_path=row.trace_path,
        metadata=redact(row.metadata),
    )


def metric_summary_from_bundle(bundle: MetricsBundle) -> MetricSummary:
    return MetricSummary(
        portability_success_rate=bundle.portability_success_rate,
        trace_completeness=bundle.trace_completeness,
        latency_p50=bundle.latency_p50,
        latency_p95=bundle.latency_p95,
        approvals=bundle.approvals,
        token_total=bundle.token_total,
        failure_histogram=bundle.failure_histogram,
    )


def write_run(run: Run, run_root: Path) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    path = run_root / "run.json"
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return path


def read_run(path: Path) -> Run:
    return Run.model_validate_json(path.read_text(encoding="utf-8"))


def list_runs() -> list[Run]:
    runs = [read_run(path) for path in sorted(ensure_runs_root().glob("*/run.json"), reverse=True)]
    return runs or [demo_run()]


def get_run(run_id: str) -> Run | None:
    for run in list_runs():
        if run.id == run_id:
            return run
    return None


def run_root_for_id(run_id: str) -> Path | None:
    for path in ensure_runs_root().glob("*/run.json"):
        run = read_run(path)
        if run.id == run_id:
            return path.parent
    if run_id == "demo_archived_portability":
        return Path("examples/mcp-server")
    return None


def resolve_artifact_path(run_id: str, artifact_path: str) -> Path | None:
    run = get_run(run_id)
    run_root = run_root_for_id(run_id)
    if run is None or run_root is None:
        return None
    if artifact_path not in {item.path for item in run.artifacts}:
        return None

    root = run_root.resolve()
    target = (root / artifact_path).resolve()
    if not target.is_relative_to(root) or not target.exists() or not target.is_file():
        return None
    return target


def compare_runs(run_id: str, baseline_id: str) -> dict[str, object] | None:
    run = get_run(run_id)
    baseline = get_run(baseline_id)
    if run is None or baseline is None:
        return None
    return {
        "run_id": run.id,
        "baseline_id": baseline.id,
        "deltas": {
            "portability_success_rate": (
                run.metrics.portability_success_rate - baseline.metrics.portability_success_rate
            ),
            "trace_completeness": (
                run.metrics.trace_completeness - baseline.metrics.trace_completeness
            ),
            "latency_p95": run.metrics.latency_p95 - baseline.metrics.latency_p95,
            "approvals": run.metrics.approvals - baseline.metrics.approvals,
            "token_total": run.metrics.token_total - baseline.metrics.token_total,
        },
        "regressions": _regressions(run, baseline),
    }


def compare_artifact(run_id: str, baseline_id: str, artifact_path: str) -> dict[str, object] | None:
    candidate = resolve_artifact_path(run_id, artifact_path)
    baseline = resolve_artifact_path(baseline_id, artifact_path)
    if baseline is None:
        baseline = _resolve_artifact_by_name(baseline_id, Path(artifact_path).name)
    if candidate is None or baseline is None:
        return None

    candidate_text = candidate.read_text(encoding="utf-8", errors="replace")
    baseline_text = baseline.read_text(encoding="utf-8", errors="replace")
    diff_lines = list(
        unified_diff(
            baseline_text.splitlines(),
            candidate_text.splitlines(),
            fromfile=f"{baseline_id}:{baseline.name}",
            tofile=f"{run_id}:{candidate.name}",
            lineterm="",
        )
    )
    return {
        "run_id": run_id,
        "baseline_id": baseline_id,
        "path": artifact_path,
        "baseline_path": str(baseline),
        "candidate_path": str(candidate),
        "changed": baseline_text != candidate_text,
        "baseline": baseline_text,
        "candidate": candidate_text,
        "diff": "\n".join(diff_lines),
        "stats": {
            "baseline_lines": len(baseline_text.splitlines()),
            "candidate_lines": len(candidate_text.splitlines()),
            "diff_lines": len(diff_lines),
        },
    }


def _resolve_artifact_by_name(run_id: str, name: str) -> Path | None:
    run = get_run(run_id)
    if run is None:
        return None
    for artifact in run.artifacts:
        if Path(artifact.path).name == name:
            return resolve_artifact_path(run_id, artifact.path)
    return None


def _regressions(run: Run, baseline: Run) -> list[dict[str, object]]:
    regressions: list[dict[str, object]] = []
    if run.metrics.portability_success_rate < baseline.metrics.portability_success_rate:
        regressions.append(
            {
                "metric": "portability_success_rate",
                "severity": "error",
                "message": "Portability success rate decreased.",
            }
        )
    if run.metrics.trace_completeness < baseline.metrics.trace_completeness:
        regressions.append(
            {
                "metric": "trace_completeness",
                "severity": "warning",
                "message": "Trace completeness decreased.",
            }
        )
    if run.metrics.latency_p95 > baseline.metrics.latency_p95:
        regressions.append(
            {
                "metric": "latency_p95",
                "severity": "warning",
                "message": "Latency p95 increased.",
            }
        )
    return regressions


def create_run_from_results(
    rows: list[Any],
    *,
    command: str,
    run_root: Path,
    artifacts: list[ArtifactManifest] | None = None,
) -> Run:
    normalised = [
        {
            "runtime": row.runtime,
            "task_id": row.task_id,
            "policy_mode": row.policy_mode,
            "success": row.success,
            "latency_ms": row.latency_ms,
            "tokens_prompt": row.tokens_prompt,
            "tokens_completion": row.tokens_completion,
            "tool_calls": row.tool_calls,
            "approvals": row.approvals,
            "failure_category": row.failure_category,
            "trace_completeness": row.trace_completeness,
            "transport": row.transport,
            "metadata": row.metadata,
        }
        for row in rows
    ]
    bundle = compute_metrics(normalised)
    now = datetime.now(timezone.utc)
    events = [
        TraceEvent(
            id=f"{row.runtime}:{row.transport}:{row.task_id}",
            timestamp=now,
            name="case.completed" if row.success else "case.failed",
            severity="success" if row.success else "error",
            runtime=row.runtime,
            task_id=row.task_id,
            transport=row.transport,
            latency_ms=row.latency_ms,
            message=f"{row.runtime} {row.task_id} via {row.transport}",
            payload={"failure_category": row.failure_category, "tool_calls": row.tool_calls},
        )
        for row in rows
    ]
    run = Run(
        status="passed" if all(row.success for row in rows) else "failed",
        command=command,
        suite=default_suite(),
        metrics=metric_summary_from_bundle(bundle),
        results=[result_to_case_result(row) for row in rows],
        events=events,
        artifacts=artifacts if artifacts is not None else discover_artifacts(run_root),
    )
    write_run(run, run_root)
    return run


def discover_artifacts(run_root: Path) -> list[ArtifactManifest]:
    if not run_root.exists():
        return []
    return [
        ArtifactManifest.from_path(path, run_root, description="Run artifact")
        for path in sorted(run_root.rglob("*"))
        if path.is_file() and path.name != "run.json"
    ]


def demo_run() -> Run:
    results = _demo_results()
    events = [
        TraceEvent(
            id="demo:launch",
            timestamp=datetime.now(timezone.utc),
            name="suite.loaded",
            severity="info",
            message="Loaded archived portability evidence from the October baseline run.",
        )
    ]
    return Run(
        id="demo_archived_portability",
        status="passed",
        command="usb-agents run --transport embedded --transport http --transport stdio",
        suite=default_suite(),
        metrics=MetricSummary(
            portability_success_rate=1.0,
            trace_completeness=1.0,
            latency_p50=35.4,
            latency_p95=420.1,
            approvals=27,
            token_total=576,
        ),
        results=results,
        events=events + _events_from_results(results),
        artifacts=_demo_artifacts(),
    )


def _demo_results() -> list[CaseResult]:
    rows: list[CaseResult] = []
    runtimes = ["openai_agents", "microsoft_agent_framework", "mistral_agents"]
    transports = ["embedded", "http", "stdio"]
    tasks = ["t1_repo_triage", "t2_calendar_merge", "t3_http_etl", "t4_code_patch"]
    for runtime_index, runtime in enumerate(runtimes):
        for transport_index, transport in enumerate(transports):
            for task_index, task_id in enumerate(tasks):
                latency = 1.0 + (runtime_index * 7.5) + (transport_index * 16.5)
                if task_id == "t4_code_patch":
                    latency += 360.0
                rows.append(
                    CaseResult(
                        runtime=runtime,
                        task_id=task_id,
                        transport=transport,
                        success=True,
                        latency_ms=latency,
                        approvals=[] if task_id == "t1_repo_triage" else [
                            ApprovalDecision(
                                tool=task_id,
                                granted=True,
                                reason=f"{runtime}:demo-approval",
                            )
                        ],
                        tool_calls=3 if task_id == "t1_repo_triage" else 1,
                        metadata={"fixture": "demo_archived_portability", "cell": task_index},
                    )
                )
    return rows


def _load_csv_results(path: Path) -> list[CaseResult]:
    if not path.exists():
        return []
    import csv

    rows: list[CaseResult] = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            approvals_raw = row.get("approvals") or "[]"
            approvals_data = json.loads(approvals_raw)
            metadata = json.loads(row.get("metadata") or "{}")
            rows.append(
                CaseResult(
                    runtime=row["runtime"],
                    task_id=row["task_id"],
                    transport=row.get("transport", "embedded"),
                    success=row["success"].lower() == "true",
                    latency_ms=float(row["latency_ms"]),
                    approvals=[
                        ApprovalDecision(
                            tool=item.get("tool", "unknown"),
                            granted=bool(item.get("granted", False)),
                            reason=str(item.get("reason", "")),
                        )
                        for item in approvals_data
                    ],
                    tool_calls=int(row["tool_calls"]),
                    failure_category=row.get("failure_category") or None,
                    failure_reason=row.get("failure_reason") or None,
                    trace_path=row.get("trace_path") or None,
                    metadata=metadata,
                )
            )
    return rows


def _events_from_results(results: list[CaseResult]) -> list[TraceEvent]:
    return [
        TraceEvent(
            id=f"demo:{index}",
            timestamp=datetime.now(timezone.utc),
            name="case.completed" if result.success else "case.failed",
            severity="success" if result.success else "error",
            runtime=result.runtime,
            task_id=result.task_id,
            transport=result.transport,
            latency_ms=result.latency_ms,
            message=f"{result.runtime} completed {result.task_id}",
        )
        for index, result in enumerate(results[:18])
    ]


def _demo_artifacts() -> list[ArtifactManifest]:
    root = Path("examples/mcp-server")
    paths = [root / "README.md"]
    return [
        ArtifactManifest.from_path(path, root, description="Archived demo artifact")
        for path in paths
        if path.exists()
    ]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
