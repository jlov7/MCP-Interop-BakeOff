"""Curated release demo generation."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence

from usb_agents.models import ArtifactManifest, Run
from usb_agents.run_executor import execute_run
from usb_agents.run_store import ensure_runs_root, write_run


def create_golden_demo(
    *,
    transports: Sequence[str] = ("embedded", "http", "stdio"),
    run_root: Path | None = None,
    timeout_seconds: float | None = 120,
) -> Run:
    target = run_root or ensure_runs_root() / "golden_release_demo"
    if target.exists():
        shutil.rmtree(target)
    run = execute_run(transports=transports, run_root=target, timeout_seconds=timeout_seconds)
    summary_path = target / "release-demo.md"
    summary_path.write_text(_summary(run), encoding="utf-8")
    run.artifacts.append(
        ArtifactManifest.from_path(
            summary_path,
            target,
            description="Curated release demo summary",
        )
    )
    write_run(run, target)
    return run


def _summary(run: Run) -> str:
    lines = [
        "# usb-agents Golden Release Demo",
        "",
        f"- Run: `{run.id}`",
        f"- Status: `{run.status}`",
        f"- Command: `{run.command}`",
        f"- Portability: `{run.metrics.portability_success_rate:.0%}`",
        f"- Trace coverage: `{run.metrics.trace_completeness:.0%}`",
        f"- Latency p95: `{run.metrics.latency_p95:.1f}ms`",
        f"- Approvals: `{run.metrics.approvals}`",
        "",
        "## Matrix",
        "",
        "| Runtime | Transport | Task | Status | Latency |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for result in run.results:
        status = "pass" if result.success else "fail"
        lines.append(
            "| "
            f"{result.runtime} | {result.transport} | {result.task_id} | {status} | "
            f"{result.latency_ms:.1f}ms |"
        )
    lines.extend(
        [
            "",
            "## Release Evidence",
            "",
            "This run is deterministic and local-first. It exercises the simulator runtimes across "
            "the configured MCP transports, records trace completeness, preserves approval "
            "decisions, and writes artifacts under the run root.",
        ]
    )
    return "\n".join(lines) + "\n"
