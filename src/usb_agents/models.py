"""Shared typed contracts for benchmark runs, API payloads, and UI data."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

TransportName = Literal["embedded", "http", "stdio"]
RunStatus = Literal["queued", "running", "passed", "failed"]
Severity = Literal["info", "success", "warning", "error"]


def new_run_id() -> str:
    return f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"


class TaskCase(BaseModel):
    id: str
    name: str
    description: str
    approval: str
    mode: str
    expected_artifact: str | None = None


class BenchmarkSuite(BaseModel):
    id: str = "default"
    name: str = "USB Agents Portability Suite"
    description: str = "Deterministic MCP portability checks across runtimes and transports."
    cases: list[TaskCase] = Field(default_factory=list)


class RuntimeAdapter(BaseModel):
    id: str
    name: str
    model: str
    status: Literal["available", "simulated", "unconfigured", "disabled"] = "simulated"
    mode: Literal["simulator", "sdk"] = "simulator"
    credential_env: str | None = None


class TransportAdapter(BaseModel):
    id: TransportName
    name: str
    protocol: str
    status: Literal["available", "experimental"] = "available"


class ApprovalDecision(BaseModel):
    tool: str
    granted: bool
    reason: str


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    transport: str
    approved: bool | None = None


class TraceEvent(BaseModel):
    id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    name: str
    severity: Severity = "info"
    runtime: str | None = None
    task_id: str | None = None
    transport: str | None = None
    latency_ms: float | None = None
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class ArtifactManifest(BaseModel):
    path: str
    kind: Literal["json", "csv", "markdown", "trace", "text"] = "text"
    bytes: int = 0
    description: str = ""

    @classmethod
    def from_path(cls, path: Path, root: Path, description: str = "") -> "ArtifactManifest":
        suffix = path.suffix.lower()
        kind: Literal["json", "csv", "markdown", "trace", "text"] = "text"
        if suffix == ".json":
            kind = "json"
        elif suffix == ".csv":
            kind = "csv"
        elif suffix == ".md":
            kind = "markdown"
        elif suffix == ".ndjson":
            kind = "trace"
        return cls(
            path=str(path.relative_to(root)),
            kind=kind,
            bytes=path.stat().st_size,
            description=description,
        )


class MetricSummary(BaseModel):
    portability_success_rate: float = 0.0
    trace_completeness: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    approvals: int = 0
    token_total: int = 0
    failure_histogram: dict[str, int] = Field(default_factory=dict)


class CaseResult(BaseModel):
    runtime: str
    task_id: str
    transport: str
    success: bool
    latency_ms: float
    approvals: list[ApprovalDecision] = Field(default_factory=list)
    tool_calls: int = 0
    failure_category: str | None = None
    failure_reason: str | None = None
    trace_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Run(BaseModel):
    id: str = Field(default_factory=new_run_id)
    status: RunStatus = "queued"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    command: str = "usb-agents run --transports embedded http stdio"
    suite: BenchmarkSuite = Field(default_factory=BenchmarkSuite)
    metrics: MetricSummary = Field(default_factory=MetricSummary)
    results: list[CaseResult] = Field(default_factory=list)
    events: list[TraceEvent] = Field(default_factory=list)
    artifacts: list[ArtifactManifest] = Field(default_factory=list)

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)
