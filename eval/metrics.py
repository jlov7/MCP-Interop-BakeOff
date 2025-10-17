"""Aggregate evaluation metrics for usb-agents runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List


@dataclass
class MetricsBundle:
    portability_success_rate: float
    trace_completeness: float
    latency_p50: float
    latency_p95: float
    approvals: int
    token_total: int
    failure_histogram: Dict[str, int]
    tool_latency_p50: Dict[str, float]
    tool_latency_p95: Dict[str, float]
    tool_latency_count: Dict[str, int]


def load_results(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def compute_portability(results: Iterable[Dict]) -> float:
    relevant = [row for row in results if row["task_id"] in {"t1_repo_triage", "t2_calendar_merge", "t3_http_etl"}]
    if not relevant:
        return 0.0
    success = sum(1 for row in relevant if row["success"])
    return success / len(relevant)


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def compute_metrics(results: List[Dict]) -> MetricsBundle:
    portability = compute_portability(results)
    latencies = [row["latency_ms"] for row in results]
    approvals = 0
    tool_latencies: Dict[str, List[float]] = {}
    for row in results:
        raw = row.get("approvals", [])
        if isinstance(raw, str):
            approvals_list = json.loads(raw or "[]")
        else:
            approvals_list = raw or []
        approvals += len(approvals_list)
        metadata = row.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata or "{}")
        durations = metadata.get("tool_durations_ms", {})
        if isinstance(durations, dict):
            for tool, values in durations.items():
                if isinstance(values, list):
                    tool_latencies.setdefault(tool, []).extend(values)
    tokens = sum(row.get("tokens_prompt", 0) + row.get("tokens_completion", 0) for row in results)
    failure_histogram: Dict[str, int] = {}
    trace_scores = [row.get("trace_completeness", 0.0) for row in results if row.get("trace_completeness") is not None]
    for row in results:
        category = row.get("failure_category")
        if category:
            failure_histogram[category] = failure_histogram.get(category, 0) + 1

    tool_latency_p50: Dict[str, float] = {}
    tool_latency_p95: Dict[str, float] = {}
    tool_latency_count: Dict[str, int] = {}
    for tool, values in tool_latencies.items():
        if not values:
            continue
        tool_latency_p50[tool] = median(values)
        tool_latency_p95[tool] = percentile(values, 0.95)
        tool_latency_count[tool] = len(values)

    return MetricsBundle(
        portability_success_rate=portability,
        trace_completeness=sum(trace_scores) / len(trace_scores) if trace_scores else 0.0,
        latency_p50=median(latencies) if latencies else 0.0,
        latency_p95=percentile(latencies, 0.95) if latencies else 0.0,
        approvals=approvals,
        token_total=tokens,
        failure_histogram=failure_histogram,
        tool_latency_p50=tool_latency_p50,
        tool_latency_p95=tool_latency_p95,
        tool_latency_count=tool_latency_count,
    )


def compute_metrics_by_transport(results: List[Dict]) -> Dict[str, MetricsBundle]:
    grouped: Dict[str, List[Dict]] = {}
    for row in results:
        transport = row.get("transport", "unknown")
        grouped.setdefault(transport, []).append(row)
    return {transport: compute_metrics(rows) for transport, rows in grouped.items()}


def bundle_to_dict(bundle: MetricsBundle) -> Dict[str, object]:
    base = asdict(bundle)
    return base


def compute_latency_alerts(
    transport_metrics: Dict[str, MetricsBundle], threshold_ms: float
) -> List[Dict[str, float | str]]:
    alerts: List[Dict[str, float | str]] = []
    for transport, bundle in transport_metrics.items():
        if bundle.latency_p95 > threshold_ms:
            alerts.append(
                {
                    "transport": transport,
                    "latency_p95": bundle.latency_p95,
                    "threshold_ms": threshold_ms,
                }
            )
    return alerts


def compute_success_alerts(results: List[Dict], threshold: float) -> List[Dict[str, float | str]]:
    totals: Dict[str, Dict[str, float]] = {}
    for row in results:
        transport = row.get("transport", "unknown")
        totals.setdefault(transport, {"success": 0.0, "count": 0.0})
        totals[transport]["count"] += 1.0
        if row.get("success"):
            totals[transport]["success"] += 1.0
    alerts: List[Dict[str, float | str]] = []
    for transport, data in totals.items():
        if data["count"] == 0:
            continue
        rate = data["success"] / data["count"]
        if rate < threshold:
            alerts.append(
                {
                    "transport": transport,
                    "success_rate": rate,
                    "threshold": threshold,
                }
            )
    return alerts


def compute_stdio_wait_alerts(
    transports: Dict[str, Dict[str, Any]], threshold_ms: float
) -> List[Dict[str, float | str]]:
    alerts: List[Dict[str, float | str]] = []
    for transport, metrics in transports.items():
        stdio_pool = metrics.get("stdio_pool")
        if not stdio_pool:
            continue
        wait_stats = stdio_pool.get("wait_ms", {})
        p95 = wait_stats.get("p95")
        if p95 is None:
            continue
        if p95 > threshold_ms:
            alerts.append(
                {
                    "transport": transport,
                    "wait_p95": p95,
                    "threshold_ms": threshold_ms,
                }
            )
    return alerts


def compare_to_baseline(
    current: Dict[str, MetricsBundle],
    baseline: Dict[str, Dict[str, float]],
    max_latency_delta: float,
    max_success_delta: float,
) -> List[Dict[str, float | str]]:
    alerts: List[Dict[str, float | str]] = []
    for transport, bundle in current.items():
        base_metrics = baseline.get(transport, {})
        base_latency = base_metrics.get("latency_p95")
        base_success = base_metrics.get("portability_success_rate")
        if base_latency is not None and bundle.latency_p95 - base_latency > max_latency_delta:
            alerts.append(
                {
                    "transport": transport,
                    "latency_p95_delta": bundle.latency_p95 - base_latency,
                    "max_latency_delta": max_latency_delta,
                }
            )
        if base_success is not None and base_success - bundle.portability_success_rate > max_success_delta:
            alerts.append(
                {
                    "transport": transport,
                    "success_rate_delta": base_success - bundle.portability_success_rate,
                    "max_success_delta": max_success_delta,
                }
            )
    return alerts
