"""Generate Markdown summaries for usb-agents evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.metrics import (
    MetricsBundle,
    compare_to_baseline,
    compute_latency_alerts,
    compute_metrics,
    compute_metrics_by_transport,
    compute_success_alerts,
    compute_stdio_wait_alerts,
)


def render_markdown(
    results: List[Dict],
    metrics: MetricsBundle,
    latency_alerts: Optional[List[Dict[str, object]]] = None,
    success_alerts: Optional[List[Dict[str, object]]] = None,
    baseline_alerts: Optional[List[Dict[str, object]]] = None,
    transport_meta: Optional[Dict[str, Dict[str, object]]] = None,
    stdio_wait_alerts: Optional[List[Dict[str, object]]] = None,
    baseline_transports: Optional[Dict[str, Dict[str, object]]] = None,
) -> str:
    lines = ["# Evaluation Results", ""]
    lines.append(f"- Portability success rate: {metrics.portability_success_rate:.2%}")
    lines.append(f"- Trace completeness (avg): {metrics.trace_completeness:.2%}")
    lines.append(f"- Latency p50: {metrics.latency_p50:.1f} ms")
    lines.append(f"- Latency p95: {metrics.latency_p95:.1f} ms")
    lines.append(f"- Total tokens: {metrics.token_total}")
    lines.append(f"- Approval hits: {metrics.approvals}")
    if metrics.tool_latency_p50:
        lines.append("")
        lines.append("## Tool Latency (ms)")
        lines.append("| Tool | Calls | p50 | p95 |")
        lines.append("| --- | --- | --- | --- |")
        for tool in sorted(metrics.tool_latency_p50.keys()):
            count = metrics.tool_latency_count.get(tool, 0)
            p50 = metrics.tool_latency_p50.get(tool, 0.0)
            p95 = metrics.tool_latency_p95.get(tool, 0.0)
            lines.append(f"| {tool} | {count} | {p50:.1f} | {p95:.1f} |")
    lines.append("")
    lines.append("## Runtime Outcomes")
    runtime_groups: Dict[str, List[Dict]] = {}
    for row in results:
        runtime_groups.setdefault(row["runtime"], []).append(row)
    lines.append("| Runtime | Successes | Failures | Median Latency (ms) | Transport |")
    lines.append("| --- | --- | --- | --- | --- |")
    for runtime, rows in sorted(runtime_groups.items()):
        successes = sum(1 for row in rows if row["success"])
        failures = len(rows) - successes
        median_latency = compute_metrics(rows).latency_p50
        transports = {row["transport"] for row in rows}
        lines.append(
            f"| {runtime} | {successes} | {failures} | {median_latency:.1f} | {','.join(sorted(transports))} |"
        )
    if metrics.failure_histogram:
        lines.append("")
        lines.append("## Failures")
        for category, count in metrics.failure_histogram.items():
            lines.append(f"- {category}: {count}")
    transport_metrics = compute_metrics_by_transport(results)
    has_stdio = bool(transport_meta) and any(
        isinstance((transport_meta or {}).get(t, {}).get("stdio_pool"), dict) for t in transport_metrics
    )
    has_baseline = bool(baseline_transports) and any(
        isinstance((baseline_transports or {}).get(t), dict) for t in transport_metrics
    )

    if transport_metrics:
        lines.append("")
        lines.append("## Transport Comparison")
        header_cells = ["Transport", "Portability", "Latency p50 (ms)", "Latency p95 (ms)"]
        if has_stdio:
            header_cells.extend(["Stdio Wait p95 (ms)", "Concurrency"])
        if has_baseline:
            header_cells.extend(["Δ Latency p95 (ms)", "Δ Success"])
        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("| " + " | ".join(["---"] * len(header_cells)) + " |")
        for transport in sorted(transport_metrics.keys()):
            tm = transport_metrics[transport]
            row_cells = [
                transport,
                f"{tm.portability_success_rate:.2%}",
                f"{tm.latency_p50:.1f}",
                f"{tm.latency_p95:.1f}",
            ]
            if has_stdio:
                pool = (transport_meta or {}).get(transport, {}).get("stdio_pool") if transport_meta else None
                if isinstance(pool, dict):
                    wait_p95 = pool.get("wait_ms", {}).get("p95", 0.0)
                    concurrency = pool.get("concurrency", 0)
                    row_cells.extend([f"{wait_p95:.1f}", str(concurrency)])
                else:
                    row_cells.extend(["-", "-"])
            if has_baseline:
                base_metrics = (baseline_transports or {}).get(transport) if baseline_transports else None
                delta_latency = delta_success = None
                if isinstance(base_metrics, dict):
                    base_latency = base_metrics.get("latency_p95")
                    base_success = base_metrics.get("portability_success_rate")
                    if base_latency is not None:
                        delta_latency = tm.latency_p95 - base_latency
                    if base_success is not None:
                        delta_success = tm.portability_success_rate - base_success
                row_cells.append(f"{delta_latency:+.1f}" if delta_latency is not None else "-")
                row_cells.append(f"{delta_success:+.2%}" if delta_success is not None else "-")
            lines.append("| " + " | ".join(row_cells) + " |")
        if {"embedded", "http"}.issubset(transport_metrics.keys()):
            http_metrics = transport_metrics["http"]
            embedded_metrics = transport_metrics["embedded"]
            delta_p50 = http_metrics.latency_p50 - embedded_metrics.latency_p50
            delta_p95 = http_metrics.latency_p95 - embedded_metrics.latency_p95
            lines.append("")
            lines.append(
                f"HTTP vs embedded Δ latency: p50 {delta_p50:+.1f} ms, p95 {delta_p95:+.1f} ms"
            )
    if latency_alerts:
        lines.append("")
        lines.append("## Latency Alerts")
        for alert in latency_alerts:
            lines.append(
                f"- {alert['transport']}: p95={alert['latency_p95']:.1f} ms exceeded threshold {alert['threshold_ms']:.1f} ms"
            )
    if success_alerts:
        lines.append("")
        lines.append("## Success Alerts")
        for alert in success_alerts:
            lines.append(
                f"- {alert['transport']}: success rate={alert['success_rate']:.2%} below threshold {alert['threshold']:.0%}"
            )
    if stdio_wait_alerts:
        lines.append("")
        lines.append("## Stdio Wait Alerts")
        for alert in stdio_wait_alerts:
            lines.append(
                f"- {alert['transport']}: wait p95={alert['wait_p95']:.1f} ms exceeded threshold {alert['threshold_ms']:.1f} ms"
            )
    if baseline_alerts:
        lines.append("")
        lines.append("## Baseline Alerts")
        for alert in baseline_alerts:
            if "latency_p95_delta" in alert:
                lines.append(
                    f"- {alert['transport']}: p95 delta={alert['latency_p95_delta']:.1f} ms (limit {alert['max_latency_delta']:.1f} ms)"
                )
            if "success_rate_delta" in alert:
                lines.append(
                    f"- {alert['transport']}: success delta={alert['success_rate_delta']:.2%} (limit {alert['max_success_delta']:.2%})"
                )
    lines.append("")
    lines.append("## Sample Rows")
    for row in results[:5]:
        lines.append(f"- {row['runtime']} {row['task_id']} success={row['success']} latency={row['latency_ms']:.1f}ms")
    return "\n".join(lines) + "\n"


def parse_csv(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [row for row in reader]


def main() -> int:
    parser = argparse.ArgumentParser(description="Render results markdown from CSV.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results.md"))
    parser.add_argument(
        "--latency-threshold-ms",
        type=float,
        default=float(os.getenv("USB_AGENTS_LATENCY_THRESHOLD_MS", 500.0)),
        help="Latency threshold (ms) for p95 alerts.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=float(os.getenv("USB_AGENTS_SUCCESS_THRESHOLD", 0.75)),
        help="Success rate threshold for alerts.",
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
        "--transport-json",
        type=Path,
        help="Optional transport_metrics.json to supply transport metadata (e.g., stdio pool stats).",
    )
    parser.add_argument(
        "--stdio-wait-threshold-ms",
        type=float,
        default=float(os.getenv("USB_AGENTS_STDIO_WAIT_THRESHOLD_MS", 100.0)),
        help="Threshold for stdio wait-time alerts (ms).",
    )
    args = parser.parse_args()

    rows = parse_csv(args.csv)
    # Convert JSON columns back to python objects for metrics computation.
    normalised = []
    for row in rows:
        normalised.append({
            "runtime": row["runtime"],
            "task_id": row["task_id"],
            "policy_mode": row["policy_mode"],
            "success": row["success"].lower() == "true",
            "latency_ms": float(row["latency_ms"]),
            "tokens_prompt": int(row["tokens_prompt"]),
            "tokens_completion": int(row["tokens_completion"]),
            "tool_calls": int(row["tool_calls"]),
            "approvals": row["approvals"],
            "failure_category": row.get("failure_category") or None,
            "trace_completeness": float(row.get("trace_completeness", 0) or 0),
            "transport": row.get("transport", "http"),
            "metadata": row.get("metadata", "{}"),
        })
    metrics = compute_metrics(normalised)
    transport_metrics = compute_metrics_by_transport(normalised)
    latency_alerts = compute_latency_alerts(transport_metrics, args.latency_threshold_ms)
    success_alerts = compute_success_alerts(normalised, args.success_threshold)
    transport_meta = None
    if args.transport_json and args.transport_json.exists():
        payload = json.loads(args.transport_json.read_text(encoding="utf-8"))
        transport_meta = payload.get("transports", payload)
    stdio_wait_alerts = compute_stdio_wait_alerts(
        transport_meta or transport_metrics,
        args.stdio_wait_threshold_ms,
        baseline_transports if baseline_transports else None,
    )
    baseline_alerts: List[Dict[str, object]] = []
    baseline_transports: Dict[str, Dict[str, object]] = {}
    if args.baseline and args.baseline.exists():
        baseline_data = json.loads(args.baseline.read_text(encoding="utf-8"))
        baseline_transports = baseline_data.get("transports", baseline_data)
        baseline_alerts = compare_to_baseline(
            transport_metrics,
            baseline_transports,
            args.max_latency_delta,
            args.max_success_delta,
        )
    markdown = render_markdown(
        normalised,
        metrics,
        latency_alerts,
        success_alerts,
        baseline_alerts,
        transport_meta,
        stdio_wait_alerts,
        baseline_transports or None,
    )
    args.output.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
