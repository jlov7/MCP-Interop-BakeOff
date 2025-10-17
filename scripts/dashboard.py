"""Generate a dashboard summarising transport metrics over multiple runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class TransportSnapshot:
    run: str
    latency_p95: float
    success_rate: float
    wait_p95: float
    concurrency: int
    alerts: List[str]


def load_metrics(path: Path) -> Dict[str, TransportSnapshot]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    transports = data.get("transports", data)
    if not isinstance(transports, dict):
        return {}
    alerts = data.get("alerts", {})
    snapshots: Dict[str, TransportSnapshot] = {}
    for name, metrics in transports.items():
        if not isinstance(metrics, dict):
            continue
        alert_tags: List[str] = []
        for category, entries in alerts.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if entry.get("transport") == name:
                    alert_tags.append(category)
        stdio_pool = metrics.get("stdio_pool", {})
        wait_p95 = stdio_pool.get("wait_ms", {}).get("p95", 0.0)
        concurrency = stdio_pool.get("concurrency", 0)
        snapshots[name] = TransportSnapshot(
            run=path.stem,
            latency_p95=metrics.get("latency_p95", 0.0),
            success_rate=metrics.get("portability_success_rate", 0.0),
            wait_p95=wait_p95,
            concurrency=concurrency,
            alerts=sorted(set(alert_tags)),
        )
    return snapshots


def build_summary(directory: Path) -> Dict[str, List[TransportSnapshot]]:
    summary: Dict[str, List[TransportSnapshot]] = {}
    for file_path in sorted(directory.glob("*.json")):
        snapshots = load_metrics(file_path)
        for transport, snapshot in snapshots.items():
            summary.setdefault(transport, []).append(snapshot)
    return summary


def render_markdown(summary: Dict[str, List[TransportSnapshot]], baseline: Optional[Dict[str, dict]]) -> str:
    lines: List[str] = ["# Transport Metrics Dashboard", ""]
    for transport in sorted(summary.keys()):
        lines.append(f"## {transport}")
        has_baseline = bool(baseline and transport in baseline)
        header = ["Run", "Latency p95 (ms)", "Success", "Stdio Wait p95 (ms)", "Concurrency", "Alerts"]
        if has_baseline:
            header.extend(["Δ Latency p95", "Δ Success"])
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for snapshot in summary[transport]:
            alerts = ",".join(snapshot.alerts) if snapshot.alerts else "-"
            row = [
                snapshot.run,
                f"{snapshot.latency_p95:.1f}",
                f"{snapshot.success_rate:.2%}",
                f"{snapshot.wait_p95:.1f}",
                str(snapshot.concurrency),
                alerts,
            ]
            if has_baseline:
                base = baseline[transport]
                base_latency = base.get("latency_p95")
                base_success = base.get("portability_success_rate")
                delta_latency = (
                    snapshot.latency_p95 - base_latency if base_latency is not None else None
                )
                delta_success = (
                    snapshot.success_rate - base_success if base_success is not None else None
                )
                row.append(f"{delta_latency:+.1f}" if delta_latency is not None else "-")
                row.append(f"{delta_success:+.2%}" if delta_success is not None else "-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def build_json(
    summary: Dict[str, List[TransportSnapshot]], baseline: Optional[Dict[str, dict]]
) -> Dict[str, List[Dict[str, object]]]:
    return {
        transport: [
            {
                "run": snapshot.run,
                "latency_p95": snapshot.latency_p95,
                "success_rate": snapshot.success_rate,
                "wait_p95": snapshot.wait_p95,
                "concurrency": snapshot.concurrency,
                "delta_latency_p95": (
                    snapshot.latency_p95 - baseline.get(transport, {}).get("latency_p95")
                    if baseline and transport in baseline and baseline[transport].get("latency_p95") is not None
                    else None
                ),
                "delta_success": (
                    snapshot.success_rate - baseline.get(transport, {}).get("portability_success_rate")
                    if baseline and transport in baseline and baseline[transport].get("portability_success_rate") is not None
                    else None
                ),
                "alerts": snapshot.alerts,
            }
            for snapshot in snapshots
        ]
        for transport, snapshots in summary.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate transport metrics dashboard.")
    parser.add_argument("directory", type=Path, help="Directory containing transport_metrics.json files.")
    parser.add_argument("--baseline", type=Path, help="Optional baseline transport_metrics.json for deltas.")
    parser.add_argument("--output", type=Path, default=Path("dashboard.md"))
    parser.add_argument("--json-output", type=Path, default=Path("dashboard.json"))
    args = parser.parse_args()

    summary = build_summary(args.directory)
    baseline = None
    if args.baseline and args.baseline.exists():
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        baseline = baseline.get("transports", baseline)
    markdown = render_markdown(summary, baseline)
    args.output.write_text(markdown + "\n", encoding="utf-8")
    json_summary = build_json(summary, baseline)
    args.json_output.write_text(json.dumps(json_summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
