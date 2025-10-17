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


def render_markdown(summary: Dict[str, List[TransportSnapshot]]) -> str:
    lines: List[str] = ["# Transport Metrics Dashboard", ""]
    for transport in sorted(summary.keys()):
        lines.append(f"## {transport}")
        lines.append("| Run | Latency p95 (ms) | Success | Stdio Wait p95 (ms) | Concurrency | Alerts |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for snapshot in summary[transport]:
            alerts = ",".join(snapshot.alerts) if snapshot.alerts else "-"
            lines.append(
                f"| {snapshot.run} | {snapshot.latency_p95:.1f} | {snapshot.success_rate:.2%} | {snapshot.wait_p95:.1f} | {snapshot.concurrency} | {alerts} |"
            )
        lines.append("")
    return "\n".join(lines)


def build_json(summary: Dict[str, List[TransportSnapshot]]) -> Dict[str, List[Dict[str, object]]]:
    return {
        transport: [
            {
                "run": snapshot.run,
                "latency_p95": snapshot.latency_p95,
                "success_rate": snapshot.success_rate,
                "wait_p95": snapshot.wait_p95,
                "concurrency": snapshot.concurrency,
                "alerts": snapshot.alerts,
            }
            for snapshot in snapshots
        ]
        for transport, snapshots in summary.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate transport metrics dashboard.")
    parser.add_argument("directory", type=Path, help="Directory containing transport_metrics.json files.")
    parser.add_argument("--output", type=Path, default=Path("dashboard.md"))
    parser.add_argument("--json-output", type=Path, default=Path("dashboard.json"))
    args = parser.parse_args()

    summary = build_summary(args.directory)
    markdown = render_markdown(summary)
    args.output.write_text(markdown + "\n", encoding="utf-8")
    json_summary = build_json(summary)
    args.json_output.write_text(json.dumps(json_summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
