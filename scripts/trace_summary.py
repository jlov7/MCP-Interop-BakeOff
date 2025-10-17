"""Summarise trace coverage and missing spans from NDJSON exports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from usb_agents.runner_base import REQUIRED_SPANS


def load_trace(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def analyse_trace(path: Path) -> Dict[str, object]:
    spans = load_trace(path)
    names = {span["name"] for span in spans}
    missing = []
    for required in REQUIRED_SPANS:
        if required not in names:
            missing.append(required)
    tool_span_present = any(name.startswith("agent.tool_call") for name in names)
    approval_present = any(name.startswith("agent.approval") for name in names)
    completeness = 1.0
    expected = set(REQUIRED_SPANS)
    if tool_span_present:
        expected.add("agent.tool_call")
    if approval_present:
        expected.add("agent.approval")
    hits = 0
    for value in expected:
        if value in {"agent.tool_call", "agent.approval"}:
            present = any(name.startswith(value) for name in names)
        else:
            present = value in names
        if present:
            hits += 1
        else:
            missing.append(value)
    if expected:
        completeness = hits / len(expected)
    return {
        "file": str(path),
        "span_count": len(spans),
        "completeness": completeness,
        "missing": sorted(set(missing)),
    }


def render_summary(traces: Iterable[Dict[str, object]]) -> str:
    lines = []
    for trace in traces:
        lines.append(f"- {trace['file']}: spans={trace['span_count']} completeness={trace['completeness']:.2f}")
        missing = trace["missing"]
        if missing:
            lines.append(f"  missing: {', '.join(missing)}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarise trace coverage from NDJSON files")
    parser.add_argument("trace_dir", type=Path, help="Directory containing *.ndjson traces")
    args = parser.parse_args()

    results = []
    for path in sorted(args.trace_dir.glob("*.ndjson")):
        results.append(analyse_trace(path))
    print(render_summary(results))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
