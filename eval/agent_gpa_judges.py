"""Optional Agent GPA judge scoring on recorded traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def score_trace(trace_path: Path) -> Dict[str, float]:
    spans = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
    tool_spans = [span for span in spans if span["name"].startswith("agent.tool_call")]
    approvals = [span for span in spans if span["name"].startswith("agent.approval")]
    logic = 1.0 if tool_spans else 0.5
    execution = 1.0 if approvals else 0.7
    adherence = 1.0 if spans and spans[-1]["name"] == "agent.final_output" else 0.6
    return {
        "logical_consistency": logic,
        "execution_efficiency": execution,
        "plan_adherence": adherence,
    }


def score_directory(traces_dir: Path) -> List[Dict[str, float]]:
    results = []
    for path in sorted(traces_dir.glob("*.ndjson")):
        scores = score_trace(path)
        scores["trace"] = path.name
        results.append(scores)
    return results
