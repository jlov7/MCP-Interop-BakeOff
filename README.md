# usb-agents – MCP Portability Testbed

usb-agents is a research-grade harness for stress-testing Multi-Channel Protocol (MCP) servers across multiple agent runtimes. It ships a shared MCP server, a reproducible task suite, and automation that runs those tasks through the OpenAI Agents SDK, Microsoft Agent Framework, and Mistral Agents API while collecting telemetry, approval behaviour, and detailed regressions.

## Why usb-agents?

| Challenge | How usb-agents Helps |
| --- | --- |
| **Portability risk** – tools behave differently per runtime. | Common MCP server exercised with identical tasks over HTTP, stdio, and embedded transports; success/latency/approval parity reported automatically. |
| **Safety drift** – approvals or guardrails might regress silently. | Approval gates, guardrails, and policy enforcement are tracked per task with alerts. |
| **Observability gaps** – hard to compare runs. | OTLP traces + metrics (tool latency, stdio pool telemetry) plus Markdown/JSON reports and dashboards for quick diffing. |
| **CI enforcement** – need to block merges on regressions. | `--fail-on-alert`, baselines, and `alerts_summary.md` give you machine-readable pass/fail signals. |

## Key Features

- **Shared MCP server** with stdio and HTTP transports, policy-driven tool approvals, and sandboxed `code_exec` support.
- **Deterministic task matrix** (triage, calendar merge, HTTP ETL, code patch) with golden artefacts for regression detection.
- **Runtime drivers** for OpenAI, Microsoft, and Mistral agents with guardrails, approval callbacks, and OTLP instrumentation.
- **Rich telemetry**: tool latency histograms, stdio pool wait/active/available metrics, plus OpenTelemetry traces.
- **Automated reporting**: `results.md`, `transport_metrics.json`, `alerts_summary.md`, and cumulative dashboards.
- **CI-first design**: thresholds, baselines, dashboards, and fail-fast modes to keep portability and safety on track.

## Architecture Overview

```
tasks/ ─▶ runners/ ─▶ scripts/run_matrix.py
   ▲          │          │
   │          │          ├─ alerts_summary.md
   │          │          ├─ results.md  + transport_metrics.json
   │          │          └─ metrics_archive/ (timestamped payloads)
   │          └─ usb_agents/ (telemetry, pooling, approvals)
   │
   └─ mcp-server/ (stdio + HTTP transports, policy, tooling)
```

1. **Task specs** describe desired behaviour (read-only triage, approval-required merges, HTTP ETL, sandboxed patch/testing).
2. **Runners** invoke the shared MCP server using embedded calls, HTTP, or pooled stdio workers – capturing approvals, tool calls, and telemetry.
3. **Automation** (`scripts/run_matrix.py`) executes runtime×task×transport combinations, stores metrics, raises alerts, refreshes dashboards, and optionally updates baselines.

## Getting Started

1. **Prerequisites**: Python 3.11+, [uv](https://github.com/astral-sh/uv), optional OTLP collector.
2. **Install & configure**:
   ```bash
   cp .env.template .env   # edit synthetic keys or OTLP endpoints if needed
   uv sync
   source .venv/bin/activate
   ```
3. **Verify tooling**: `pytest -q`
4. **Run a baseline matrix**:
   ```bash
   ./scripts/run_matrix.py \
     --transports embedded http stdio \
     --latency-threshold-ms 500 --success-threshold 0.75 \
     --stdio-wait-threshold-ms 100 --stdio-concurrency 2 --stdio-warmup 2 \
     --fail-on-alert \
     --archive-dir metrics_archive \
     --alerts-output alerts_summary.md
   ```
5. **Inspect outputs**:
   - `results.md`: transport comparison, alert sections, task excerpts.
   - `transport_metrics.json`: machine-readable metrics + alerts.
   - `alerts_summary.md`: single-page CI-friendly alert recap.
   - `metrics_archive/`: timestamped payloads + auto-refreshed `dashboard.md` / `dashboard.json`.
   - `traces/*.ndjson`: OTLP spans (useful for local inspection if no collector).

## CLI Highlights

| Flag | Purpose |
| --- | --- |
| `--transports embedded http stdio` | Explicit runtime transports (embedded-fast, HTTP-MCP, pooled stdio). |
| `--stdio-concurrency` / `--stdio-warmup` | Control stdio worker pool size and prewarming. |
| `--latency-threshold-ms`, `--success-threshold`, `--stdio-wait-threshold-ms` | Alert thresholds for latency, success rate, and stdio wait p95. |
| `--baseline baseline_metrics.json` | Compare against stored metrics (with optional `--update-baseline`). |
| `--fail-on-alert` | Return non-zero exit if any alert fires (latency, success, wait, baseline). |
| `--archive-dir metrics_archive` | Persist run artefacts; dashboards refresh automatically unless `--skip-dashboard` is set. |
| `--alerts-output alerts_summary.md` | Emit human-readable alert summary (ideal for CI artefact). |

Companion scripts:

- `scripts/report.py` – regenerate Markdown reports from `results.csv`/`transport_metrics.json` (supports the same thresholds/baseline options).
- `scripts/dashboard.py` – summarise archived runs into Markdown/JSON dashboards (accepts `--baseline` for delta columns).
- `scripts/trace_summary.py` – quick sanity check for OTLP spans.

## Telemetry & Alerts

- **Traces**: GenAI semantic spans for model calls, tool calls, approvals, and final outputs.
- **Metrics**:
  - `usb_agents.tool.latency` – per-tool request latency (transport + runtime attributes).
  - `usb_agents.stdio.wait_time` – wait-to-acquire pooled stdio worker.
  - `usb_agents.stdio.active_workers` / `available_workers` – pool utilisation snapshots.
- **Alerts**:
  - Latency (p95), success rate, stdio wait-time, and baseline deltas.
  - Alerts surface in `results.md`, `transport_metrics.json`, `alerts_summary.md`, and dashboards.
  - Use `--fail-on-alert` to gate CI, `--stdio-wait-threshold-ms` for pooled latencies.

## CI Integration Checklist

1. Run `scripts/run_matrix.py` with thresholds, baselines, alerts output, and archive dir (see quickstart example).
2. Regenerate dashboards and reports (run `dashboard.py` / `report.py` if you skip the built-in archive refresh).
3. Upload artefacts: `results.csv`, `results.md`, `transport_metrics.json`, `alerts_summary.md`, `metrics_archive/*` (dashboards + historical metrics), and optionally `traces/*.ndjson`.
4. Optionally push `metrics_archive/dashboard.json` to your analytics store or surface `alerts_summary.md` in pull requests for quick triage.
4. Fail the pipeline on non-zero exit or inspect `alerts_summary.md` for triage.

## Repository Structure (abridged)

```
README.md
.env.template
mcp-server/          # Shared MCP server + policy
runners/             # Runtime adapters (OpenAI, Microsoft, Mistral)
tasks/               # Task definitions, golden artefacts, code fixtures
usb_agents/          # Core logic (telemetry, pooling, task helpers)
scripts/             # run_matrix.py, report.py, dashboard.py, trace_summary.py
eval/                # Metrics aggregation, taxonomy, agent GPA helpers
tests/               # Unit, integration, pooling, smoke tests
metrics_archive/     # (Generated) historical transport_metrics.json
alerts_summary.md    # (Generated) latest alert recap
```

## Contributing & Support

Issues and pull requests are welcome. Please run `pytest -q` and `scripts/run_matrix.py --transports embedded --fail-on-alert` before submitting changes. For questions or feature requests, open a GitHub issue describing the runtime(s) and transport(s) you care about.

---

Harness the same MCP tasks across three Agent stacks, track portability and safety automatically, and wire the outputs straight into your CI/CD pipeline.
