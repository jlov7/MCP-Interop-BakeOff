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

## Environment Configuration

- Copy `.env.template` to `.env` and fill in runtime credentials (`OPENAI_API_KEY`, `MS_AGENT_KEY`, `MISTRAL_API_KEY`). Use dedicated sandbox keys; the harness never persists them.
- Export the same variables in your shell (or `source .env`) before invoking runners. A quick smoke test is:
  ```bash
  uv run python - <<'PY'
  import os
  for name in ["OPENAI_API_KEY", "MS_AGENT_KEY", "MISTRAL_API_KEY"]:
      assert os.getenv(name), f"Missing {name}"
  print("Runtime secrets loaded")
  PY
  ```
- For HTTP transport runs, confirm `MCP_SERVER_HOST`/`PORT` are reachable from the client machine. When running locally you can rely on the defaults (`127.0.0.1:8000`); in remote CI ensure firewalls allow inbound traffic.
- The stdio transport uses `MCP_SERVER_STDIO_CMD`; update it if you relocate the repository or want to wrap the server with additional tooling.
- OTLP telemetry is optional. If you do not run a collector, either clear `OTEL_EXPORTER_OTLP_ENDPOINT` or point it at a local no-op receiver to avoid connection noise.

## HTTP Transport Tips

- `scripts/run_matrix.py` will auto-spawn the HTTP server when you pass `--transports http` (or the default transport is http). Explicitly set `--spawn-server=false` if you already have a server running elsewhere.
- To preload or debug the server manually, run `uv run python -m mcp_server.server --transport http --host 0.0.0.0 --port 8000` and watch for `INFO:     Application startup complete`.
- When the matrix script spawns the server, it polls `GET /health`. If you customise the policy path or ports, keep the environment variables in sync so health checks succeed.
- Granting approvals over HTTP still honours policy enforcement. Ensure agents can surface the approval prompt; missing `approval` payloads will lead to `400` errors returned by the server.

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

## Extending the Harness

- Create a new task under `tasks/` (`t5_new_task.yaml`) with an accompanying entry in `tasks/artifacts/` if you expect deterministic outputs. Follow the schema used in existing YAML files.
- Update runners in `usb_agents/runner_base.py` (and subclasses if behaviour diverges) to plan the new tool calls and capture approvals. Store any artefacts beneath `tasks/artifacts/` to keep CI write access deterministic.
- If the task exercises a new tool, wire the approval mode into `mcp-server/policy.yaml` and extend the tool implementation in `mcp_server/server.py`.
- Add regression coverage in `tests/`—prefer unit tests that simulate the new behaviour (e.g. metrics, telemetry, or approval flows) to keep runtime short.
- Finally, refresh baselines by running `scripts/run_matrix.py ... --update-baseline` once the new task is stable so dashboards and alerts include it.

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
5. Fail the pipeline on non-zero exit or inspect `alerts_summary.md` for triage.

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
