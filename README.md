# usb-agents

Research harness for evaluating Multi-Channel Protocol (MCP) server portability across multiple agent runtimes. The project provisions a common MCP server and executes deterministic task specs through the OpenAI Agents SDK, Microsoft Agent Framework (preview), and the Mistral Agents API while collecting approval, latency, and OpenTelemetry trace fidelity metrics.

## Repository Layout

- `mcp-server/` – MCP server implementation, policy, and docs.
- `runners/` – Client runners for OpenAI, Microsoft, and Mistral agent runtimes.
- `tasks/` – Deterministic task definitions and golden artifacts.
- `eval/` – Metrics processing, taxonomy, and optional Agent GPA judges.
- `scripts/` – Automation for running the evaluation matrix and reporting results.
- `tests/` – Unit and smoke tests.
- `traces/` – Collected OpenTelemetry exports (gitignored).

## Quickstart

1. Copy environment defaults: `cp .env.template .env` and adjust API keys if you plan to proxy real services (synthetic strings are fine for offline runs).
2. Install dependencies with [uv](https://github.com/astral-sh/uv):
   ```bash
   uv sync
   source .venv/bin/activate
   ```
3. Run the safety/interop test suite: `pytest -q`.
4. Execute the evaluation matrix (embedded transport): `./scripts/run_matrix.py`.
   - To exercise the real HTTP transport, run `./scripts/run_matrix.py --transport http` (spawns the MCP server automatically).
   - To compare transports in one sweep, run `./scripts/run_matrix.py --compare-transports` or pass an explicit list (`--transports embedded http stdio`) to generate combined CSV/markdown with deltas plus a `transport_metrics.json` summary (override with `--transport-json`).
   - To archive runs and generate dashboards, add `--archive-dir metrics_archive` (optionally `--dashboard-output` / `--dashboard-json`).
5. Review outputs: `results.csv`, `results.md`, `transport_metrics.json`, `alerts_summary.md`, and `traces/*.ndjson`.

## Environment Variables

`.env.template` documents all configuration knobs:

- `OPENAI_API_KEY`, `MS_AGENT_KEY`, `MISTRAL_API_KEY` – synthetic credentials for local adapters.
- `OPENAI_AGENT_MODEL`, `MS_AGENT_MODEL`, `MISTRAL_AGENT_MODEL` – default model identifiers surfaced in traces.
- `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_HEADERS` – optional OTLP target; falls back to console exporter when unset.
- `MCP_SERVER_HTTP_URL`, `MCP_SERVER_HOST`, `MCP_SERVER_PORT`, `MCP_SERVER_STDIO_CMD` – connector hints for real deployments (not required for local runs). `MCP_SERVER_STDIO_CMD` overrides the stdio launch command.
- `USB_AGENTS_LATENCY_THRESHOLD_MS` – alert threshold for p95 latency per transport (default 500 ms).
- `USB_AGENTS_SUCCESS_THRESHOLD` – minimum success rate per transport before raising an alert (default 0.75).
- `USB_AGENTS_MAX_LATENCY_DELTA`, `USB_AGENTS_MAX_SUCCESS_DELTA` – maximum allowed regressions when comparing against a stored baseline.
- `USB_AGENTS_STDIO_CONCURRENCY`, `USB_AGENTS_STDIO_WARMUP` – tune stdio worker pool size and prewarmed workers for pooled execution.
- `USB_AGENTS_STDIO_WAIT_THRESHOLD_MS` – alert threshold for stdio worker wait-time p95 (default 100 ms).
- `RESULTS_DIR` – output directory used by orchestration scripts.

## Task Matrix

| ID | Goal | Approval | Artifact |
| --- | --- | --- | --- |
| `t1_repo_triage` | Summarise core repo files and emit JSON aligned with `tasks/artifacts/t1_repo_triage_expected.json`. | `never` | JSON summary |
| `t2_calendar_merge` | Merge synthetic calendar entries with write approval. | `by_tool` | none (merge plan only) |
| `t3_http_etl` | Fetch allowlisted JSON and persist `tasks/artifacts/http_etl.csv`. | `by_tool` | CSV |
| `t4_code_patch` | Add `multiply` helper and run sandboxed tests via `code_exec`. | `by_tool` | none |

Golden artefacts live under `tasks/artifacts/` and are asserted in `tests/`.

## Approval Policy

`mcp-server/policy.yaml` configures tool-level gating:

| Tool | Policy | Notes |
| --- | --- | --- |
| `fs_read` | `never` | Read-only operations do not require oversight. |
| `http_fetch` | `by_tool` | Approval recorded to demonstrate guardrail parity. |
| `calendar_merge` | `by_tool` | Must log override reason before merges. |
| `code_exec` | `by_tool` | Sandbox commands limited to `py_compile` and `pytest`. |

Runners expose human-in-the-loop callbacks by overriding `BaseRunner.request_approval` with runtime-specific annotations (`openai-hitl:*`, `ms-hitl:*`, `mistral-hitl:*`).

## Transports

- `embedded` (default): Runners share an in-process MCP server for fast feedback loops.
- `http`: `scripts/run_matrix.py --transport http` launches the FastAPI/uvicorn server via `mcp_server.server` and routes tool calls over REST (`/tools/<name>` plus `/health`).
- Individual runner CLIs accept `--transport` and `--http-url` flags to target external MCP deployments.
- Comparison mode: `./scripts/run_matrix.py --compare-transports` executes the entire matrix twice (embedded + HTTP), aggregates per-transport metrics, and surfaces latency deltas in `results.md`. Use `--transports embedded http stdio` to customise the set or include stdio comparisons (stdio columns include wait-time p95 and concurrency; baseline deltas appear when `--baseline` is supplied).
- `stdio`: A dedicated MCP worker pool is spawned and reused per runner; control pool size with `--stdio-concurrency` (or `USB_AGENTS_STDIO_CONCURRENCY`) and prewarm workers via `--stdio-warmup`.
- Alerting controls: `--latency-threshold-ms`, `--success-threshold`, `--max-latency-delta`, `--max-success-delta`, and `--fail-on-alert` expose knobs for regression gating in CI (baseline comparisons via `--baseline`).

## Telemetry and Reporting

- Traces: All runs export OpenTelemetry spans (GenAI semantic conventions) to `traces/*.ndjson` via an in-memory exporter. Configure `OTEL_EXPORTER_OTLP_ENDPOINT` to forward to an external collector.
- Metrics: `eval/metrics.py` computes portability %, latency p50/p95, per-tool latency distributions, per-transport summaries, token totals, approval counts, stdio pool telemetry, and a failure histogram using `eval/taxonomy.json` labels. Per-transport metrics (plus all alert categories) are written to `transport_metrics.json` for downstream dashboards.
- Reporting: `scripts/report.py results.csv` renders the Markdown summary consumed by `run_matrix.py`.
- Dashboard: `python scripts/dashboard.py transport-metrics-dir` produces cumulative dashboards (`dashboard.md`/`dashboard.json`) from archived `transport_metrics.json` runs.
- Optional: `eval/agent_gpa_judges.py` scores traces with lightweight Agent GPA heuristics (Logical Consistency, Execution Efficiency, Plan Adherence).
- Trace QA: `scripts/trace_summary.py traces/` audits NDJSON exports for missing span categories before shipping to external collectors.
- OTLP Metrics: When `OTEL_EXPORTER_OTLP_ENDPOINT` is set, per-tool latency histograms (`usb_agents.tool.latency`) are exported with attributes for tool name, runtime, and transport. Use this to observe runtime regressions in your collector.
- Latency Alerts: Set `USB_AGENTS_LATENCY_THRESHOLD_MS` (or CLI flag `--latency-threshold-ms`) to surface transports whose p95 latency exceeds the threshold; alerts are rendered in `results.md` and captured in `transport_metrics.json`.
- Success Alerts: Configure `USB_AGENTS_SUCCESS_THRESHOLD` (or `--success-threshold`) to flag transports whose success rate drops below the target; optionally promote alerts to hard failures with `--fail-on-alert` for CI enforcement.
- Baseline Comparison: Provide a historical `transport_metrics.json` via `--baseline` (and optionally `--update-baseline`) with `--max-latency-delta` / `--max-success-delta` to detect regressions relative to prior runs; baseline alerts appear in both Markdown and JSON outputs.
- Stdio Pool Metrics: The stdio worker pool records wait-time, active-worker, and available-worker histograms (`usb_agents.stdio.wait_time`, `usb_agents.stdio.active_workers`, `usb_agents.stdio.available_workers`) for debugging pooled execution; summaries appear in `results.md`, `transport_metrics.json`, and dashboards.

## Running the Evaluation Matrix

```bash
./scripts/run_matrix.py \
  --policy mcp-server/policy.yaml \
  --results results.csv \
  --markdown results.md
```

The script iterates the runtime × task matrix, captures traces, materialises per-run JSON rows, and writes `results.csv` plus a Markdown digest. Re-run after code changes to keep metrics current. Use `scripts/report.py` directly if you need to regenerate the Markdown from a stored CSV.

## References

- [OpenAI Agents SDK: MCP connectors & guardrails](https://platform.openai.com/docs/assistants/mcp)
- [Microsoft Agent Framework (preview)](https://learn.microsoft.com/azure/ai-services/agents/overview)
- [Mistral Agents API](https://docs.mistral.ai/agents)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
