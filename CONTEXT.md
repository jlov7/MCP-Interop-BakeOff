# Domain Context

## Core Terms

- **MCP portability lab**: Local OSS tool for comparing how agent runtimes behave against the
  same MCP tools, task suite, transports, approvals, traces, and artifacts.
- **Run**: One execution of a suite across one or more runtimes and transports.
- **Task case**: A deterministic scenario that exercises a specific tool or approval shape.
- **Runtime adapter**: The adapter that makes one agent runtime execute the shared task cases.
- **Transport adapter**: The adapter used to call MCP tools: embedded, stdio, or Streamable HTTP.
- **Artifact manifest**: The indexed list of trace/report/output files produced by a run.
- **Regression**: A measurable negative delta against a baseline run: success, latency, approvals,
  trace completeness, or failure category.

## Architecture Vocabulary

- Keep runtime and transport adapters deep: callers should not know process spawning, JSON-RPC
  envelope construction, approval payloads, or trace storage details.
- Treat the run executor as the orchestration module. CLI, API, and future UI-triggered runs should
  call that module instead of duplicating orchestration.
- Keep task fixtures immutable. Generated outputs belong under `.usb-agents/runs/<run-id>/`.
