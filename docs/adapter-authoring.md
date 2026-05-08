# Adapter Authoring

usb-agents separates runtime adapters from transport adapters so new agent frameworks can be
added without changing benchmark tasks or MCP server code.

## Runtime Adapter Contract

A runtime adapter owns orchestration behavior for one agent framework. It receives a task case,
an approval policy, and a tool client. It returns structured case results, tool calls, approval
decisions, trace events, and artifacts through the shared run executor.

Register adapters in `usb-agents.yaml`:

```yaml
runtimes:
  - id: openai_agents
    mode: simulator
    enabled: true
  - id: my_runtime
    mode: external
    enabled: true
    credential_env: MY_RUNTIME_API_KEY
```

The registry intentionally exposes unconfigured real adapters as `unconfigured` until their
credential environment variables are present. This keeps a fresh clone deterministic while
making real SDK paths explicit.

## Transport Adapter Contract

Transport adapters expose the same tool-call behavior through different protocol surfaces:

- `embedded`: direct in-process tool execution for deterministic tests.
- `http`: Streamable HTTP JSON-RPC through `/mcp`.
- `stdio`: line-delimited JSON-RPC over a subprocess boundary.

Transport contract tests should prove that `tools/list`, `tools/call`, approval denial, error
envelopes, and artifact writes produce equivalent results across transports.

## Adding a Runtime

1. Add a runtime descriptor to `src/usb_agents/adapters.py`.
2. Add an adapter implementation behind the existing run executor seam.
3. Keep simulator mode deterministic and credential-gate real SDK calls.
4. Add contract tests that run at least one read task and one approval-gated write task.
5. Document required environment variables and any local model/service prerequisites.

## Acceptance Bar

- Fresh clone tests pass without credentials.
- `uv run usb-agents sdk-smoke` shows the SDK package and credential state for each vendor.
- `uv run usb-agents sdk-smoke --runtime <id> --require` fails unless the relevant credential
  environment variable is present.
- The adapter emits trace events for task start, tool call, approval decision, and task result.
- Secrets are redacted before traces and artifacts are persisted.
