# Example MCP Server Fixture

This fixture documents the protocol shape usb-agents expects from local MCP-compatible servers.
The primary runnable server lives in `src/usb_agents_mcp/server.py`; this example is intentionally
small so adapter authors have a stable reference.

## Expected Methods

- `initialize`
- `tools/list`
- `tools/call`
- `resources/read`
- `prompts/get`

## HTTP Shape

```bash
uv run usb-agents mcp --transport http --port 9000
curl -s http://127.0.0.1:9000/mcp \
  -H 'content-type: application/json' \
  -H 'accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":"1","method":"initialize","params":{}}'
```

## Stdio Shape

```bash
printf '%s\n' '{"jsonrpc":"2.0","id":"1","method":"tools/list","params":{}}' \
  | uv run usb-agents mcp --transport stdio
```

Use this fixture as a checklist when comparing external MCP servers against usb-agents'
transport contracts.
