# MCP Server

The shared MCP server supplies four deterministic tools that can be exercised over HTTP or stdio:

| Tool | Description | Approval default |
| --- | --- | --- |
| `fs_read` | Read files from the project allowlist. Supports UTF‑8 text only. | `never` |
| `http_fetch` | Fetch JSON from an allowlisted URL set. Falls back to baked-in fixtures. | `by_tool` |
| `calendar_merge` | Merge a proposed calendar into `tasks/calendar.json` without destructive edits. | `by_tool` |
| `code_exec` | Execute lint/test commands (`py_compile`, `pytest`) within a sandboxed temporary directory. | `by_tool` |

## Policy Model

The `policy.yaml` file configures the approval mode per tool (`never`, `always`, `by_tool`). The server enforces write gates by requiring an approval token whenever a tool is marked `by_tool` or `always`. Runners must surface the approval prompt before performing the action.

## Running

```bash
python -m mcp_server.server --transport http --host 0.0.0.0 --port 8000
```

or stdio:

```bash
python -m mcp_server.server --transport stdio
```

Both transports expose the same tool schemas. HTTP surfaces `GET /health` for readiness checks and accepts `POST /tools/<name>` with body:

```json
{
  "arguments": {...},
  "approval": {"granted": true, "reason": "manual approval"}
}
```

Stdio uses newline-delimited JSON with the same payload.
