# MCP Compliance Notes

usb-agents implements the MCP surfaces needed for local portability testing and keeps explicit
tests around each protocol behavior.

## Implemented

- JSON-RPC request and response envelopes.
- `initialize` with `MCP-Protocol-Version` response header and session metadata.
- `tools/list` and `tools/call`.
- Structured content in tool results.
- `resources/read` for policy resources.
- `prompts/get` for benchmark guidance.
- Streamable HTTP POST endpoint at `/mcp`.
- Streamable HTTP GET endpoint for server-sent events.
- Streamable HTTP DELETE endpoint for session teardown.
- Origin allowlist rejection.
- Stdio server with stdout reserved for JSON-RPC messages.
- Path traversal rejection for local artifacts.

## Deliberately Local-First

The HTTP server is designed for localhost development. It still validates `Origin` and has
auth hooks in the API/server seams, but the first supported deployment shape is a local console,
not a hosted multi-tenant service.

## External Conformance

The project includes an external smoke gate against the official MCP conformance framework:

```bash
uv run usb-agents conformance
```

This currently runs `@modelcontextprotocol/conformance@0.1.16` against these server scenarios:

- `server-initialize`
- `tools-list`
- `resources-list`
- `prompts-list`

Local evidence from May 8, 2026: all four scenarios passed against the local `/mcp` endpoint.
The broader conformance suite includes scenarios for capabilities usb-agents does not yet claim,
such as sampling, elicitation, subscriptions, binary resources, and prompt image responses.

## Release Gate

The compliance gate is:

```bash
uv run pytest tests/test_mcp_protocol.py tests/test_transport_contracts.py
uv run usb-agents run --transport embedded --transport http --transport stdio
uv run usb-agents conformance
```
