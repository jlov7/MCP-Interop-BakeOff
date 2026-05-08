from pathlib import Path

from fastapi.testclient import TestClient

from mcp_server.server import MCPServer, app
from usb_agents.policy import load_policy


def make_client() -> TestClient:
    app.state.server = MCPServer(load_policy(Path("mcp-server/policy.yaml")))
    return TestClient(app)


def test_mcp_tools_list_json_rpc():
    client = make_client()
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": "1", "method": "tools/list", "params": {}},
        headers={"Accept": "application/json", "Mcp-Method": "tools/list"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["jsonrpc"] == "2.0"
    assert {tool["name"] for tool in payload["result"]["tools"]} >= {
        "fs_read",
        "http_fetch",
        "calendar_merge",
        "code_exec",
    }


def test_mcp_tools_call_returns_structured_content():
    client = make_client()
    init = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": "init", "method": "initialize", "params": {}},
        headers={"Accept": "application/json", "Mcp-Method": "initialize"},
    )
    session_id = init.headers["Mcp-Session-Id"]
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": "2",
            "method": "tools/call",
            "params": {"name": "fs_read", "arguments": {"path": "README.md", "limit": 20}},
        },
        headers={
            "Accept": "application/json",
            "Mcp-Method": "tools/call",
            "Mcp-Name": "fs_read",
            "Mcp-Session-Id": session_id,
        },
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert "structuredContent" in result
    assert "usb-agents" in result["structuredContent"]["content"]


def test_mcp_initialize_assigns_streamable_http_session():
    client = make_client()
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": "init", "method": "initialize", "params": {}},
        headers={"Accept": "application/json", "Mcp-Method": "initialize"},
    )

    assert response.status_code == 200
    assert response.headers["MCP-Protocol-Version"] == "2025-06-18"
    assert response.headers["Mcp-Session-Id"]


def test_mcp_requires_known_session_after_initialize():
    client = make_client()
    client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": "init", "method": "initialize", "params": {}},
        headers={"Accept": "application/json", "Mcp-Method": "initialize"},
    )

    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": "1", "method": "tools/list", "params": {}},
        headers={"Accept": "application/json", "Mcp-Method": "tools/list"},
    )

    assert response.status_code == 404


def test_mcp_event_stream_and_delete_session():
    client = make_client()
    init = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": "init", "method": "initialize", "params": {}},
        headers={"Accept": "application/json", "Mcp-Method": "initialize"},
    )
    session_id = init.headers["Mcp-Session-Id"]

    stream = client.get(
        "/mcp", headers={"Accept": "text/event-stream", "Mcp-Session-Id": session_id}
    )
    assert stream.status_code == 200
    assert "event: endpoint" in stream.text

    deleted = client.delete("/mcp", headers={"Mcp-Session-Id": session_id})
    assert deleted.status_code == 204


def test_mcp_rejects_bad_origin():
    client = make_client()
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": "3", "method": "tools/list", "params": {}},
        headers={
            "Accept": "application/json",
            "Origin": "https://attacker.example",
            "Mcp-Method": "tools/list",
        },
    )
    assert response.status_code == 403


def test_mcp_rejects_path_traversal():
    client = make_client()
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": "4",
            "method": "tools/call",
            "params": {"name": "fs_read", "arguments": {"path": "../.env"}},
        },
        headers={
            "Accept": "application/json",
            "Mcp-Method": "tools/call",
            "Mcp-Name": "fs_read",
        },
    )
    payload = response.json()
    assert payload["error"]["code"] == -32001
