import asyncio
from pathlib import Path

import pytest

from mcp_server.server import ApprovalContext, MCPServer
from usb_agents.policy import load_policy


@pytest.fixture(scope="module")
def server() -> MCPServer:
    policy = load_policy(Path("mcp-server/policy.yaml"))
    return MCPServer(policy=policy, traces_dir=Path("traces"))


def test_fs_read(server: MCPServer):
    result = asyncio.run(server.call_tool("fs_read", {"path": "README.md"}, None))
    assert "usb-agents" in result["content"]


def test_calendar_requires_approval(server: MCPServer):
    with pytest.raises(PermissionError):
        asyncio.run(
            server.call_tool(
                "calendar_merge",
                {"proposed": []},
                None,
            )
        )


def test_calendar_merge_with_approval(server: MCPServer):
    approval = ApprovalContext(granted=True, reason="test")
    result = asyncio.run(
        server.call_tool(
            "calendar_merge",
            {
                "proposed": [
                    {
                        "title": "Interop Dry Run",
                        "time": "2024-07-03T16:00:00Z",
                        "duration_minutes": 30,
                    }
                ]
            },
            approval,
        )
    )
    assert result["total_events"] >= 3


def test_code_exec_pyflakes(server: MCPServer):
    approval = ApprovalContext(granted=True, reason="test")
    result = asyncio.run(server.call_tool("code_exec", {"command": "py_compile"}, approval))
    assert result["returncode"] == 0
