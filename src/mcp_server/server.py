"""Compatibility wrapper for the usb-agents MCP server."""

from usb_agents_mcp.server import *  # noqa: F403

if __name__ == "__main__":  # pragma: no cover
    from usb_agents_mcp.server import main

    raise SystemExit(main())
