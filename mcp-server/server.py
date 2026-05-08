"""Script compatibility wrapper for the installable usb-agents MCP server."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from usb_agents_mcp.server import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
