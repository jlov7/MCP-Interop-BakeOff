"""Loader proxy exposing mcp-server/server.py as a package import."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict


def _load_impl() -> ModuleType:
    impl_path = Path(__file__).resolve().parent.parent / "mcp-server" / "server.py"
    spec = importlib.util.spec_from_file_location("mcp_server._impl", impl_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError("Unable to locate mcp-server/server.py implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_IMPL = _load_impl()


def _export_public(module: ModuleType) -> Dict[str, object]:
    exported: Dict[str, object] = {}
    for name in dir(module):
        if name.startswith("_"):
            continue
        exported[name] = getattr(module, name)
    return exported


globals().update(_export_public(_IMPL))
__all__ = [name for name in globals() if not name.startswith("_")]


if __name__ == "__main__":  # pragma: no cover
    if hasattr(_IMPL, "main"):
        raise SystemExit(_IMPL.main())
    raise SystemExit("mcp_server.server implementation missing main()")
