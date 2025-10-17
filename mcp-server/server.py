"""Shared MCP server exposing fs_read, http_fetch, calendar_merge, and code_exec tools.

The module can be executed as a script (`python mcp-server/server.py --transport http`)
or imported via the `mcp_server.server` alias registered at import time. Tool calls always
emit OpenTelemetry spans using the GenAI semantic conventions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import httpx
from fastapi import FastAPI, HTTPException
from jsonschema import Draft7Validator, ValidationError
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, Field
from usb_agents.policy import PolicyConfig
from usb_agents.telemetry import configure_tracer


# Register module alias so `import mcp_server.server` succeeds even though the file lives
# under `mcp-server/server.py`.
_module = sys.modules.setdefault("mcp_server", sys.modules[__name__])
setattr(_module, "__path__", [str(Path(__file__).parent)])
sys.modules.setdefault("mcp_server.server", sys.modules[__name__])


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY_PATH = Path(__file__).resolve().parent / "policy.yaml"
SAFE_FS_ROOTS = [
    ROOT / "tasks",
    ROOT / "README.md",
    ROOT / "pyproject.toml",
    ROOT / "mcp-server",
]
HTTP_ALLOWLIST = {
    "https://example.com/synthetic-dataset": {
        "users": [
            {"id": 1, "name": "Ada Lovelace", "role": "engineer"},
            {"id": 2, "name": "Grace Hopper", "role": "admiral"},
        ]
    }
}
ALLOWED_COMMANDS = {
    "py_compile": [sys.executable, "-m", "py_compile", "code_module/example.py"],
    "pytest": [sys.executable, "-m", "pytest", "-q"],
}



class ApprovalContext(BaseModel):
    granted: bool = Field(default=False)
    reason: str = Field(default="unspecified")


class ToolRequest(BaseModel):
    arguments: Dict[str, Any]
    approval: Optional[ApprovalContext] = None


TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "fs_read": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "minimum": 0, "default": 0},
            "limit": {"type": "integer", "minimum": 0, "maximum": 16384},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    "http_fetch": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "format": "uri"},
            "timeout": {"type": "number", "minimum": 0.1, "maximum": 10, "default": 2.0},
        },
        "required": ["url"],
        "additionalProperties": False,
    },
    "calendar_merge": {
        "type": "object",
        "properties": {
            "proposed": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "time": {"type": "string"},
                        "duration_minutes": {"type": "integer", "minimum": 1, "maximum": 480},
                    },
                    "required": ["title", "time", "duration_minutes"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["proposed"],
        "additionalProperties": False,
    },
    "code_exec": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "enum": list(ALLOWED_COMMANDS.keys())},
            "source": {"type": "string", "default": ""},
        },
        "required": ["command"],
        "additionalProperties": False,
    },
}


class MCPServer:
    def __init__(self, policy: PolicyConfig, traces_dir: Optional[Path] = None) -> None:
        self.policy = policy
        self.validators = {
            name: Draft7Validator(schema) for name, schema in TOOL_SCHEMAS.items()
        }
        self.tracer = trace.get_tracer("usb_agents.mcp_server")
        self.traces_dir = traces_dir or Path("traces")
        self.traces_dir.mkdir(parents=True, exist_ok=True)

    async def call_tool(self, tool: str, arguments: Dict[str, Any], approval: Optional[ApprovalContext]) -> Dict[str, Any]:
        """Validate, authorize, execute, and trace the tool call."""
        if tool not in TOOL_SCHEMAS:
            raise ValueError(f"Unknown tool '{tool}'")

        with self.tracer.start_as_current_span(
            "tool.call",
            attributes={
                "gen_ai.tool.name": tool,
                "gen_ai.request.model": "mcp-server",
            },
        ) as tool_span:
            validator = self.validators[tool]
            try:
                validator.validate(arguments)
            except ValidationError as exc:
                tool_span.record_exception(exc)
                tool_span.set_status(
                    Status(status_code=StatusCode.ERROR, description=f"validation:{exc.message}")
                )
                raise

            approval_required = self.policy.requires_approval(tool)
            granted = approval.granted if approval else False
            if approval_required and not granted:
                msg = f"Tool '{tool}' requires approval"
                tool_span.add_event(
                    "approval.required",
                    {
                        "gen_ai.tool.name": tool,
                        "approval.required": True,
                        "approval.reason": self.policy.requirement_for(tool),
                    },
                )
                raise PermissionError(msg)

            resolver = getattr(self, f"_tool_{tool}")
            with self.tracer.start_as_current_span(
                f"tool.execute.{tool}",
                attributes={
                    "gen_ai.tool.name": tool,
                    "approval.granted": granted,
                },
            ) as exec_span:
                result = await resolver(arguments, approval)
                exec_span.set_attribute("gen_ai.response.output_length", len(json.dumps(result)))
                tool_span.set_attribute("gen_ai.response.output_length", len(json.dumps(result)))
                return result

    async def _tool_fs_read(self, args: Dict[str, Any], _: Optional[ApprovalContext]) -> Dict[str, Any]:
        target = Path(args["path"]).resolve()
        allowed = False
        for root in SAFE_FS_ROOTS:
            if root.is_dir() and target.is_relative_to(root):
                allowed = True
                break
            if root.is_file() and target == root:
                allowed = True
                break
        if not allowed:
            raise PermissionError(f"Path '{target}' not in allowlist")
        offset = int(args.get("offset", 0))
        limit = args.get("limit")
        data = target.read_text(encoding="utf-8")
        if limit is not None:
            limit = int(limit)
            snippet = data[offset : offset + limit]
        else:
            snippet = data[offset:]
        return {"path": str(target), "content": snippet}

    async def _tool_http_fetch(self, args: Dict[str, Any], _: Optional[ApprovalContext]) -> Dict[str, Any]:
        url = args["url"]
        timeout = float(args.get("timeout", 2.0))
        if url in HTTP_ALLOWLIST:
            return {"url": url, "data": HTTP_ALLOWLIST[url], "source": "fixture"}

        with httpx.Client(timeout=timeout) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
        return {"url": url, "data": data, "source": "http"}

    async def _tool_calendar_merge(self, args: Dict[str, Any], _: Optional[ApprovalContext]) -> Dict[str, Any]:
        calendar_path = ROOT / "tasks" / "calendar.json"
        existing = json.loads(calendar_path.read_text(encoding="utf-8"))
        existing_titles = {entry["title"] for entry in existing["events"]}
        merged = existing["events"][:]
        additions = []
        for event in args["proposed"]:
            if event["title"] in existing_titles:
                continue
            additions.append(event)
            merged.append(event)
        return {"added": additions, "total_events": len(merged)}

    async def _tool_code_exec(self, args: Dict[str, Any], _: Optional[ApprovalContext]) -> Dict[str, Any]:
        command_key = args["command"]
        command = ALLOWED_COMMANDS[command_key]
        source = args.get("source", "")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            module_src = ROOT / "tasks" / "code_module"
            if module_src.exists():
                shutil.copytree(module_src, tmp_path / "code_module", dirs_exist_ok=True)
            if source:
                (tmp_path / "snippet.py").write_text(source, encoding="utf-8")
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=tmpdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
        return {
            "command": command_key,
            "returncode": proc.returncode,
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
        }


class ToolResponse(BaseModel):
    result: Dict[str, Any]


app = FastAPI()


@app.post("/tools/{tool_name}")
async def call_tool(tool_name: str, payload: ToolRequest) -> ToolResponse:
    server = app.state.server  # type: ignore[attr-defined]
    try:
        result = await server.call_tool(tool_name, payload.arguments, payload.approval)
    except (ValidationError, PermissionError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ToolResponse(result=result)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


def run_http_server(host: str, port: int, server: MCPServer) -> None:
    import uvicorn

    app.state.server = server  # type: ignore[attr-defined]
    uvicorn.run(app, host=host, port=port)


async def run_stdio(server: MCPServer) -> None:
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    writer_transport, writer_protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout
    )
    writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, loop)

    while True:
        line = await reader.readline()
        if not line:
            break
        try:
            payload = json.loads(line.decode())
            tool = payload["tool"]
            arguments = payload.get("arguments", {})
            approval_payload = payload.get("approval")
            approval = ApprovalContext.model_validate(approval_payload) if approval_payload else None
            result = await server.call_tool(tool, arguments, approval)
        except Exception as exc:  # pragma: no cover - exercised via tests
            writer.write(json.dumps({"error": str(exc)}).encode() + b"\n")
        else:
            writer.write(json.dumps({"result": result}).encode() + b"\n")
        await writer.drain()


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the usb-agents MCP server.")
    parser.add_argument("--transport", choices=["http", "stdio"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy", type=str, default=str(DEFAULT_POLICY_PATH))
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    configure_tracer(service_name="usb-agents-mcp-server")
    policy = PolicyConfig.from_file(Path(args.policy))
    server = MCPServer(policy=policy)
    if args.transport == "http":
        run_http_server(args.host, args.port, server)
        return 0
    asyncio.run(run_stdio(server))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
