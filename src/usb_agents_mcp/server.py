"""MCP-compatible server exposing the usb-agents benchmark tools."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence, cast
from uuid import uuid4

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from jsonschema import Draft7Validator, ValidationError
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, Field

from usb_agents.policy import PolicyConfig
from usb_agents.telemetry import configure_tracer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = ROOT / "mcp-server" / "policy.yaml"
SAFE_FS_ROOTS = [
    ROOT / "tasks",
    ROOT / "README.md",
    ROOT / "pyproject.toml",
    ROOT / "mcp-server",
    ROOT / "PRODUCT.md",
    ROOT / "DESIGN.md",
]
ALLOWED_ORIGINS = {"http://127.0.0.1", "http://localhost"}
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
PROTOCOL_VERSION = "2025-06-18"


class ApprovalContext(BaseModel):
    granted: bool = Field(default=False)
    reason: str = Field(default="unspecified")


class ToolRequest(BaseModel):
    arguments: dict[str, Any]
    approval: ApprovalContext | None = None


class ToolResponse(BaseModel):
    result: dict[str, Any]


TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "fs_read": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Allowlisted relative file path to read.",
                "x-mcp-header": "path",
            },
            "offset": {"type": "integer", "minimum": 0, "default": 0},
            "limit": {"type": "integer", "minimum": 0, "maximum": 16384},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    "http_fetch": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "format": "uri",
                "description": "Allowlisted URL to fetch.",
                "x-mcp-header": "url",
            },
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
            "source": {
                "type": "string",
                "default": "",
                "description": "Optional replacement source for code_module/example.py.",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    },
}

TOOL_DESCRIPTIONS = {
    "fs_read": "Read allowlisted repository files for deterministic triage tasks.",
    "http_fetch": "Fetch allowlisted synthetic JSON fixtures or approved HTTP JSON data.",
    "calendar_merge": "Plan calendar additions behind an explicit approval gate.",
    "code_exec": "Run allowlisted Python commands in a temporary sandbox.",
}


class MCPServer:
    def __init__(self, policy: PolicyConfig, traces_dir: Path | None = None) -> None:
        self.policy = policy
        self.validators = {name: Draft7Validator(schema) for name, schema in TOOL_SCHEMAS.items()}
        self.tracer = trace.get_tracer("usb_agents.mcp_server")
        self.traces_dir = traces_dir or Path("traces")
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: set[str] = set()

    async def call_tool(
        self,
        tool: str,
        arguments: dict[str, Any],
        approval: ApprovalContext | None,
    ) -> dict[str, Any]:
        """Validate, authorize, execute, and trace a tool call."""
        if tool not in TOOL_SCHEMAS:
            raise ValueError(f"Unknown tool '{tool}'")

        with self.tracer.start_as_current_span(
            "tool.call",
            attributes={"gen_ai.tool.name": tool, "gen_ai.request.model": "mcp-server"},
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
                attributes={"gen_ai.tool.name": tool, "approval.granted": granted},
            ) as exec_span:
                result = await resolver(arguments, approval)
                output_length = len(json.dumps(result))
                exec_span.set_attribute("gen_ai.response.output_length", output_length)
                tool_span.set_attribute("gen_ai.response.output_length", output_length)
                return result

    def list_tools(self) -> dict[str, Any]:
        return {
            "tools": [
                {
                    "name": name,
                    "description": TOOL_DESCRIPTIONS[name],
                    "inputSchema": schema,
                    "annotations": {
                        "readOnlyHint": name in {"fs_read", "http_fetch"},
                        "destructiveHint": False,
                        "idempotentHint": name != "code_exec",
                    },
                }
                for name, schema in sorted(TOOL_SCHEMAS.items())
            ]
        }

    def list_resources(self) -> dict[str, Any]:
        return {
            "resources": [
                {
                    "uri": "file:///README.md",
                    "name": "README.md",
                    "description": "Project overview and CLI guidance.",
                    "mimeType": "text/markdown",
                },
                {
                    "uri": "file:///mcp-server/policy.yaml",
                    "name": "policy.yaml",
                    "description": "Tool approval policy.",
                    "mimeType": "application/yaml",
                },
            ]
        }

    async def read_resource(self, uri: str) -> dict[str, Any]:
        if not uri.startswith("file:///"):
            raise ValueError(f"Unsupported resource URI '{uri}'")
        path = uri.removeprefix("file:///")
        result = await self._tool_fs_read({"path": path}, None)
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": _mime_type(Path(path)),
                    "text": result["content"],
                }
            ]
        }

    def list_prompts(self) -> dict[str, Any]:
        return {
            "prompts": [
                {
                    "name": "run-portability-suite",
                    "description": (
                        "Guide an agent through running and interpreting the default suite."
                    ),
                    "arguments": [],
                }
            ]
        }

    def get_prompt(self, name: str) -> dict[str, Any]:
        if name != "run-portability-suite":
            raise ValueError(f"Unknown prompt '{name}'")
        return {
            "description": "Run the usb-agents portability suite and summarize regressions.",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": (
                            "Run `usb-agents run --transports embedded http stdio`, then inspect "
                            "the generated metrics, traces, approvals, and artifacts."
                        ),
                    },
                }
            ],
        }

    async def handle_json_rpc(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = payload.get("id")
        try:
            if payload.get("jsonrpc") != "2.0":
                raise JsonRpcError(-32600, "Invalid Request")
            method = payload.get("method")
            params = payload.get("params") or {}
            result: dict[str, Any]
            if method == "initialize":
                session_id = uuid4().hex
                self.sessions.add(session_id)
                result = {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"subscribe": False, "listChanged": False},
                        "prompts": {"listChanged": False},
                    },
                    "serverInfo": {"name": "usb-agents", "version": "0.1.0"},
                    "_meta": {"sessionId": session_id},
                }
            elif method == "tools/list":
                result = self.list_tools()
            elif method == "tools/call":
                name = params.get("name")
                if not isinstance(name, str):
                    raise JsonRpcError(-32602, "tools/call requires a string name")
                arguments = params.get("arguments") or {}
                approval_payload = params.get("approval")
                approval = (
                    ApprovalContext.model_validate(approval_payload) if approval_payload else None
                )
                structured = await self.call_tool(name, arguments, approval)
                result = {
                    "content": [{"type": "text", "text": json.dumps(structured, sort_keys=True)}],
                    "structuredContent": structured,
                    "isError": False,
                }
            elif method == "resources/list":
                result = self.list_resources()
            elif method == "resources/read":
                result = await self.read_resource(params.get("uri", ""))
            elif method == "prompts/list":
                result = self.list_prompts()
            elif method == "prompts/get":
                result = self.get_prompt(params.get("name", ""))
            else:
                raise JsonRpcError(-32601, f"Method not found: {method}")
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except JsonRpcError as exc:
            return exc.to_response(request_id)
        except ValidationError as exc:
            return JsonRpcError(-32602, exc.message).to_response(request_id)
        except PermissionError as exc:
            return JsonRpcError(-32001, str(exc)).to_response(request_id)
        except Exception as exc:
            return JsonRpcError(-32000, str(exc)).to_response(request_id)

    async def _tool_fs_read(
        self, args: dict[str, Any], _: ApprovalContext | None
    ) -> dict[str, Any]:
        target = (ROOT / args["path"]).resolve()
        allowed = False
        for root in SAFE_FS_ROOTS:
            root = root.resolve()
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
        snippet = data[offset : offset + int(limit)] if limit is not None else data[offset:]
        return {"path": str(target), "content": snippet}

    async def _tool_http_fetch(
        self, args: dict[str, Any], _: ApprovalContext | None
    ) -> dict[str, Any]:
        url = args["url"]
        timeout = float(args.get("timeout", 2.0))
        if url in HTTP_ALLOWLIST:
            return {"url": url, "data": HTTP_ALLOWLIST[url], "source": "fixture"}

        with httpx.Client(timeout=timeout) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
        return {"url": url, "data": data, "source": "http"}

    async def _tool_calendar_merge(
        self, args: dict[str, Any], _: ApprovalContext | None
    ) -> dict[str, Any]:
        calendar_path = ROOT / "tasks" / "calendar.json"
        existing = json.loads(calendar_path.read_text(encoding="utf-8"))
        existing_titles = {entry["title"] for entry in existing["events"]}
        merged = existing["events"][:]
        additions = []
        skipped = []
        for event in args["proposed"]:
            if event["title"] in existing_titles:
                skipped.append(event)
                continue
            additions.append(event)
            merged.append(event)
        return {"added": additions, "skipped": skipped, "total_events": len(merged)}

    async def _tool_code_exec(
        self, args: dict[str, Any], _: ApprovalContext | None
    ) -> dict[str, Any]:
        command_key = args["command"]
        command = ALLOWED_COMMANDS[command_key]
        source = args.get("source", "")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            module_src = ROOT / "tasks" / "code_module"
            if module_src.exists():
                shutil.copytree(module_src, tmp_path / "code_module", dirs_exist_ok=True)
            if source:
                (tmp_path / "code_module" / "example.py").write_text(source, encoding="utf-8")
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=tmpdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=20)
        return {
            "command": command_key,
            "returncode": proc.returncode,
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
        }


class JsonRpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def to_response(self, request_id: Any) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": self.code, "message": self.message},
        }


app = FastAPI(title="usb-agents MCP server")


@app.post("/mcp")
async def mcp_endpoint(
    request: Request,
    mcp_method: str | None = Header(default=None, alias="Mcp-Method"),
    mcp_name: str | None = Header(default=None, alias="Mcp-Name"),
    mcp_session_id: str | None = Header(default=None, alias="Mcp-Session-Id"),
) -> JSONResponse:
    _validate_origin(request)
    accept = request.headers.get("accept", "")
    if (
        "application/json" not in accept
        and "text/event-stream" not in accept
        and "*/*" not in accept
    ):
        raise HTTPException(
            status_code=406, detail="MCP endpoint requires JSON or SSE Accept header"
        )
    raw_payload = await request.json()
    if not isinstance(raw_payload, dict):
        raise HTTPException(status_code=400, detail="MCP request body must be a JSON object")
    payload = cast(dict[str, Any], raw_payload)
    if mcp_method and mcp_method != payload.get("method"):
        raise HTTPException(
            status_code=400, detail="Mcp-Method header does not match request method"
        )
    params = payload.get("params") or {}
    request_name = params.get("name") or params.get("uri")
    if mcp_name and request_name and mcp_name != request_name:
        raise HTTPException(status_code=400, detail="Mcp-Name header does not match request params")
    server: MCPServer = request.app.state.server
    if payload.get("method") != "initialize":
        _validate_session(server, mcp_session_id)
    response_payload = await server.handle_json_rpc(payload)
    headers = {"MCP-Protocol-Version": PROTOCOL_VERSION}
    session_id = response_payload.get("result", {}).get("_meta", {}).get("sessionId")
    if isinstance(session_id, str):
        headers["Mcp-Session-Id"] = session_id
    return JSONResponse(response_payload, headers=headers)


@app.get("/mcp")
async def mcp_event_stream(
    request: Request,
    mcp_session_id: str | None = Header(default=None, alias="Mcp-Session-Id"),
) -> StreamingResponse:
    _validate_origin(request)
    server: MCPServer = request.app.state.server
    _validate_session(server, mcp_session_id)

    async def stream():
        yield f"event: endpoint\ndata: {json.dumps({'status': 'ready'})}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"MCP-Protocol-Version": PROTOCOL_VERSION},
    )


@app.delete("/mcp")
async def mcp_delete_session(
    request: Request,
    mcp_session_id: str | None = Header(default=None, alias="Mcp-Session-Id"),
) -> Response:
    _validate_origin(request)
    server: MCPServer = request.app.state.server
    _validate_session(server, mcp_session_id)
    server.sessions.discard(str(mcp_session_id))
    return Response(status_code=204, headers={"MCP-Protocol-Version": PROTOCOL_VERSION})


@app.post("/tools/{tool_name}")
async def call_tool(tool_name: str, payload: ToolRequest) -> ToolResponse:
    server = app.state.server  # type: ignore[attr-defined]
    try:
        result = await server.call_tool(tool_name, payload.arguments, payload.approval)
    except (ValidationError, PermissionError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ToolResponse(result=result)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "protocol": "mcp-json-rpc"}


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
            if "tool" in payload:
                approval_payload = payload.get("approval")
                approval = (
                    ApprovalContext.model_validate(approval_payload) if approval_payload else None
                )
                result = await server.call_tool(
                    payload["tool"], payload.get("arguments", {}), approval
                )
                response: dict[str, Any] = {"result": result}
            else:
                response = await server.handle_json_rpc(payload)
        except Exception as exc:  # pragma: no cover - exercised via tests
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32000, "message": str(exc)},
            }
        writer.write(json.dumps(response).encode() + b"\n")
        await writer.drain()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the usb-agents MCP server.")
    parser.add_argument("--transport", choices=["http", "stdio"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy", type=str, default=str(DEFAULT_POLICY_PATH))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_tracer(service_name="usb-agents-mcp-server")
    policy = PolicyConfig.from_file(Path(args.policy))
    server = MCPServer(policy=policy)
    if args.transport == "http":
        run_http_server(args.host, args.port, server)
        return 0
    asyncio.run(run_stdio(server))
    return 0


def _validate_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if not origin:
        return
    if any(origin.startswith(allowed) for allowed in ALLOWED_ORIGINS):
        return
    raise HTTPException(status_code=403, detail="Origin is not allowed")


def _validate_session(server: MCPServer, session_id: str | None) -> None:
    if not server.sessions:
        return
    if not session_id or session_id not in server.sessions:
        raise HTTPException(status_code=404, detail="Unknown MCP session")


def _mime_type(path: Path) -> str:
    if path.suffix == ".md":
        return "text/markdown"
    if path.suffix in {".yaml", ".yml"}:
        return "application/yaml"
    if path.suffix == ".json":
        return "application/json"
    return "text/plain"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
