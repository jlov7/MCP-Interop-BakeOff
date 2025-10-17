"""Transport abstraction for calling MCP tools via embedded, HTTP, or stdio transports."""

from __future__ import annotations

import asyncio
import asyncio.subprocess as aio_subprocess
import contextlib
import json
import os
import shlex
import sys
import time
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from mcp_server.server import ApprovalContext, MCPServer
from usb_agents.telemetry import (
    record_stdio_active,
    record_stdio_available,
    record_stdio_wait,
)


class ToolClient:
    async def call_tool(
        self, name: str, arguments: Dict[str, Any], approval_payload: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None

    def close(self) -> None:
        if asyncio.get_event_loop().is_running():  # pragma: no cover
            asyncio.create_task(self.aclose())
        else:
            asyncio.run(self.aclose())


@dataclass
class EmbeddedToolClient(ToolClient):
    """Direct in-process calls against an MCPServer instance."""

    server: MCPServer

    async def call_tool(
        self, name: str, arguments: Dict[str, Any], approval_payload: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        approval = ApprovalContext.model_validate(approval_payload or {}) if approval_payload else None
        return await self.server.call_tool(name, arguments, approval)


class HttpToolClient(ToolClient):
    """HTTP client targeting the shared MCP REST surface."""

    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def call_tool(
        self, name: str, arguments: Dict[str, Any], approval_payload: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"arguments": arguments}
        if approval_payload:
            payload["approval"] = approval_payload
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            response = await client.post(f"/tools/{name}", json=payload)
        response.raise_for_status()
        data = response.json()
        if "result" not in data:
            raise ValueError(f"HTTP tool response missing result field: {data}")
        return data["result"]


class StdioToolClient(ToolClient):
    """Stdio client that spawns the MCP server and communicates over pipes."""

    @dataclass
    class _Worker:
        proc: aio_subprocess.Process
        stderr_task: asyncio.Task
        stderr_lines: List[str]

    def __init__(
        self,
        command: Optional[str] = None,
        *,
        cwd: Optional[Path] = None,
        concurrency: int = 1,
    ) -> None:
        default_policy = Path(__file__).resolve().parents[1] / "mcp-server" / "policy.yaml"
        default_cmd_list = [
            sys.executable,
            "-m",
            "mcp_server.server",
            "--transport",
            "stdio",
            "--policy",
            str(default_policy),
        ]
        env_cmd = os.getenv("MCP_SERVER_STDIO_CMD")
        if command or env_cmd:
            cmd_str = command or env_cmd
            self.command = cmd_str
            self._cmd = shlex.split(cmd_str)
        else:
            self.command = " ".join(default_cmd_list)
            self._cmd = default_cmd_list
        self.cwd = cwd
        self._max_workers = max(1, concurrency)
        self._available: asyncio.Queue[StdioToolClient._Worker] = asyncio.Queue()
        self._workers: List[StdioToolClient._Worker] = []
        self._creation_lock = asyncio.Lock()
        self._active_workers = 0.0
        self._wait_samples: List[float] = []
        self._active_samples: List[float] = []
        self._available_samples: List[float] = []

    async def warmup(self, count: int) -> None:
        target = min(self._max_workers, max(0, count))
        await self._ensure_workers(target)

    async def _ensure_workers(self, target: int) -> None:
        async with self._creation_lock:
            while len(self._workers) < target:
                worker = await self._create_worker()
                self._workers.append(worker)
                await self._available.put(worker)

    async def _create_worker(self) -> _Worker:
        proc = await asyncio.create_subprocess_exec(
            *self._cmd,
            stdin=aio_subprocess.PIPE,
            stdout=aio_subprocess.PIPE,
            stderr=aio_subprocess.PIPE,
            cwd=str(self.cwd) if self.cwd else None,
        )
        stderr_lines: List[str] = []
        stderr_task = asyncio.create_task(self._consume_stderr(proc.stderr, stderr_lines))
        return StdioToolClient._Worker(proc=proc, stderr_task=stderr_task, stderr_lines=stderr_lines)

    async def _consume_stderr(
        self, stream: Optional[asyncio.StreamReader], buffer: List[str]
    ) -> None:
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            buffer.append(line.decode(errors="ignore").strip())

    async def _acquire_worker(self) -> _Worker:
        start = time.perf_counter()
        worker: Optional[StdioToolClient._Worker] = None
        try:
            worker = self._available.get_nowait()
        except asyncio.QueueEmpty:
            async with self._creation_lock:
                if len(self._workers) < self._max_workers:
                    worker = await self._create_worker()
                    self._workers.append(worker)
                else:
                    worker = None
        if worker is None:
            worker = await self._available.get()
        wait_ms = (time.perf_counter() - start) * 1000
        self._wait_samples.append(wait_ms)
        record_stdio_wait(wait_ms)
        self._active_workers += 1
        self._active_samples.append(self._active_workers)
        record_stdio_active(self._active_workers)
        available = max(0.0, self._max_workers - self._active_workers)
        self._available_samples.append(available)
        record_stdio_available(available)
        return worker

    async def _release_worker(self, worker: _Worker) -> None:
        if worker.proc.returncode is None:
            self._active_workers = max(0.0, self._active_workers - 1)
            record_stdio_active(self._active_workers)
            available = max(0.0, self._max_workers - self._active_workers)
            self._available_samples.append(available)
            record_stdio_available(available)
            worker.stderr_lines.clear()
            await self._available.put(worker)
        else:
            await self._retire_worker(worker)

    async def _retire_worker(self, worker: _Worker, decrement_active: bool = True) -> None:
        if decrement_active:
            self._active_workers = max(0.0, self._active_workers - 1)
            record_stdio_active(self._active_workers)
            available = max(0.0, self._max_workers - self._active_workers)
            self._available_samples.append(available)
            record_stdio_available(available)
        if worker in self._workers:
            self._workers.remove(worker)
        await self._close_worker(worker)

    async def call_tool(
        self, name: str, arguments: Dict[str, Any], approval_payload: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        worker = await self._acquire_worker()
        try:
            proc = worker.proc
            if proc.stdin is None or proc.stdout is None:
                raise RuntimeError("stdio MCP server missing stdin/stdout pipes")
            payload = {"tool": name, "arguments": arguments}
            if approval_payload:
                payload["approval"] = approval_payload
            proc.stdin.write(json.dumps(payload).encode() + b"\n")
            await proc.stdin.drain()
            raw_line = await proc.stdout.readline()
            if not raw_line:
                stderr_text = "\n".join(worker.stderr_lines)
                raise RuntimeError(f"stdio MCP server produced no output. stderr: {stderr_text}")
            response = json.loads(raw_line.decode())
            if "error" in response:
                raise RuntimeError(response["error"])
            if "result" not in response:
                stderr_text = "\n".join(worker.stderr_lines)
                raise ValueError(f"Invalid stdio response: {response}, stderr: {stderr_text}")
            result = response["result"]
            worker.stderr_lines.clear()
        except Exception:
            await self._retire_worker(worker)
            raise
        else:
            await self._release_worker(worker)
            return result

    async def aclose(self) -> None:
        async with self._creation_lock:
            workers = list(self._workers)
            self._workers.clear()
            while not self._available.empty():
                with contextlib.suppress(asyncio.QueueEmpty):
                    self._available.get_nowait()
        for worker in workers:
            await self._close_worker(worker)

    async def collect_stats(self, reset: bool = True) -> Dict[str, Any]:
        async with self._creation_lock:
            wait_samples = list(self._wait_samples)
            active_samples = list(self._active_samples)
            available_samples = list(self._available_samples)
            if reset:
                self._wait_samples.clear()
                self._active_samples.clear()
                self._available_samples.clear()
        return {
            "concurrency": self._max_workers,
            "wait_ms": self._compute_stats(wait_samples),
            "active_workers": self._compute_stats(active_samples),
            "available_workers": self._compute_stats(available_samples),
        }

    @staticmethod
    def _compute_stats(samples: List[float]) -> Dict[str, float]:
        if not samples:
            return {"count": 0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        samples_sorted = sorted(samples)
        count = len(samples_sorted)
        avg = sum(samples_sorted) / count
        median = statistics.median(samples_sorted)
        p95 = StdioToolClient._percentile(samples_sorted, 0.95)
        maximum = samples_sorted[-1]
        return {
            "count": count,
            "avg": avg,
            "p50": median,
            "p95": p95,
            "max": maximum,
        }

    @staticmethod
    def _percentile(values: List[float], pct: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        k = (len(values) - 1) * pct
        f = int(k)
        c = min(f + 1, len(values) - 1)
        if f == c:
            return values[f]
        return values[f] + (values[c] - values[f]) * (k - f)

    async def _close_worker(self, worker: _Worker) -> None:
        proc = worker.proc
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        if worker.stderr_task:
            worker.stderr_task.cancel()
            with contextlib.suppress(Exception):
                await worker.stderr_task
