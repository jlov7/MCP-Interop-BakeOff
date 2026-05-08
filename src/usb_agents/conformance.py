"""External MCP conformance runner integration."""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import httpx

DEFAULT_CONFORMANCE_PACKAGE = "@modelcontextprotocol/conformance@0.1.16"
DEFAULT_SCENARIOS = ("server-initialize", "tools-list", "resources-list", "prompts-list")


@dataclass(frozen=True)
class ConformanceResult:
    scenario: str
    returncode: int
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def conformance_command(
    *,
    package: str,
    url: str,
    scenario: str,
    verbose: bool = True,
) -> list[str]:
    command = ["npx", "-y", package, "server", "--url", url, "--scenario", scenario]
    if verbose:
        command.append("--verbose")
    return command


def run_conformance(
    *,
    url: str | None = None,
    host: str = "127.0.0.1",
    port: int = 9010,
    policy_path: Path = Path("mcp-server/policy.yaml"),
    scenarios: Sequence[str] = DEFAULT_SCENARIOS,
    package: str = DEFAULT_CONFORMANCE_PACKAGE,
) -> list[ConformanceResult]:
    server_process: subprocess.Popen[bytes] | None = None
    target_url = url or f"http://{host}:{port}/mcp"
    if url is None:
        server_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mcp_server.server",
                "--transport",
                "http",
                "--host",
                host,
                "--port",
                str(port),
                "--policy",
                str(policy_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _wait_for_health(f"http://{host}:{port}")

    try:
        results = []
        for scenario in scenarios:
            completed = subprocess.run(
                conformance_command(package=package, url=target_url, scenario=scenario),
                check=False,
                capture_output=True,
                text=True,
            )
            results.append(
                ConformanceResult(
                    scenario=scenario,
                    returncode=completed.returncode,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                )
            )
        return results
    finally:
        if server_process is not None:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()


def _wait_for_health(base_url: str) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{base_url}/health", timeout=1)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for MCP server at {base_url}")
