"""Run orchestration shared by the CLI and local API."""

from __future__ import annotations

import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

import httpx

from usb_agents.adapters import configured_runtimes
from usb_agents.config import load_config
from usb_agents.models import Run, new_run_id
from usb_agents.policy import load_policy
from usb_agents.run_store import create_run_from_results, ensure_runs_root
from usb_agents.tasks import load_tasks

VALID_TRANSPORTS = {"embedded", "http", "stdio"}


def execute_run(
    *,
    transports: Sequence[str],
    policy_path: Path = Path("mcp-server/policy.yaml"),
    run_root: Path | None = None,
    config_path: Path | None = None,
    timeout_seconds: float | None = None,
    cancel_event: threading.Event | None = None,
) -> Run:
    config = load_config(config_path)
    selected = [transport.strip() for transport in transports if transport.strip()]
    if not selected:
        selected = config.default_transports
    unsupported = sorted(set(selected) - VALID_TRANSPORTS)
    if unsupported:
        raise ValueError(f"Unsupported transport(s): {', '.join(unsupported)}")

    run_id = new_run_id().removeprefix("run_")
    run_root = run_root or ensure_runs_root() / run_id
    traces_dir = run_root / "traces"
    artifacts_dir = run_root / "artifacts"
    traces_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    policy_config = load_policy(policy_path if policy_path else config.policy_path)
    tasks = load_tasks(config.suite_path)
    rows = []
    server_processes: list[subprocess.Popen[bytes]] = []
    deadline = time.monotonic() + (timeout_seconds or config.run_timeout_seconds)
    runtime_registrations = configured_runtimes(config.runtimes)

    try:
        for transport in selected:
            base_url = None
            if transport == "http":
                port = _free_port()
                base_url = f"http://127.0.0.1:{port}"
                server_processes.append(_start_server(policy_path, port))
                _wait_for_health(base_url)

            for registration in runtime_registrations:
                _raise_if_stopped(deadline, cancel_event)
                runner = registration.runner_cls(
                    policy_config,
                    traces_dir,
                    transport_mode=transport,
                    base_url=base_url,
                    artifacts_dir=artifacts_dir,
                )
                try:
                    for task in tasks:
                        _raise_if_stopped(deadline, cancel_event)
                        rows.append(runner.run_task(task))
                finally:
                    runner.close()
    finally:
        for proc in server_processes:
            _stop_process(proc)

    command = "usb-agents run " + " ".join(f"--transport {item}" for item in selected)
    return create_run_from_results(rows, command=command, run_root=run_root)


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    _, port = sock.getsockname()
    sock.close()
    return int(port)


def _start_server(policy: Path, port: int) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mcp_server.server",
            "--transport",
            "http",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--policy",
            str(policy),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _wait_for_health(base_url: str) -> None:
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base_url}/health", timeout=1)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for {base_url}")


def _raise_if_stopped(deadline: float, cancel_event: threading.Event | None) -> None:
    if cancel_event and cancel_event.is_set():
        raise RuntimeError("Run canceled")
    if time.monotonic() > deadline:
        raise TimeoutError("Run timed out")


def _stop_process(proc: subprocess.Popen[bytes]) -> None:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
