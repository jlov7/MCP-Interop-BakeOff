import json
import socket
import subprocess
import sys
from pathlib import Path

from scripts.run_matrix import start_http_server, wait_for_server
from scripts.trace_summary import analyse_trace

from runners.mistral_runner import MistralRunner
from runners.ms_runner import MicrosoftRunner
from runners.openai_runner import OpenAIRunner
from usb_agents.policy import load_policy
from usb_agents.tasks import TaskSpec


TASK_PATH = Path("tasks/t1_repo_triage.yaml")


def load_task() -> TaskSpec:
    return TaskSpec.from_path(TASK_PATH)


def test_openai_runner_smoke(tmp_path):
    policy = load_policy(Path("mcp-server/policy.yaml"))
    runner = OpenAIRunner(policy, tmp_path)
    result = runner.run_task(load_task())
    assert result.success
    expected = json.loads(Path("tasks/artifacts/t1_repo_triage_expected.json").read_text(encoding="utf-8"))
    assert result.metadata["output"] == expected
    assert "tool_metrics" in result.metadata
    fs_metrics = result.metadata["tool_metrics"]["fs_read"]
    assert fs_metrics["count"] >= 1
    assert fs_metrics["median_ms"] > 0


def test_ms_runner_smoke(tmp_path):
    policy = load_policy(Path("mcp-server/policy.yaml"))
    runner = MicrosoftRunner(policy, tmp_path)
    result = runner.run_task(load_task())
    assert result.success


def test_mistral_runner_smoke(tmp_path):
    policy = load_policy(Path("mcp-server/policy.yaml"))
    runner = MistralRunner(policy, tmp_path)
    result = runner.run_task(load_task())
    assert result.success


def _get_free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    _, port = sock.getsockname()
    sock.close()
    return port


def test_openai_runner_http_transport(tmp_path):
    policy_path = Path("mcp-server/policy.yaml")
    port = _get_free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = start_http_server(policy_path, "127.0.0.1", port)
    try:
        wait_for_server(base_url)
        policy = load_policy(policy_path)
        runner = OpenAIRunner(policy, tmp_path, transport_mode="http", base_url=base_url)
        result = runner.run_task(load_task())
        runner.close()
        assert result.success
        assert result.transport == "http"
        assert "tool_metrics" in result.metadata
    finally:
        proc.terminate()
        try:
            stdout_data, stderr_data = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout_data, stderr_data = proc.communicate(timeout=5)
        else:
            _ = (stdout_data, stderr_data)


def test_trace_completeness_embedded(tmp_path):
    policy = load_policy(Path("mcp-server/policy.yaml"))
    runner = OpenAIRunner(policy, tmp_path)
    result = runner.run_task(load_task())
    runner.close()
    assert result.trace_completeness == 1.0
    assert result.trace_path is not None
    trace_path = Path(result.trace_path)
    assert trace_path.exists()
    summary = analyse_trace(trace_path)
    assert summary["completeness"] == 1.0
    assert not summary["missing"]


def test_stdio_transport_smoke(tmp_path):
    policy = load_policy(Path("mcp-server/policy.yaml"))
    cmd = (
        f"{sys.executable} -m mcp_server.server --transport stdio --policy mcp-server/policy.yaml"
    )
    runner = OpenAIRunner(policy, tmp_path, transport_mode="stdio", stdio_cmd=cmd)
    result = runner.run_task(load_task())
    runner.close()
    assert result.success
    assert result.transport == "stdio"
