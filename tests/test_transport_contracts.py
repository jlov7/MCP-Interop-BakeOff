from pathlib import Path

import pytest

from usb_agents.run_executor import execute_run


@pytest.mark.parametrize("transport", ["embedded", "stdio", "http"])
def test_transport_contract_completes_default_suite(tmp_path, transport):
    run = execute_run(
        transports=[transport],
        policy_path=Path("mcp-server/policy.yaml"),
        run_root=tmp_path / transport,
        timeout_seconds=30,
    )

    assert run.status == "passed"
    assert {result.transport for result in run.results} == {transport}
    assert all(result.trace_path for result in run.results)
    assert all(result.metadata.get("tool_metrics") for result in run.results)


def test_default_run_roots_are_unique():
    first = execute_run(transports=["embedded"], timeout_seconds=30)
    second = execute_run(transports=["embedded"], timeout_seconds=30)

    first_root = Path(first.results[0].trace_path).parents[1]
    second_root = Path(second.results[0].trace_path).parents[1]

    assert first_root != second_root
