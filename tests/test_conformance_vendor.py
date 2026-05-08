from usb_agents.conformance import conformance_command
from usb_agents.vendor_smoke import check_all_vendor_sdks


def test_conformance_command_targets_official_server_runner():
    command = conformance_command(
        package="@modelcontextprotocol/conformance@0.1.16",
        url="http://127.0.0.1:9010/mcp",
        scenario="server-initialize",
    )

    assert command[:5] == [
        "npx",
        "-y",
        "@modelcontextprotocol/conformance@0.1.16",
        "server",
        "--url",
    ]
    assert "--scenario" in command
    assert "server-initialize" in command


def test_vendor_smoke_reports_all_runtime_sdks():
    results = check_all_vendor_sdks()

    assert {result.runtime for result in results} == {
        "openai_agents",
        "microsoft_agent_framework",
        "mistral_agents",
    }
    assert all(result.module for result in results)
