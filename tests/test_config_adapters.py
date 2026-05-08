from pathlib import Path

from usb_agents.adapters import configured_runtimes, runtime_adapters
from usb_agents.config import RuntimeConfig, load_config


def test_loads_project_config(tmp_path):
    config_path = tmp_path / "usb-agents.yaml"
    config_path.write_text(
        """
suite_path: tasks
policy_path: mcp-server/policy.yaml
default_transports: [embedded, stdio]
run_timeout_seconds: 42
runtimes:
  - id: openai_agents
    mode: simulator
    enabled: true
  - id: mistral_agents
    mode: sdk
    enabled: false
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.default_transports == ["embedded", "stdio"]
    assert config.run_timeout_seconds == 42
    assert len(config.runtimes) == 2


def test_runtime_registry_respects_enabled_config():
    config = load_config(Path("usb-agents.yaml"))
    enabled = configured_runtimes(config.runtimes)
    adapters = runtime_adapters(config.runtimes)

    assert {item.id for item in enabled} >= {"openai_agents", "mistral_agents"}
    assert {item.status for item in adapters} <= {
        "available",
        "simulated",
        "unconfigured",
        "disabled",
    }


def test_sdk_runtime_is_credential_gated(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    missing = runtime_adapters([RuntimeConfig(id="openai_agents", mode="sdk")])
    assert next(item for item in missing if item.id == "openai_agents").status == "unconfigured"

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    available = runtime_adapters([RuntimeConfig(id="openai_agents", mode="sdk")])
    assert next(item for item in available if item.id == "openai_agents").status == "available"
