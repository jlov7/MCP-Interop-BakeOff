"""Project configuration loading for usb-agents."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class RuntimeConfig(BaseModel):
    id: str
    mode: Literal["simulator", "sdk"] = "simulator"
    enabled: bool = True


class UsbAgentsConfig(BaseModel):
    suite_path: Path = Path("tasks")
    policy_path: Path = Path("mcp-server/policy.yaml")
    default_transports: list[str] = Field(default_factory=lambda: ["embedded"])
    run_timeout_seconds: float = 120.0
    task_timeout_seconds: float = 30.0
    network_allowlist: list[str] = Field(
        default_factory=lambda: ["https://example.com/synthetic-dataset"]
    )
    runtimes: list[RuntimeConfig] = Field(
        default_factory=lambda: [
            RuntimeConfig(id="openai_agents"),
            RuntimeConfig(id="microsoft_agent_framework"),
            RuntimeConfig(id="mistral_agents"),
        ]
    )


def load_config(path: Path | None = None) -> UsbAgentsConfig:
    config_path = path or Path("usb-agents.yaml")
    if not config_path.exists():
        return UsbAgentsConfig()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return UsbAgentsConfig.model_validate(data)
