"""Runtime adapter registry."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Type

from runners.mistral_runner import MistralRunner
from runners.ms_runner import MicrosoftRunner
from runners.openai_runner import OpenAIRunner
from usb_agents.config import RuntimeConfig
from usb_agents.models import RuntimeAdapter
from usb_agents.runner_base import BaseRunner


@dataclass(frozen=True)
class RuntimeRegistration:
    id: str
    name: str
    model: str
    runner_cls: Type[BaseRunner]
    credential_env: str | None = None

    def adapter(self, config: RuntimeConfig | None = None) -> RuntimeAdapter:
        mode = config.mode if config else "simulator"
        enabled = config.enabled if config else True
        status: Literal["available", "simulated", "unconfigured", "disabled"]
        if not enabled:
            status = "disabled"
        elif mode == "sdk" and self.credential_env and os.getenv(self.credential_env):
            status = "available"
        elif mode == "sdk":
            status = "unconfigured"
        else:
            status = "simulated"
        return RuntimeAdapter(
            id=self.id,
            name=self.name,
            model=self.model,
            status=status,
            mode=mode,
            credential_env=self.credential_env,
        )


RUNTIME_REGISTRY: dict[str, RuntimeRegistration] = {
    "openai_agents": RuntimeRegistration(
        id="openai_agents",
        name="OpenAI Agents",
        model="gpt-4.1-mini",
        runner_cls=OpenAIRunner,
        credential_env="OPENAI_API_KEY",
    ),
    "microsoft_agent_framework": RuntimeRegistration(
        id="microsoft_agent_framework",
        name="Microsoft Agent Framework",
        model="azure-gpt-4o-mini",
        runner_cls=MicrosoftRunner,
        credential_env="MS_AGENT_KEY",
    ),
    "mistral_agents": RuntimeRegistration(
        id="mistral_agents",
        name="Mistral Agents",
        model="mistral-large-latest",
        runner_cls=MistralRunner,
        credential_env="MISTRAL_API_KEY",
    ),
}


def configured_runtimes(configs: list[RuntimeConfig]) -> list[RuntimeRegistration]:
    selected = []
    for runtime_config in configs:
        registration = RUNTIME_REGISTRY.get(runtime_config.id)
        if registration and runtime_config.enabled:
            selected.append(registration)
    return selected


def runtime_adapters(configs: list[RuntimeConfig] | None = None) -> list[RuntimeAdapter]:
    config_by_id = {item.id: item for item in configs or []}
    return [
        registration.adapter(config_by_id.get(registration.id))
        for registration in RUNTIME_REGISTRY.values()
    ]
