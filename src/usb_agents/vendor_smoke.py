"""Credential-gated vendor SDK readiness checks."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass

from usb_agents.adapters import RUNTIME_REGISTRY

SDK_MODULES = {
    "openai_agents": "openai",
    "microsoft_agent_framework": "azure.core",
    "mistral_agents": "mistralai",
}


@dataclass(frozen=True)
class VendorSmokeResult:
    runtime: str
    module: str
    credential_env: str | None
    installed: bool
    credential_present: bool
    ready: bool


def check_vendor_sdk(runtime: str) -> VendorSmokeResult:
    registration = RUNTIME_REGISTRY[runtime]
    module = SDK_MODULES[runtime]
    installed = importlib.util.find_spec(module) is not None
    credential_present = bool(
        registration.credential_env and os.getenv(registration.credential_env)
    )
    return VendorSmokeResult(
        runtime=runtime,
        module=module,
        credential_env=registration.credential_env,
        installed=installed,
        credential_present=credential_present,
        ready=installed and credential_present,
    )


def check_all_vendor_sdks() -> list[VendorSmokeResult]:
    return [check_vendor_sdk(runtime) for runtime in SDK_MODULES]
