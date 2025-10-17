"""Policy helpers for mapping MCP tools to approval rules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import yaml


@dataclass
class PolicyConfig:
    default: str
    tools: Dict[str, str]

    @classmethod
    def from_file(cls, path: Path) -> "PolicyConfig":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        default = data.get("default", "by_tool")
        tools = data.get("tools", {})
        return cls(default=default, tools=tools)

    def requirement_for(self, tool: str) -> str:
        return self.tools.get(tool, self.default)

    def requires_approval(self, tool: str) -> bool:
        requirement = self.requirement_for(tool)
        if requirement not in {"never", "by_tool", "always"}:
            raise ValueError(f"Unknown approval policy '{requirement}' for tool '{tool}'")
        return requirement in {"by_tool", "always"}


def load_policy(path: Path) -> PolicyConfig:
    return PolicyConfig.from_file(path)
