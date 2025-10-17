"""Task specification helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml


@dataclass
class TaskSpec:
    task_id: str
    name: str
    description: str
    runner_prompt: str
    approval: str
    expected_artifact: Optional[str]
    mode: str
    path: Path

    @classmethod
    def from_path(cls, path: Path) -> "TaskSpec":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            task_id=data["id"],
            name=data["name"],
            description=data["description"],
            runner_prompt=data["runner_prompt"],
            approval=data.get("approval", "by_tool"),
            expected_artifact=data.get("expected_artifact"),
            mode=data.get("mode", "read"),
            path=path,
        )


def load_tasks(task_dir: Path) -> List[TaskSpec]:
    specs: List[TaskSpec] = []
    for path in sorted(task_dir.glob("t*.yaml")):
        specs.append(TaskSpec.from_path(path))
    return specs
