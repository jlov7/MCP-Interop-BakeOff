"""Secret redaction helpers for traces, manifests, and API payloads."""

from __future__ import annotations

import re
from typing import Any

SECRET_KEY_PARTS = ("api_key", "apikey", "authorization", "password", "secret", "token")
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"sk-proj-[A-Za-z0-9_-]{12,}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{12,}"),
]
REDACTED = "[REDACTED]"


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: REDACTED if _secret_key(key) else redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    if isinstance(value, str):
        redacted = value
        for pattern in SECRET_PATTERNS:
            redacted = pattern.sub(_replace_secret, redacted)
        return redacted
    return value


def _secret_key(key: object) -> bool:
    key_text = str(key).lower().replace("-", "_")
    return any(part in key_text for part in SECRET_KEY_PARTS)


def _replace_secret(match: re.Match[str]) -> str:
    if match.lastindex:
        return f"{match.group(1)}{REDACTED}"
    return REDACTED
