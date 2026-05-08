from usb_agents.redaction import REDACTED, redact
from usb_agents.runner_base import RunResult


def test_redacts_secret_shaped_values():
    payload = {
        "OPENAI_API_KEY": "sk-proj-1234567890abcdefghijkl",
        "nested": ["Bearer abcdefghijklmnopqrstuvwxyz"],
        "safe": "visible",
    }

    redacted = redact(payload)

    assert redacted["OPENAI_API_KEY"] == REDACTED
    assert redacted["nested"] == [f"Bearer {REDACTED}"]
    assert redacted["safe"] == "visible"


def test_run_result_row_redacts_metadata():
    result = RunResult(
        runtime="openai_agents",
        task_id="t1_repo_triage",
        policy_mode="never",
        success=True,
        latency_ms=1.0,
        tokens_prompt=1,
        tokens_completion=1,
        tool_calls=1,
        metadata={"token": "sk-1234567890abcdefghijkl"},
    )

    row = result.to_row()

    assert "sk-" not in row["metadata"]
    assert REDACTED in row["metadata"]
