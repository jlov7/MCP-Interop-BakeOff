from pathlib import Path

from runners.openai_runner import OpenAIRunner
from usb_agents.policy import load_policy
from usb_agents.run_store import create_run_from_results
from usb_agents.tasks import TaskSpec


def test_write_tasks_do_not_mutate_tracked_fixtures(tmp_path):
    policy = load_policy(Path("mcp-server/policy.yaml"))
    runner = OpenAIRunner(policy, tmp_path / "traces", artifacts_dir=tmp_path / "artifacts")
    example_path = Path("tasks/code_module/example.py")
    etl_path = Path("tasks/artifacts/http_etl.csv")
    before_example = example_path.read_text(encoding="utf-8")
    before_etl = etl_path.read_text(encoding="utf-8")
    try:
        runner.run_task(TaskSpec.from_path(Path("tasks/t3_http_etl.yaml")))
        runner.run_task(TaskSpec.from_path(Path("tasks/t4_code_patch.yaml")))
    finally:
        runner.close()

    assert example_path.read_text(encoding="utf-8") == before_example
    assert etl_path.read_text(encoding="utf-8") == before_etl
    assert list((tmp_path / "artifacts").glob("*http_etl.csv"))


def test_run_manifest_discovers_generated_artifacts(tmp_path):
    policy = load_policy(Path("mcp-server/policy.yaml"))
    run_root = tmp_path / "run"
    runner = OpenAIRunner(policy, run_root / "traces", artifacts_dir=run_root / "artifacts")
    try:
        result = runner.run_task(TaskSpec.from_path(Path("tasks/t3_http_etl.yaml")))
    finally:
        runner.close()

    run = create_run_from_results(
        [result], command="usb-agents run --transport embedded", run_root=run_root
    )

    artifact_paths = {artifact.path for artifact in run.artifacts}
    assert any(path.endswith("http_etl.csv") for path in artifact_paths)
    assert any(path.endswith(".ndjson") for path in artifact_paths)
