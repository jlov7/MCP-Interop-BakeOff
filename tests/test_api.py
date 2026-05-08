import time

from fastapi.testclient import TestClient

from usb_agents_api.app import app


def test_api_exposes_demo_run():
    client = TestClient(app)
    response = client.get("/api/runs")
    assert response.status_code == 200
    runs = response.json()["runs"]
    assert runs
    assert runs[0]["suite"]["cases"]


def test_api_optional_bearer_auth(monkeypatch):
    monkeypatch.setenv("USB_AGENTS_API_TOKEN", "local-secret")
    client = TestClient(app)

    rejected = client.get("/api/runs")
    accepted = client.get("/api/runs", headers={"authorization": "Bearer local-secret"})

    assert rejected.status_code == 401
    assert accepted.status_code == 200


def test_api_exposes_adapters():
    client = TestClient(app)
    response = client.get("/api/adapters")
    assert response.status_code == 200
    payload = response.json()
    assert {item["id"] for item in payload["transports"]} >= {"embedded", "http", "stdio"}
    assert {item["id"] for item in payload["runtimes"]} >= {"openai_agents", "mistral_agents"}


def test_api_can_create_run_and_fetch_manifest_artifact():
    client = TestClient(app)
    response = client.post("/api/runs", json={"transports": ["embedded"]})
    assert response.status_code == 200
    run = response.json()
    assert run["status"] == "passed"

    artifact = next(item for item in run["artifacts"] if item["path"].endswith("http_etl.csv"))
    artifact_response = client.get(f"/api/runs/{run['id']}/artifacts/{artifact['path']}")
    assert artifact_response.status_code == 200
    assert "id,name,role" in artifact_response.text


def test_api_can_create_background_run_job():
    client = TestClient(app)
    response = client.post("/api/run-jobs", json={"transports": ["embedded"]})
    assert response.status_code == 202
    job_id = response.json()["id"]

    for _ in range(100):
        status = client.get(f"/api/run-jobs/{job_id}").json()
        if status["status"] in {"passed", "failed", "canceled"}:
            break
        time.sleep(0.1)
    else:
        raise AssertionError("run job did not finish")

    assert status["status"] == "passed"
    assert status["run"]["artifacts"]


def test_api_can_cancel_background_run_job():
    client = TestClient(app)
    response = client.post("/api/run-jobs", json={"transports": ["embedded"]})
    job_id = response.json()["id"]

    canceled = client.delete(f"/api/run-jobs/{job_id}")

    assert canceled.status_code == 202
    assert canceled.json()["status"] in {"canceling", "canceled", "passed"}


def test_api_streams_run_events():
    client = TestClient(app)
    run = client.get("/api/runs").json()["runs"][0]
    response = client.get(f"/api/runs/{run['id']}/events/stream")
    assert response.status_code == 200
    assert "event: trace" in response.text
    assert "event: done" in response.text


def test_api_compares_runs():
    client = TestClient(app)
    run = client.get("/api/runs").json()["runs"][0]
    response = client.get(f"/api/runs/{run['id']}/compare/{run['id']}")
    assert response.status_code == 200
    comparison = response.json()
    assert comparison["deltas"]["latency_p95"] == 0
    assert comparison["regressions"] == []


def test_api_compares_artifacts():
    client = TestClient(app)
    run = client.get("/api/runs").json()["runs"][0]
    artifact = next(item for item in run["artifacts"] if item["kind"] in {"csv", "json", "trace"})

    response = client.get(
        f"/api/runs/{run['id']}/artifact-diff",
        params={"baseline_id": run["id"], "path": artifact["path"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["changed"] is False
    assert payload["stats"]["diff_lines"] == 0


def test_api_rejects_unknown_transport():
    client = TestClient(app)
    response = client.post("/api/runs", json={"transports": ["invalid-transport"]})
    assert response.status_code == 400


def test_api_rejects_artifact_path_not_in_manifest():
    client = TestClient(app)
    run = client.get("/api/runs").json()["runs"][0]
    response = client.get(f"/api/runs/{run['id']}/artifacts/../../pyproject.toml")
    assert response.status_code == 404
