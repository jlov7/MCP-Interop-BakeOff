"""FastAPI app that serves local usb-agents run data to the web console."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from usb_agents.models import Run
from usb_agents.run_executor import execute_run
from usb_agents.run_store import (
    compare_artifact,
    compare_runs,
    default_suite,
    get_run,
    list_runs,
    resolve_artifact_path,
    runtime_adapters,
    transport_adapters,
)

app = FastAPI(title="usb-agents local API")


@app.middleware("http")
async def optional_bearer_auth(request: Request, call_next):
    token = os.getenv("USB_AGENTS_API_TOKEN")
    if token and request.url.path != "/api/health":
        expected = f"Bearer {token}"
        if request.headers.get("authorization") != expected:
            return JSONResponse(
                {"detail": "Missing or invalid API token"},
                status_code=401,
            )
    return await call_next(request)


@dataclass
class RunJob:
    id: str
    request: "CreateRunRequest"
    cancel_event: threading.Event = field(default_factory=threading.Event)
    status: str = "queued"
    run: Run | None = None
    error: str | None = None
    events: list[dict[str, object]] = field(default_factory=list)


RUN_JOBS: dict[str, RunJob] = {}


class CreateRunRequest(BaseModel):
    transports: list[str] = Field(default_factory=lambda: ["embedded"])
    policy_path: str = "mcp-server/policy.yaml"
    config_path: str = "usb-agents.yaml"
    timeout_seconds: float | None = None


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "usb-agents-api"}


@app.get("/api/suites")
async def suites() -> dict[str, object]:
    return {"suites": [default_suite().model_dump()]}


@app.get("/api/adapters")
async def adapters() -> dict[str, object]:
    return {
        "runtimes": [adapter.model_dump() for adapter in runtime_adapters()],
        "transports": [adapter.model_dump() for adapter in transport_adapters()],
    }


@app.get("/api/runs")
async def runs() -> dict[str, object]:
    return {"runs": [run.model_dump(mode="json") for run in list_runs()]}


@app.post("/api/runs")
async def create_run(payload: CreateRunRequest | None = None) -> Run:
    request = payload or CreateRunRequest()
    try:
        return await asyncio.to_thread(
            execute_run,
            transports=request.transports,
            policy_path=Path(request.policy_path),
            config_path=Path(request.config_path),
            timeout_seconds=request.timeout_seconds,
        )
    except (TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/run-jobs", status_code=202)
async def create_run_job(payload: CreateRunRequest | None = None) -> dict[str, object]:
    request = payload or CreateRunRequest()
    job = RunJob(id=uuid4().hex, request=request)
    RUN_JOBS[job.id] = job
    job.events.append({"name": "job.queued", "message": "Run job queued."})
    threading.Thread(target=_execute_job, args=(job,), daemon=True).start()
    return _job_payload(job)


@app.get("/api/run-jobs/{job_id}")
async def run_job(job_id: str) -> dict[str, object]:
    job = RUN_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Run job not found")
    return _job_payload(job)


@app.delete("/api/run-jobs/{job_id}", status_code=202)
async def cancel_run_job(job_id: str) -> dict[str, object]:
    job = RUN_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Run job not found")
    job.cancel_event.set()
    if job.status in {"queued", "running"}:
        job.status = "canceling"
        job.events.append({"name": "job.canceling", "message": "Cancellation requested."})
    return _job_payload(job)


@app.get("/api/run-jobs/{job_id}/events/stream")
async def run_job_event_stream(job_id: str) -> StreamingResponse:
    job = RUN_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Run job not found")

    async def stream():
        seen = 0
        while True:
            while seen < len(job.events):
                yield f"event: job\ndata: {json.dumps(job.events[seen])}\n\n"
                seen += 1
            if job.status in {"passed", "failed", "canceled"}:
                yield f"event: done\ndata: {json.dumps(_job_payload(job))}\n\n"
                break
            await asyncio.sleep(0.1)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/runs/{run_id}")
async def run_detail(run_id: str) -> Run:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: str) -> dict[str, object]:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"events": [event.model_dump(mode="json") for event in run.events]}


@app.get("/api/runs/{run_id}/compare/{baseline_id}")
async def run_compare(run_id: str, baseline_id: str) -> dict[str, object]:
    comparison = compare_runs(run_id, baseline_id)
    if comparison is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return comparison


@app.get("/api/runs/{run_id}/artifact-diff")
async def run_artifact_diff(run_id: str, baseline_id: str, path: str) -> dict[str, object]:
    comparison = compare_artifact(run_id, baseline_id, path)
    if comparison is None:
        raise HTTPException(status_code=404, detail="Artifact comparison not found")
    return comparison


@app.get("/api/runs/{run_id}/events/stream")
async def run_event_stream(run_id: str) -> StreamingResponse:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    async def stream():
        for event in run.events:
            payload = json.dumps(event.model_dump(mode="json"))
            yield f"event: trace\ndata: {payload}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/runs/{run_id}/artifacts/{artifact_path:path}")
async def artifact(run_id: str, artifact_path: str) -> PlainTextResponse:
    target = resolve_artifact_path(run_id, artifact_path)
    if target is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return PlainTextResponse(target.read_text(encoding="utf-8"))


def _execute_job(job: RunJob) -> None:
    job.status = "running"
    job.events.append({"name": "job.running", "message": "Run job started."})
    try:
        run = execute_run(
            transports=job.request.transports,
            policy_path=Path(job.request.policy_path),
            config_path=Path(job.request.config_path),
            timeout_seconds=job.request.timeout_seconds,
            cancel_event=job.cancel_event,
        )
    except RuntimeError as exc:
        if job.cancel_event.is_set():
            job.status = "canceled"
            job.events.append({"name": "job.canceled", "message": str(exc)})
        else:
            job.status = "failed"
            job.error = str(exc)
            job.events.append({"name": "job.failed", "message": str(exc)})
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        job.events.append({"name": "job.failed", "message": str(exc)})
    else:
        job.run = run
        job.status = run.status
        job.events.append({"name": "job.completed", "message": f"Run {run.id} completed."})


def _job_payload(job: RunJob) -> dict[str, object]:
    return {
        "id": job.id,
        "status": job.status,
        "error": job.error,
        "run_id": job.run.id if job.run else None,
        "run": job.run.model_dump(mode="json") if job.run else None,
        "events": list(job.events),
    }
