"""Command line interface for usb-agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from usb_agents.conformance import DEFAULT_SCENARIOS, run_conformance
from usb_agents.golden import create_golden_demo
from usb_agents.run_executor import execute_run
from usb_agents.run_store import (
    RUNS_ROOT,
    default_suite,
    ensure_runs_root,
    list_runs,
    runtime_adapters,
    transport_adapters,
)
from usb_agents.vendor_smoke import check_all_vendor_sdks, check_vendor_sdk

app = typer.Typer(help="MCP portability lab for agent runtimes.")


@app.command()
def init() -> None:
    """Create local runtime directories and print the configured suite."""
    ensure_runs_root()
    typer.echo(f"Initialized {RUNS_ROOT}")
    typer.echo(default_suite().model_dump_json(indent=2))


@app.command()
def doctor() -> None:
    """Check local project prerequisites."""
    checks = {
        "tasks": Path("tasks").exists(),
        "policy": Path("mcp-server/policy.yaml").exists(),
        "product_context": Path("PRODUCT.md").exists(),
        "design_context": Path("DESIGN.md").exists(),
        "runs_root": ensure_runs_root().exists(),
    }
    for name, passed in checks.items():
        status = "ok" if passed else "missing"
        typer.echo(f"{name}: {status}")
    if not all(checks.values()):
        raise typer.Exit(1)


@app.command()
def run(
    transports: Annotated[
        list[str],
        typer.Option("--transport", "--transports", help="Transport to run. Repeatable."),
    ] = ["embedded"],
    policy: Annotated[Path, typer.Option(help="Path to policy YAML.")] = Path(
        "mcp-server/policy.yaml"
    ),
    config: Annotated[Path, typer.Option(help="Path to usb-agents YAML config.")] = Path(
        "usb-agents.yaml"
    ),
    timeout: Annotated[float | None, typer.Option(help="Run timeout in seconds.")] = None,
) -> None:
    """Run the deterministic benchmark suite and write a local run artifact."""
    try:
        run_model = execute_run(
            transports=transports,
            policy_path=policy,
            config_path=config,
            timeout_seconds=timeout,
        )
    except (TimeoutError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(run_model.model_dump_json(indent=2))


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="API host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="API port.")] = 8765,
) -> None:
    """Serve the local dashboard API."""
    import uvicorn

    uvicorn.run("usb_agents_api.app:app", host=host, port=port, reload=False)


@app.command()
def report() -> None:
    """Print known local runs."""
    typer.echo(json.dumps([run.model_dump(mode="json") for run in list_runs()], indent=2))


@app.command()
def compare() -> None:
    """Print adapter and suite inventory for comparison setup."""
    payload = {
        "suite": default_suite().model_dump(mode="json"),
        "runtimes": [adapter.model_dump(mode="json") for adapter in runtime_adapters()],
        "transports": [adapter.model_dump(mode="json") for adapter in transport_adapters()],
    }
    typer.echo(json.dumps(payload, indent=2))


@app.command()
def conformance(
    url: Annotated[str | None, typer.Option(help="Existing MCP server /mcp URL.")] = None,
    scenario: Annotated[
        list[str] | None,
        typer.Option("--scenario", help="MCP conformance scenario. Repeatable."),
    ] = None,
    port: Annotated[int, typer.Option(help="Local MCP server port when --url is omitted.")] = 9010,
    policy: Annotated[Path, typer.Option(help="Path to policy YAML.")] = Path(
        "mcp-server/policy.yaml"
    ),
) -> None:
    """Run the official MCP conformance framework against the local MCP server."""
    scenarios = tuple(scenario or DEFAULT_SCENARIOS)
    results = run_conformance(url=url, port=port, policy_path=policy, scenarios=scenarios)
    for result in results:
        typer.echo(result.stdout.strip())
        if result.stderr.strip():
            typer.echo(result.stderr.strip(), err=True)
    if not all(result.passed for result in results):
        raise typer.Exit(1)


@app.command("sdk-smoke")
def sdk_smoke(
    runtime: Annotated[
        str | None,
        typer.Option(help="Runtime id to check. Omit for all vendor SDKs."),
    ] = None,
    require: Annotated[bool, typer.Option(help="Fail when credentials are absent.")] = False,
) -> None:
    """Check installed vendor SDKs and credential readiness without sending live requests."""
    results = [check_vendor_sdk(runtime)] if runtime else check_all_vendor_sdks()
    payload = [result.__dict__ for result in results]
    typer.echo(json.dumps(payload, indent=2))
    if require and not all(result.ready for result in results):
        raise typer.Exit(1)


@app.command("golden-demo")
def golden_demo(
    transports: Annotated[
        list[str],
        typer.Option("--transport", "--transports", help="Transport to include. Repeatable."),
    ] = ["embedded", "http", "stdio"],
    output: Annotated[
        Path | None,
        typer.Option(
            help="Run root to overwrite. Defaults to .usb-agents/runs/golden_release_demo."
        ),
    ] = None,
    timeout: Annotated[float | None, typer.Option(help="Run timeout in seconds.")] = 120,
) -> None:
    """Create the curated local release demo run."""
    run_model = create_golden_demo(
        transports=transports,
        run_root=output,
        timeout_seconds=timeout,
    )
    typer.echo(run_model.model_dump_json(indent=2))
