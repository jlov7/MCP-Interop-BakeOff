# USB Agents Frontier Revamp

## Purpose / Big Picture

Turn usb-agents from a compact MCP bake-off harness into an OSS-first MCP portability lab. Success means the repo installs cleanly, exposes a real local CLI/API/MCP surface, runs hermetic benchmark tasks without mutating tracked fixtures, and ships a polished local web console for comparing runtimes, transports, traces, approvals, and regressions.

## Progress

- [x] Create `PRODUCT.md` and `DESIGN.md` context for Impeccable-guided product work.
- [x] Move installable Python packages into `src/`.
- [x] Fix packaging and existing quality gate failures.
- [x] Add structured Pydantic models and hermetic artifact storage.
- [x] Add MCP JSON-RPC stdio and Streamable HTTP-compatible `/mcp` surface.
- [x] Add local FastAPI dashboard API and Typer CLI.
- [x] Add a Next.js local console in `apps/web`.
- [x] Add regression, protocol, API, and visual verification tests.

## Surprises & Discoveries

- Date: 2026-05-08
  Discovery: Normal `uv run pytest`, `ruff`, and `mypy` fail at editable package build because setuptools auto-discovers generated/data directories in flat layout.
  Impact: Package layout must be fixed before other gates are meaningful.

- Date: 2026-05-08
  Discovery: Bypassing project install showed tests pass but `ruff` has 23 issues and `mypy` cannot resolve package bases.
  Impact: The revamp needs packaging, lint, and import topology as first-order work.

- Date: 2026-05-08
  Discovery: Stdio transport command strings break when the Python executable path contains spaces.
  Impact: The stdio smoke test now quotes the executable, and the transport path stays covered.

- Date: 2026-05-08
  Discovery: CLI runs generated artifacts correctly but did not include them in the run manifest.
  Impact: The run store now discovers trace and artifact files under each run root.

- Date: 2026-05-08
  Discovery: The first implementation was a foundation slice, not a release candidate.
  Impact: Added a 15-category burn-down and raised the active score from 46/100 to 58/100 with API execution, event streaming, artifact retrieval, CI, docs, web run launcher, and redaction.

- Date: 2026-05-08
  Discovery: Parallel CLI runs collided because run roots used second-resolution timestamps.
  Impact: Run roots now use the same UUID-suffixed ID generator as run records, and the regression is covered by a unique-run-root test.

- Date: 2026-05-08
  Discovery: Visual snapshots were consuming live API data, so a new run changed screenshots.
  Impact: Playwright routes through `/?fixture=1` for deterministic screenshots while production usage still loads live runs.

- Date: 2026-05-08
  Discovery: The official MCP conformance package is available via `npx @modelcontextprotocol/conformance`.
  Impact: Added `usb-agents conformance` and CI wiring for the core claimed server scenarios.

## Decision Log

- Date: 2026-05-08
  Decision: Build v1 as CLI plus local web console, not a hosted SaaS.
  Rationale: OSS adoption and shareable demos matter first; enterprise readiness comes from architecture, reproducibility, and security posture.
  Alternatives considered: Hosted platform, library-only.

- Date: 2026-05-08
  Decision: Use a product-register “signal cockpit” design system.
  Rationale: The tool is a developer workbench, so trust, density, and fast scanning matter more than marketing theatrics.
  Alternatives considered: Minimal bright infrastructure, docs-first hybrid.

## Outcomes & Retrospective

- Completed: Installable `src/` Python package, MCP JSON-RPC server, local API, Typer CLI, hermetic artifacts, typed run models, Next.js console, backend/frontend verification, API-triggered jobs, event streaming, cancellation, timeout controls, adapter registry/config, Streamable HTTP session basics, CI workflow, release docs, redaction, Docker/devcontainer, deterministic visual fixtures, external conformance smoke, SDK readiness smoke, artifact side-by-side diffs, optional local API bearer auth, and golden demo generation.
- Deferred: Hosted SaaS, real cloud credentials for non-simulated runtime adapters, advanced MCP scenarios for capabilities not currently claimed, long-horizon trend store, flake classification, and release media.
- Risks left: Real vendor SDK adapters still need credentialed integration tests and hosted deployments need a real identity layer beyond the local bearer-token gate.
- Follow-ups: Add live OpenAI/Microsoft/Mistral SDK paths behind the registry when credentials and exact provider targets are available, then add trend/flake intelligence and release screenshots/video.

## Verification Evidence

- Commands run: `uv sync --all-extras`, `uv run usb-agents doctor`, `uv run usb-agents run --transport embedded`, `uv run usb-agents run --transport http`, `uv run usb-agents run --transport stdio`.
- Tests run: `pnpm verify` (`uv run ruff check .`, `uv run mypy .`, `uv run pytest -q`, `pnpm lint`, `pnpm --filter @usb-agents/web build`, `pnpm verify:visual`).
- Manual checks: Opened `http://127.0.0.1:3000` in the in-app browser and captured a full-page screenshot after fixing the favicon diagnostic.
