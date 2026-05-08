## Current Task

Frontier revamp implementation for usb-agents: installable Python package, MCP-compatible backend, local API, CLI, web console, burn-down closure, and release scaffolding.

## Status

Completed

## Plan

1. [x] Create project context files and implementation tracker.
2. [x] Move installable Python packages into `src/`.
3. [x] Fix Python packaging, lint, and typing blockers.
4. [x] Add structured models, hermetic run artifacts, MCP JSON-RPC surface, API, and CLI.
5. [x] Build the local Next.js console.
6. [x] Add backend/frontend tests and run quality gates.
7. [x] Add runtime config/registry, job lifecycle, cancellation, baseline UI, docs, Docker, and release scaffolding.

## Decisions Made

- Product direction is OSS-first CLI plus local browser console.
- Primary audience is AI tool builders, with enterprise-grade trust as the quality bar.
- Frontend visual lane is a dark signal cockpit: dense, precise, restrained, and code-native.
- Generated run artifacts live under `.usb-agents/runs/`, not tracked fixture files.
- Playwright screenshots use `?fixture=1` so visual tests stay deterministic while the app uses live API data by default.
- Run roots include a UUID suffix to prevent artifact mixing when transports run concurrently.
- Official MCP conformance smoke uses `@modelcontextprotocol/conformance@0.1.16`.
- Local API auth is opt-in via `USB_AGENTS_API_TOKEN`.

## Open Questions

- None blocking.

## Verification

- `uv run ruff check .`
- `uv run mypy .`
- `uv run pytest -q`
- `pnpm lint`
- `pnpm --filter @usb-agents/web build`
- `pnpm test`
- `uv run usb-agents run --transport embedded`
- `uv run usb-agents run --transport http`
- `uv run usb-agents run --transport stdio`

## Burn-Down Status

- Current estimate: 86/100.
- Completed this slice: adapter registry/config, Streamable HTTP session lifecycle basics, API run jobs, cancellation, timeout controls, live UI streaming, comparison UI, docs route, Docker/devcontainer, release checklist, changelog, transport contract tests, and performance budget test.
- Completed follow-up slice: external MCP conformance command, SDK readiness smoke, golden demo generator, artifact side-by-side diffs, and optional dashboard API bearer auth.
- Completed public-readiness slice: actual GitHub package metadata, committed lockfiles, Python build gate, Docker ignore hygiene, frontend README, repo structure guide, and removal of tracked generated metric archives.
- Highest remaining gaps: live credentialed vendor SDK execution, hosted identity, advanced MCP capabilities not currently claimed, long-horizon trends, flake classification, and release media.
