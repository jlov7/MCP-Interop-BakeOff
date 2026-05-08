# Contributing

usb-agents aims to be a release-quality MCP portability lab. Contributions should improve
portability evidence, protocol correctness, security, visual quality, or maintainer ergonomics.

## Development Setup

```bash
uv sync --all-extras
pnpm install
uv run usb-agents doctor
```

## Quality Gates

Run these before opening a pull request:

```bash
uv run ruff check .
uv run mypy .
uv run pytest -q
pnpm lint
pnpm --filter @usb-agents/web build
pnpm verify:visual
uv run usb-agents run --transport embedded
```

## Working Rules

- Keep generated files under `.usb-agents/runs/`.
- Do not mutate tracked task fixtures during benchmark execution.
- Add tests for protocol changes, security controls, and regression metrics.
- Update `.codex/BURNDOWN.md` when completing or adding release gates.
- Commit screenshot baselines only when a visual change is intentional.

## Adapter Contributions

Runtime adapters should hide SDK-specific orchestration behind a small interface. The run executor
and tests should not know credential loading, tool registration internals, approval callback
plumbing, or trace export details for a specific vendor.
