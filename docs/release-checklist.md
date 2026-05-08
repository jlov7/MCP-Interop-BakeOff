# Release Checklist

## Preflight

- [ ] `uv sync --all-extras`
- [ ] `pnpm install`
- [ ] `uv run ruff check .`
- [ ] `uv run mypy .`
- [ ] `uv run pytest`
- [ ] `pnpm --filter @usb-agents/web lint`
- [ ] `pnpm --filter @usb-agents/web test`
- [ ] `pnpm verify:conformance`
- [ ] `uv run usb-agents doctor`
- [ ] `uv run usb-agents run --transport embedded --transport http --transport stdio`
- [ ] `uv run usb-agents sdk-smoke`
- [ ] `uv run usb-agents golden-demo`

## Product

- [ ] README quickstart works on a fresh clone.
- [ ] Console screenshots are regenerated intentionally.
- [ ] `PRODUCT.md`, `DESIGN.md`, and docs agree on product positioning.
- [ ] Demo run artifacts live under `.usb-agents/runs/`, not tracked fixture paths.
- [ ] Golden demo summary exists at `.usb-agents/runs/golden_release_demo/release-demo.md`.

## Security

- [ ] `SECURITY.md` contact is correct.
- [ ] Secret redaction regression tests pass.
- [ ] Path traversal and origin rejection tests pass.
- [ ] No `.env`, credentials, node_modules, or generated run directories are staged.

## Versioning

- [ ] Update `CHANGELOG.md`.
- [ ] Tag version after all gates pass.
- [ ] Publish package artifacts only from a clean working tree.
