# usb-agents Web Console

The web console is the local product surface for usb-agents. It is intentionally an operator
console, not a marketing page: the first screen shows run status, runtime x transport coverage,
trace evidence, approvals, artifact inspection, and regression comparison.

## Run Locally

Start the API from the repository root:

```bash
uv run usb-agents serve --host 127.0.0.1 --port 8765
```

Start the console:

```bash
pnpm --filter @usb-agents/web dev --hostname 127.0.0.1 --port 3000
```

Open [http://127.0.0.1:3000](http://127.0.0.1:3000).

## Verification

```bash
pnpm --filter @usb-agents/web lint
pnpm --filter @usb-agents/web build
pnpm --filter @usb-agents/web test
```

The Playwright suite uses `/?fixture=1` for screenshot tests so live run IDs and latency values do
not make visual snapshots flaky. The default route still reads live API data.

## Design Rules

- Keep the UI dense, calm, and operational.
- Avoid landing-page composition; this is a workbench.
- Prefer stable dimensions for matrix cells, toolbars, counters, and artifact panes.
- Preserve reduced-motion behavior and screenshot determinism.
- Add a test whenever a view, interaction, or layout constraint changes.
