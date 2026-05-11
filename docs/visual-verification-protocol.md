# Visual Verification Protocol

Version: 2026-05-08

The web console is a visual-critical surface. Every UI change should preserve these gates:

1. Run `pnpm verify:visual`.
2. Keep Playwright desktop and mobile projects enabled.
3. Wait for `window.__USB_AGENTS_READY === true` before assertions.
4. Use `window.__USB_AGENTS_DEBUG__()` for machine-readable run state.
5. Assert geometry for panel widths, overlap area, and horizontal overflow.
6. Capture deterministic screenshots with `prefers-reduced-motion: reduce` and disabled animations.
7. Keep screenshot diff budgets explicit and narrow. The console allows `maxDiffPixelRatio: 0.025`
   for Linux/macOS text rasterization drift; layout regressions still fail through geometry assertions.
8. Commit screenshot baselines only when the visual change is intentional.

Current baseline locations:

- `apps/web/tests/console.spec.ts`
- `apps/web/tests/console.spec.ts-snapshots/`
