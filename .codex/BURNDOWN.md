# USB Agents Frontier Burn-Down

Scoring model: 100 means a credible open-source release candidate for a local MCP portability
lab. The current score is deliberately harsh: a clean prototype foundation is not the same as a
frontier-grade tool.

## Scorecard

| Category | Current | Target | Gap | Release Gate |
| --- | ---: | ---: | ---: | --- |
| 1. Product clarity and positioning | 87 | 90 | 3 | README, PRODUCT, DESIGN, onboarding, examples tell one story. |
| 2. Installation and repo hygiene | 90 | 95 | 5 | Fresh clone installs with `uv sync --all-extras && pnpm install`. |
| 3. MCP protocol compliance | 84 | 95 | 11 | Stdio and HTTP pass protocol envelope, headers, errors, resources, prompts, and session tests. |
| 4. Runtime adapter depth | 66 | 90 | 24 | Simulator plus credentialed OpenAI/Microsoft/Mistral adapters behind one adapter interface. |
| 5. Transport adapter depth | 78 | 90 | 12 | Embedded, stdio, Streamable HTTP, pooling, lifecycle, and failure recovery covered. |
| 6. Hermetic execution and artifact model | 88 | 95 | 7 | Every run has isolated traces/artifacts, manifest, retrieval, and cleanup. |
| 7. Metrics and regression intelligence | 78 | 95 | 17 | Baselines, deltas, flaky categorization, trends, and release blockers are first-class. |
| 8. API and run lifecycle | 84 | 90 | 6 | API can launch runs, stream events, fetch artifacts, and expose history safely. |
| 9. Frontend product surface | 89 | 95 | 6 | Console handles live runs, diffs, drill-downs, empty/loading/error states, and docs. |
| 10. Visual verification | 84 | 90 | 6 | Deterministic screenshot and geometry suite is versioned and easy to update intentionally. |
| 11. Security and trust | 78 | 95 | 17 | Threat model, traversal tests, command/network allowlists, redaction, auth hooks, and audit logs. |
| 12. Observability and trace quality | 72 | 90 | 18 | Trace completeness, event schemas, OTLP export, and local inspection are reliable. |
| 13. CI and release engineering | 91 | 95 | 4 | GitHub Actions gates Python, frontend, visual, CLI sample, and packaging. |
| 14. Documentation and examples | 89 | 90 | 1 | Quickstart, architecture, MCP compliance, adapter authoring, and troubleshooting docs exist. |
| 15. Maintainer ergonomics | 86 | 90 | 4 | Burn-down, ADRs, CONTRIBUTING, issue templates, scripts, and typed seams guide contributors. |

Current weighted estimate after the latest slice: **86/100**.

## Execution Tranches

### Tranche 1: Make the foundation honest

- [x] Move Python to `src/` and fix editable packaging.
- [x] Add CLI, API, MCP JSON-RPC, web console, and deterministic tests.
- [x] Replace dummy API run creation with real run execution.
- [x] Fix artifact retrieval for per-run manifests.
- [x] Add event streaming endpoint.
- [x] Add CI workflow for all existing gates.
- [x] Add threat model and architecture docs.

### Tranche 2: Deepen the core seams

- [x] Extract runtime adapter registry and execution orchestrator.
- [x] Add transport adapter contract tests for embedded, stdio, and HTTP.
- [x] Add redaction module and trace/artifact secret regression tests.
- [x] Add baseline comparison API.
- [x] Add baseline comparison UI diff view.
- [x] Add run cancellation and timeout controls.
- [x] Add config file support for suites, adapters, thresholds, and network allowlists.

### Tranche 3: Make it release-shaped

- [x] Add vendor SDK readiness command with credential-gated failure mode.
- [ ] Add live real OpenAI Agents SDK execution path with credentialed integration tests.
- [x] Add Microsoft and Mistral adapter registry flags and credential-gated status tests.
- [x] Add GitHub issue templates, CONTRIBUTING, and SECURITY.
- [x] Add CODE_OF_CONDUCT.
- [x] Add Docker/devcontainer path for reproducible local demos.
- [x] Add docs site route or static docs export.
- [x] Add example MCP server fixtures and adapter authoring guide.

### Tranche 4: Push toward excellent

- [x] Live web console streaming for run progress and trace waterfall.
- [x] Regression diff UX with artifact side-by-side.
- [x] Performance budget for 10x tasks and 5x adapters.
- [x] Protocol conformance smoke against the official external MCP conformance package.
- [x] Golden demo run curated for screenshots, docs, and releases.
- [x] Release checklist and versioned changelog.

### Elite Hardening Backlog

- [x] Optional bearer-token auth for the local dashboard API.
- [x] Web API proxy forwards bearer auth when configured.
- [x] External MCP conformance command pins `@modelcontextprotocol/conformance@0.1.16`.
- [x] Side-by-side artifact diff API and console flow.
- [x] Golden release demo generator writes `release-demo.md`.
- [x] SDK readiness smoke reports installed package and credential state.
- [x] Commit `uv.lock` and `pnpm-lock.yaml` for reproducible installs.
- [x] Add Python package build gate and Docker build context hygiene.
- [x] Remove tracked generated metric archive files from the public tree.
- [x] Add frontend README and repository structure guide.
- [ ] Credentialed live OpenAI Agents SDK adapter run.
- [ ] Credentialed live Microsoft Agent Framework adapter run.
- [ ] Credentialed live Mistral Agents adapter run.
- [ ] OAuth or user identity layer for hosted/multi-user deployment.
- [ ] Full MCP advanced conformance for sampling, elicitation, subscriptions, binary resources,
  and prompt image responses if those capabilities become claimed.
- [ ] Trend persistence beyond run manifests for long-lived historical analytics.
- [ ] Flake classification across repeated runs.
- [ ] Release demo video/screenshot pack.
- [ ] Package publishing workflow once repository URL and package ownership are final.

## Active Slice

This session completed every locally verifiable release-shaping gap. Remaining gaps require live
vendor credentials, hosted identity decisions, or claiming advanced MCP capabilities that the
project does not currently expose.
