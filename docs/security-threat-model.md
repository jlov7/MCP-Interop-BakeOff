# Security Threat Model

## Assets

- Runtime credentials in the environment.
- Task fixtures and source files in the repository.
- Generated traces and artifacts under `.usb-agents/runs/`.
- Local browser/API access to run data.

## Trust Boundaries

- Browser to local API.
- Runtime adapter to MCP transport.
- MCP server to filesystem, network, and subprocess tools.
- Run artifact reader to local filesystem.

## Required Controls

- Tool schemas reject unknown input fields.
- Filesystem reads are allowlisted.
- Artifact retrieval only resolves paths present in the run manifest.
- Command execution uses an allowlist and temporary workspace.
- HTTP fetching uses an allowlist for deterministic fixtures.
- Origin validation rejects cross-site browser calls to `/mcp`.
- `USB_AGENTS_API_TOKEN` enables bearer-token protection for the local dashboard API.
- Traces and artifacts must redact obvious secret-shaped values before release.

## Open Risks

- Real runtime SDK integrations need credential-scoped tests.
- Hosted or multi-user deployments still need a real identity layer beyond the local bearer-token
  gate.
- The external MCP conformance smoke now covers core claimed capabilities; unsupported advanced
  scenarios such as elicitation, subscriptions, binary resources, and sampling remain out of scope.
