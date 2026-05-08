# Security Policy

## Supported Versions

The project is pre-1.0. Security fixes should target `master` and active release branches once
release branches exist.

## Reporting a Vulnerability

Open a private security advisory if the repository is hosted on GitHub. If private advisories are
not available, open a minimal public issue that describes impact without exploit details and ask
for a maintainer contact path.

## Security Expectations

- No secrets in source, traces, screenshots, or artifacts.
- MCP tools must use schemas, allowlists, and explicit approval checks.
- Artifact retrieval must only serve paths listed in a run manifest.
- Command execution must stay allowlisted and run in temporary workspaces.
- Network access must stay allowlisted for deterministic tests.

See [docs/security-threat-model.md](docs/security-threat-model.md).
