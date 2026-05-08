# PRODUCT.md

## Product

usb-agents is an OSS-first MCP portability lab. It helps AI tool builders prove that a tool server behaves consistently across agent runtimes, transports, approval policies, and telemetry backends before they ship it to users.

## Register

product

## Primary Users

- AI tool builders evaluating MCP servers and agent runtimes.
- Developer platform teams that need repeatable CI evidence for tool safety and compatibility.
- OSS researchers comparing model/tool behavior across runtimes.

## Product Purpose

The product should make portability risk visible. A user should be able to run a deterministic suite, inspect every tool call and approval, compare transports, understand regressions, and export evidence without reading raw trace files.

## Strategic Principles

- Local-first by default: no account, hosted service, or cloud dependency for the core workflow.
- Reproducibility over spectacle: every UI number must map back to a run artifact.
- Protocol honesty: distinguish true MCP behavior from simulator behavior.
- Safety is part of the benchmark: approvals, command execution, path access, network access, and redaction are measured surfaces.
- Enterprise-grade posture without enterprise friction: predictable CLI, typed models, clean artifacts, and CI gates.

## Tone

Precise, technical, terse, and confident. Avoid inflated SaaS claims. Prefer concrete verbs: run, compare, inspect, replay, export, fail.

## Anti-References

- Generic analytics dashboards with identical metric cards.
- Marketing landing pages that hide the actual product behind hero copy.
- Neon AI gradients, purple-blue glow, fake agent magic, and decorative terminal noise.
- Benchmarks that report a score without showing the trace and scoring evidence.
