const docs = [
  ["Architecture", "docs/architecture.md"],
  ["MCP compliance", "docs/mcp-compliance.md"],
  ["Adapter authoring", "docs/adapter-authoring.md"],
  ["Security threat model", "docs/security-threat-model.md"],
  ["Visual verification", "docs/visual-verification-protocol.md"],
  ["Release checklist", "docs/release-checklist.md"]
];

export default function DocsPage() {
  return (
    <main className="docs-page">
      <section>
        <span>usb-agents</span>
        <h1>Operator docs</h1>
        <div className="docs-grid">
          {docs.map(([label, path]) => (
            <article className="doc-row" key={path}>
              <div>
                <strong>{label}</strong>
                <code>{path}</code>
              </div>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}
