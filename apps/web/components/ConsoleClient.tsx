"use client";

import {
  Activity,
  Box,
  Check,
  ChevronRight,
  CircleStop,
  Clipboard,
  FileText,
  GitCompare,
  Play,
  RadioTower,
  ScrollText,
  ShieldCheck,
  TerminalSquare
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { ArtifactDiff, CaseResult, DashboardRun, RunJob, TraceEvent } from "../lib/data";

type View = "matrix" | "trace" | "compare" | "artifacts" | "docs";

declare global {
  interface Window {
    __USB_AGENTS_READY?: boolean;
    __USB_AGENTS_DEBUG__?: () => {
      runId: string;
      view: View;
      resultCount: number;
      selectedTask?: string;
    };
  }
}

const runtimeLabels: Record<string, string> = {
  openai_agents: "OpenAI",
  microsoft_agent_framework: "Microsoft",
  mistral_agents: "Mistral"
};

const taskLabels: Record<string, string> = {
  t1_repo_triage: "Repo triage",
  t2_calendar_merge: "Calendar merge",
  t3_http_etl: "HTTP ETL",
  t4_code_patch: "Code patch"
};

export function ConsoleClient({ run, runs }: { run: DashboardRun; runs: DashboardRun[] }) {
  const [activeRun, setActiveRun] = useState<DashboardRun>(run);
  const [knownRuns, setKnownRuns] = useState<DashboardRun[]>(runs);
  const [baselineId, setBaselineId] = useState(
    runs.find((item) => item.id !== run.id)?.id ?? run.id
  );
  const [selected, setSelected] = useState<CaseResult | undefined>(run.results[0]);
  const [view, setView] = useState<View>("matrix");
  const [copied, setCopied] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [job, setJob] = useState<RunJob | null>(null);
  const [liveEvents, setLiveEvents] = useState<string[]>([]);
  const runtimes = unique(activeRun.results.map((result) => result.runtime));
  const transports = unique(activeRun.results.map((result) => result.transport));
  const tasks = activeRun.suite.cases.map((item) => item.id);
  const selectedApprovals = selected?.approvals ?? [];
  const baseline = knownRuns.find((item) => item.id === baselineId) ?? knownRuns[0] ?? activeRun;

  useEffect(() => {
    setSelected(activeRun.results[0]);
  }, [activeRun]);

  useEffect(() => {
    window.__USB_AGENTS_READY = true;
    window.__USB_AGENTS_DEBUG__ = () => ({
      runId: activeRun.id,
      view,
      resultCount: activeRun.results.length,
      selectedTask: selected?.task_id
    });
    return () => {
      window.__USB_AGENTS_READY = false;
      window.__USB_AGENTS_DEBUG__ = undefined;
    };
  }, [activeRun.id, activeRun.results.length, selected?.task_id, view]);

  const matrixRows = useMemo(
    () =>
      runtimes.flatMap((runtime) =>
        transports.map((transport) => ({
          runtime,
          transport,
          cells: tasks.map((task) =>
            activeRun.results.find(
              (result) =>
                result.runtime === runtime &&
                result.transport === transport &&
                result.task_id === task
            )
          )
        }))
      ),
    [activeRun.results, runtimes, tasks, transports]
  );

  async function copyCommand() {
    await navigator.clipboard.writeText(activeRun.command);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  }

  async function launchEmbeddedRun() {
    setIsRunning(true);
    setRunError(null);
    setLiveEvents([]);
    try {
      const response = await fetch("/api/run-jobs", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ transports: ["embedded"], timeout_seconds: 20 })
      });
      if (!response.ok) {
        const payload = (await response.json().catch(() => ({}))) as { error?: string };
        throw new Error(payload.error ?? `Run failed with HTTP ${response.status}`);
      }
      const created = (await response.json()) as RunJob;
      setJob(created);
      const stream = new EventSource(`/api/run-jobs/${created.id}/events/stream`);
      stream.addEventListener("job", (event) => {
        const payload = JSON.parse(event.data) as { name: string; message: string };
        setLiveEvents((items) => [...items, `${payload.name}: ${payload.message}`].slice(-5));
      });
      stream.addEventListener("done", (event) => {
        const payload = JSON.parse(event.data) as RunJob;
        setJob(payload);
        if (payload.run) {
          setActiveRun(payload.run);
          setKnownRuns((items) => dedupeRuns([payload.run as DashboardRun, ...items]));
          setBaselineId((current) => current || payload.run?.id || activeRun.id);
        }
        setIsRunning(false);
        stream.close();
      });
      stream.onerror = () => {
        setIsRunning(false);
        setRunError("Run event stream disconnected");
        stream.close();
      };
    } catch (error) {
      setRunError(error instanceof Error ? error.message : "Run failed");
      setIsRunning(false);
    }
  }

  async function cancelRun() {
    if (!job) {
      return;
    }
    try {
      const response = await fetch(`/api/run-jobs/${job.id}`, { method: "DELETE" });
      if (response.ok) {
        setJob((await response.json()) as RunJob);
        setLiveEvents((items) => [...items, "job.canceling: Cancellation requested."].slice(-5));
      }
    } finally {
      setIsRunning(false);
    }
  }

  return (
    <main className="console-shell">
      <aside className="rail" aria-label="Primary">
        <div className="brand">
          <div className="brand-mark">
            <RadioTower size={18} aria-hidden />
          </div>
          <div>
            <strong>usb-agents</strong>
            <span>MCP portability lab</span>
          </div>
        </div>

        <nav className="rail-nav">
          <button className={view === "matrix" ? "active" : ""} onClick={() => setView("matrix")}>
            <GitCompare size={16} aria-hidden />
            Matrix
          </button>
          <button className={view === "trace" ? "active" : ""} onClick={() => setView("trace")}>
            <Activity size={16} aria-hidden />
            Trace
          </button>
          <button
            className={view === "compare" ? "active" : ""}
            onClick={() => setView("compare")}
          >
            <GitCompare size={16} aria-hidden />
            Compare
          </button>
          <button
            className={view === "artifacts" ? "active" : ""}
            onClick={() => setView("artifacts")}
          >
            <FileText size={16} aria-hidden />
            Artifacts
          </button>
          <button className={view === "docs" ? "active" : ""} onClick={() => setView("docs")}>
            <ScrollText size={16} aria-hidden />
            Docs
          </button>
        </nav>

        <div className="rail-footer">
          <span>Run</span>
          <code>{activeRun.id}</code>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="status-line">
              <span className={`status-dot ${activeRun.status}`} />
              {isRunning
                ? "Running embedded suite"
                : activeRun.status === "passed"
                  ? "Passing baseline"
                  : activeRun.status}
            </p>
            <h1>Compare agent runtime behavior before it ships.</h1>
            {runError ? <p className="error-line">{runError}</p> : null}
          </div>
          <div className="topbar-actions">
            <button
              className="run-button"
              onClick={launchEmbeddedRun}
              disabled={isRunning}
              aria-label="Run embedded suite"
            >
              <Play size={16} aria-hidden />
              {isRunning ? "Running" : "Run embedded"}
            </button>
            {isRunning ? (
              <button className="cancel-button" onClick={cancelRun} aria-label="Cancel run">
                <CircleStop size={16} aria-hidden />
                Cancel run
              </button>
            ) : null}
            <button className="command" onClick={copyCommand} aria-label="Copy run command">
              <TerminalSquare size={16} aria-hidden />
              <code>{activeRun.command}</code>
              {copied ? <Check size={16} aria-hidden /> : <Clipboard size={16} aria-hidden />}
            </button>
          </div>
        </header>

        <section className="summary-grid" aria-label="Run summary">
          <SummaryMetric
            label="Portability"
            value={`${Math.round(activeRun.metrics.portability_success_rate * 100)}%`}
            detail="success rate"
          />
          <SummaryMetric
            label="Trace coverage"
            value={`${Math.round(activeRun.metrics.trace_completeness * 100)}%`}
            detail="required spans"
          />
          <SummaryMetric
            label="Latency p95"
            value={`${activeRun.metrics.latency_p95.toFixed(1)}ms`}
            detail="slowest transport"
          />
          <SummaryMetric
            label="Approvals"
            value={`${activeRun.metrics.approvals}`}
            detail="HITL gates"
          />
        </section>

        <div className="content-grid">
          <section className="main-panel">
            {view === "matrix" ? (
              <MatrixView rows={matrixRows} selected={selected} onSelect={setSelected} />
            ) : null}
            {view === "trace" ? (
              <TraceView
                events={activeRun.events}
                onSelect={setSelected}
                results={activeRun.results}
              />
            ) : null}
            {view === "compare" ? (
              <CompareView run={activeRun} baseline={baseline} onBaselineChange={setBaselineId} runs={knownRuns} />
            ) : null}
            {view === "artifacts" ? <ArtifactView run={activeRun} /> : null}
            {view === "docs" ? <DocsView /> : null}
          </section>

          <aside className="inspector" aria-label="Selected case inspector">
            <div className="inspector-header">
              <span>Selected case</span>
              <strong>{taskLabels[selected?.task_id ?? ""] ?? selected?.task_id}</strong>
            </div>
            <dl className="detail-list">
              <div>
                <dt>Runtime</dt>
                <dd>{runtimeLabels[selected?.runtime ?? ""] ?? selected?.runtime}</dd>
              </div>
              <div>
                <dt>Transport</dt>
                <dd>{selected?.transport}</dd>
              </div>
              <div>
                <dt>Latency</dt>
                <dd>{selected ? `${selected.latency_ms.toFixed(1)}ms` : "-"}</dd>
              </div>
              <div>
                <dt>Tool calls</dt>
                <dd>{selected?.tool_calls ?? 0}</dd>
              </div>
            </dl>

            <div className="approval-panel">
              <div className="panel-title">
                <ShieldCheck size={15} aria-hidden />
                Approval timeline
              </div>
              {selectedApprovals.length ? (
                selectedApprovals.map((approval) => (
                  <div className="approval-row" key={`${approval.tool}:${approval.reason}`}>
                    <span className={approval.granted ? "pass-chip" : "fail-chip"}>
                      {approval.granted ? "granted" : "denied"}
                    </span>
                    <div>
                      <strong>{approval.tool}</strong>
                      <small>{approval.reason}</small>
                    </div>
                  </div>
                ))
              ) : (
                <p className="empty-copy">Read-only task. No approval gate was required.</p>
              )}
            </div>

            <div className="payload-panel">
              <div className="panel-title">
                <Box size={15} aria-hidden />
                {job ? `Job ${job.status}` : "Evidence payload"}
              </div>
              <pre>
                {liveEvents.length
                  ? liveEvents.join("\n")
                  : JSON.stringify(selected?.metadata ?? {}, null, 2)}
              </pre>
            </div>
          </aside>
        </div>
      </section>
    </main>
  );
}

function SummaryMetric({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="summary-metric">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}

function MatrixView({
  rows,
  selected,
  onSelect
}: {
  rows: { runtime: string; transport: string; cells: (CaseResult | undefined)[] }[];
  selected?: CaseResult;
  onSelect: (result: CaseResult) => void;
}) {
  const tasks = ["t1_repo_triage", "t2_calendar_merge", "t3_http_etl", "t4_code_patch"];
  return (
    <div className="matrix-view">
      <div className="section-heading">
        <div>
          <span>Runtime x transport</span>
          <h2>Compatibility matrix</h2>
        </div>
        <p>Each cell links status, latency, approvals, and trace evidence.</p>
      </div>
      <div className="matrix-table" role="table" aria-label="Compatibility matrix">
        <div className="matrix-head" role="row">
          <span>Adapter</span>
          {tasks.map((task) => (
            <span key={task}>{taskLabels[task]}</span>
          ))}
        </div>
        {rows.map((row) => (
          <div className="matrix-row" role="row" key={`${row.runtime}:${row.transport}`}>
            <div className="adapter-cell">
              <strong>{runtimeLabels[row.runtime] ?? row.runtime}</strong>
              <small>{row.transport}</small>
            </div>
            {row.cells.map((cell, index) => (
              <button
                className={`result-cell ${cell?.success ? "pass" : "fail"} ${
                  selected === cell ? "selected" : ""
                }`}
                key={`${row.runtime}:${row.transport}:${tasks[index]}`}
                onClick={() => cell && onSelect(cell)}
                disabled={!cell}
              >
                <span>{cell?.success ? "pass" : "miss"}</span>
                <code>{cell ? `${cell.latency_ms.toFixed(1)}ms` : "-"}</code>
              </button>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

function TraceView({
  events,
  results,
  onSelect
}: {
  events: TraceEvent[];
  results: CaseResult[];
  onSelect: (result: CaseResult) => void;
}) {
  return (
    <div className="trace-view">
      <div className="section-heading">
        <div>
          <span>Execution evidence</span>
          <h2>Trace rail</h2>
        </div>
        <p>Structured events stay tied to runtime, transport, task, and latency.</p>
      </div>
      <div className="trace-list">
        {events.map((eventItem) => {
          const match = results.find(
            (result) =>
              result.runtime === eventItem.runtime &&
              result.task_id === eventItem.task_id &&
              result.transport === eventItem.transport
          );
          return (
            <button
              className={`trace-row ${eventItem.severity}`}
              key={eventItem.id}
              onClick={() => match && onSelect(match)}
            >
              <span className="trace-pin" />
              <div>
                <strong>{eventItem.name}</strong>
                <small>{eventItem.message}</small>
              </div>
              <code>{eventItem.latency_ms ? `${eventItem.latency_ms.toFixed(1)}ms` : "event"}</code>
              <ChevronRight size={16} aria-hidden />
            </button>
          );
        })}
      </div>
    </div>
  );
}

function ArtifactView({ run }: { run: DashboardRun }) {
  return (
    <div className="artifact-view">
      <div className="section-heading">
        <div>
          <span>Run outputs</span>
          <h2>Artifact inspector</h2>
        </div>
        <p>Reports, transport metrics, traces, and generated task outputs are local files.</p>
      </div>
      <div className="artifact-list">
        {run.artifacts.map((artifact) => (
          <article className="artifact-row" key={artifact.path}>
            <FileText size={16} aria-hidden />
            <div>
              <strong>{artifact.path}</strong>
              <small>{artifact.description}</small>
            </div>
            <span>{artifact.kind}</span>
            <code>{artifact.bytes}b</code>
          </article>
        ))}
      </div>
    </div>
  );
}

function CompareView({
  run,
  baseline,
  runs,
  onBaselineChange
}: {
  run: DashboardRun;
  baseline: DashboardRun;
  runs: DashboardRun[];
  onBaselineChange: (runId: string) => void;
}) {
  const comparableArtifacts = run.artifacts.filter((artifact) =>
    ["csv", "json", "markdown", "text", "trace"].includes(artifact.kind)
  );
  const [artifactPath, setArtifactPath] = useState(comparableArtifacts[0]?.path ?? "");
  const [artifactDiff, setArtifactDiff] = useState<ArtifactDiff | null>(null);
  const [artifactError, setArtifactError] = useState<string | null>(null);
  const deltas = [
    delta("Portability", run.metrics.portability_success_rate, baseline.metrics.portability_success_rate, "%"),
    delta("Trace", run.metrics.trace_completeness, baseline.metrics.trace_completeness, "%"),
    delta("Latency p95", run.metrics.latency_p95, baseline.metrics.latency_p95, "ms", true),
    delta("Approvals", run.metrics.approvals, baseline.metrics.approvals, ""),
    delta("Tokens", run.metrics.token_total, baseline.metrics.token_total, "")
  ];
  const regressions = deltas.filter((item) => item.regressed);

  useEffect(() => {
    setArtifactPath(comparableArtifacts[0]?.path ?? "");
  }, [run.id]);

  async function loadArtifactDiff() {
    if (!artifactPath) {
      return;
    }
    setArtifactError(null);
    const query = new URLSearchParams({ baseline_id: baseline.id, path: artifactPath });
    const response = await fetch(`/api/runs/${run.id}/artifact-diff?${query}`);
    if (!response.ok) {
      setArtifactError(`Artifact diff failed with HTTP ${response.status}`);
      setArtifactDiff(null);
      return;
    }
    setArtifactDiff((await response.json()) as ArtifactDiff);
  }

  return (
    <div className="compare-view">
      <div className="section-heading">
        <div>
          <span>Baseline diff</span>
          <h2>Regression compare</h2>
        </div>
        <label className="baseline-select">
          <span>Baseline</span>
          <select value={baseline.id} onChange={(event) => onBaselineChange(event.target.value)}>
            {runs.map((item) => (
              <option key={item.id} value={item.id}>
                {item.id}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="compare-grid">
        {deltas.map((item) => (
          <article className={item.regressed ? "delta-row regressed" : "delta-row"} key={item.label}>
            <span>{item.label}</span>
            <strong>{item.display}</strong>
            <small>{item.detail}</small>
          </article>
        ))}
      </div>
      <div className="regression-list">
        <strong>{regressions.length ? "Regressions" : "No regressions"}</strong>
        {regressions.length ? (
          regressions.map((item) => <span key={item.label}>{item.label}</span>)
        ) : (
          <span>Candidate is within baseline thresholds.</span>
        )}
      </div>
      <div className="artifact-diff-panel">
        <div className="artifact-diff-controls">
          <label className="baseline-select">
            <span>Artifact</span>
            <select value={artifactPath} onChange={(event) => setArtifactPath(event.target.value)}>
              {comparableArtifacts.map((artifact) => (
                <option key={artifact.path} value={artifact.path}>
                  {artifact.path}
                </option>
              ))}
            </select>
          </label>
          <button className="diff-button" onClick={loadArtifactDiff} disabled={!artifactPath}>
            Compare artifact
          </button>
        </div>
        {artifactError ? <p className="error-line">{artifactError}</p> : null}
        {artifactDiff ? (
          <div className="side-by-side-diff">
            <pre>{artifactDiff.baseline || "(empty baseline)"}</pre>
            <pre>{artifactDiff.candidate || "(empty candidate)"}</pre>
            <pre className="unified-diff">{artifactDiff.diff || "No textual changes"}</pre>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function DocsView() {
  const links = [
    ["Architecture", "docs/architecture.md"],
    ["MCP compliance", "docs/mcp-compliance.md"],
    ["Adapter authoring", "docs/adapter-authoring.md"],
    ["Release checklist", "docs/release-checklist.md"]
  ];

  return (
    <div className="docs-view">
      <div className="section-heading">
        <div>
          <span>Operator docs</span>
          <h2>Project cockpit</h2>
        </div>
        <p>Local-first release, security, and adapter materials.</p>
      </div>
      <div className="docs-grid">
        {links.map(([label, path]) => (
          <article className="doc-row" key={path}>
            <ScrollText size={16} aria-hidden />
            <div>
              <strong>{label}</strong>
              <code>{path}</code>
            </div>
          </article>
        ))}
      </div>
      <pre className="docs-command">uv run usb-agents doctor{"\n"}uv run usb-agents run --transport embedded --transport http --transport stdio{"\n"}pnpm verify</pre>
    </div>
  );
}

function unique(values: string[]) {
  return Array.from(new Set(values));
}

function dedupeRuns(runs: DashboardRun[]) {
  const seen = new Set<string>();
  return runs.filter((run) => {
    if (seen.has(run.id)) {
      return false;
    }
    seen.add(run.id);
    return true;
  });
}

function delta(label: string, value: number, baseline: number, unit: string, lowerIsBetter = false) {
  const change = value - baseline;
  const regressed = lowerIsBetter ? change > 0.01 : change < -0.01;
  const format = unit === "%" ? `${Math.round(value * 100)}%` : `${value.toFixed(unit ? 1 : 0)}${unit}`;
  const sign = change > 0 ? "+" : "";
  const detail =
    unit === "%"
      ? `${sign}${Math.round(change * 100)} pts vs baseline`
      : `${sign}${change.toFixed(unit ? 1 : 0)}${unit} vs baseline`;
  return { label, display: format, detail, regressed };
}
