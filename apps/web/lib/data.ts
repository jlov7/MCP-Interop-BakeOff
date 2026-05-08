export type CaseResult = {
  runtime: string;
  task_id: string;
  transport: string;
  success: boolean;
  latency_ms: number;
  tool_calls: number;
  approvals: { tool: string; granted: boolean; reason: string }[];
  failure_category?: string | null;
  trace_path?: string | null;
  metadata?: Record<string, unknown>;
};

export type TraceEvent = {
  id: string;
  name: string;
  severity: "info" | "success" | "warning" | "error";
  runtime?: string | null;
  task_id?: string | null;
  transport?: string | null;
  latency_ms?: number | null;
  message: string;
};

export type DashboardRun = {
  id: string;
  status: "queued" | "running" | "passed" | "failed" | "canceled" | "canceling";
  command: string;
  metrics: {
    portability_success_rate: number;
    trace_completeness: number;
    latency_p95: number;
    approvals: number;
    token_total: number;
  };
  suite: {
    cases: { id: string; name: string; approval: string; mode: string }[];
  };
  results: CaseResult[];
  events: TraceEvent[];
  artifacts: { path: string; kind: string; bytes: number; description: string }[];
};

export type RunJob = {
  id: string;
  status: DashboardRun["status"];
  error?: string | null;
  run_id?: string | null;
  run?: DashboardRun | null;
  events: { name: string; message: string }[];
};

export type RunComparison = {
  candidate: string;
  baseline: string;
  deltas: {
    portability_success_rate: number;
    trace_completeness: number;
    latency_p95: number;
    approvals: number;
    token_total: number;
  };
  regressions: string[];
};

export type ArtifactDiff = {
  run_id: string;
  baseline_id: string;
  path: string;
  changed: boolean;
  baseline: string;
  candidate: string;
  diff: string;
  stats: {
    baseline_lines: number;
    candidate_lines: number;
    diff_lines: number;
  };
};

const fallbackRun: DashboardRun = {
  id: "demo_archived_portability",
  status: "passed",
  command: "usb-agents run --transport embedded --transport http --transport stdio",
  metrics: {
    portability_success_rate: 1,
    trace_completeness: 1,
    latency_p95: 420.1,
    approvals: 27,
    token_total: 576
  },
  suite: {
    cases: [
      { id: "t1_repo_triage", name: "Repository Triage", approval: "never", mode: "read" },
      { id: "t2_calendar_merge", name: "Calendar Merge", approval: "by_tool", mode: "write" },
      { id: "t3_http_etl", name: "HTTP ETL", approval: "by_tool", mode: "read_write" },
      { id: "t4_code_patch", name: "Code Patch", approval: "by_tool", mode: "write" }
    ]
  },
  results: [
    result("openai_agents", "embedded", "t1_repo_triage", 0.9, 3, true),
    result("openai_agents", "embedded", "t2_calendar_merge", 0.4, 1, true),
    result("openai_agents", "embedded", "t3_http_etl", 0.2, 1, true),
    result("openai_agents", "embedded", "t4_code_patch", 395.3, 1, true),
    result("microsoft_agent_framework", "http", "t1_repo_triage", 35.4, 3, true),
    result("microsoft_agent_framework", "http", "t2_calendar_merge", 34.8, 1, true),
    result("microsoft_agent_framework", "http", "t3_http_etl", 33.9, 1, true),
    result("microsoft_agent_framework", "http", "t4_code_patch", 420.1, 1, true),
    result("mistral_agents", "stdio", "t1_repo_triage", 1.7, 3, true),
    result("mistral_agents", "stdio", "t2_calendar_merge", 1.1, 1, true),
    result("mistral_agents", "stdio", "t3_http_etl", 0.9, 1, true),
    result("mistral_agents", "stdio", "t4_code_patch", 397.3, 1, true)
  ],
  events: [
    event("suite.loaded", "info", "Archived baseline loaded"),
    event("case.completed", "success", "OpenAI completed repository triage", "openai_agents", "t1_repo_triage", "embedded", 0.9),
    event("approval.granted", "warning", "calendar_merge approved", "openai_agents", "t2_calendar_merge", "embedded", 0.4),
    event("tool.executed", "success", "code_exec finished pytest", "mistral_agents", "t4_code_patch", "stdio", 397.3)
  ],
  artifacts: [
    { path: "results.md", kind: "markdown", bytes: 1458, description: "Archived report" },
    { path: "transport_metrics.json", kind: "json", bytes: 3184, description: "Transport metrics" },
    { path: "tasks/artifacts/http_etl.csv", kind: "csv", bytes: 56, description: "ETL output" }
  ]
};

export async function getDashboardRun(forceFixture = false): Promise<DashboardRun> {
  const runs = await getDashboardRuns(forceFixture);
  return runs[0] ?? fallbackRun;
}

export async function getDashboardRuns(forceFixture = false): Promise<DashboardRun[]> {
  if (forceFixture) {
    return [fallbackRun];
  }
  const baseUrl = process.env.USB_AGENTS_API_URL ?? "http://127.0.0.1:8765";
  try {
    const response = await fetch(`${baseUrl}/api/runs`, { cache: "no-store" });
    if (!response.ok) {
      return [fallbackRun];
    }
    const payload = (await response.json()) as { runs?: DashboardRun[] };
    return payload.runs?.length ? payload.runs : [fallbackRun];
  } catch {
    return [fallbackRun];
  }
}

function result(
  runtime: string,
  transport: string,
  taskId: string,
  latency: number,
  toolCalls: number,
  success: boolean
): CaseResult {
  return {
    runtime,
    transport,
    task_id: taskId,
    success,
    latency_ms: latency,
    tool_calls: toolCalls,
    approvals: taskId === "t1_repo_triage" ? [] : [{ tool: taskId, granted: true, reason: `${runtime}:hitl` }]
  };
}

function event(
  name: TraceEvent["name"],
  severity: TraceEvent["severity"],
  message: string,
  runtime?: string,
  taskId?: string,
  transport?: string,
  latency?: number
): TraceEvent {
  return {
    id: `${name}:${message}`,
    name,
    severity,
    message,
    runtime,
    task_id: taskId,
    transport,
    latency_ms: latency
  };
}
