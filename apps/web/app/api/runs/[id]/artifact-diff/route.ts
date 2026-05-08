import { NextResponse } from "next/server";

const apiBaseUrl = process.env.USB_AGENTS_API_URL ?? "http://127.0.0.1:8765";
const apiToken = process.env.USB_AGENTS_API_TOKEN;

export async function GET(request: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  const query = new URL(request.url).searchParams;
  const upstream = new URL(`${apiBaseUrl}/api/runs/${id}/artifact-diff`);
  const baselineId = query.get("baseline_id");
  const path = query.get("path");
  if (baselineId) {
    upstream.searchParams.set("baseline_id", baselineId);
  }
  if (path) {
    upstream.searchParams.set("path", path);
  }
  try {
    const response = await fetch(upstream, { headers: authHeader(), cache: "no-store" });
    const text = await response.text();
    return new NextResponse(text, {
      status: response.status,
      headers: {
        "content-type": response.headers.get("content-type") ?? "application/json"
      }
    });
  } catch {
    return NextResponse.json(
      { error: "usb-agents API is not running", apiBaseUrl },
      { status: 503 }
    );
  }
}

function authHeader(): Record<string, string> {
  return apiToken ? { authorization: `Bearer ${apiToken}` } : {};
}
