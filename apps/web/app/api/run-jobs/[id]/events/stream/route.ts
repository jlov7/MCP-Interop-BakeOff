import { NextResponse } from "next/server";

const apiBaseUrl = process.env.USB_AGENTS_API_URL ?? "http://127.0.0.1:8765";
const apiToken = process.env.USB_AGENTS_API_TOKEN;

export async function GET(_request: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  try {
    const response = await fetch(`${apiBaseUrl}/api/run-jobs/${id}/events/stream`, {
      headers: authHeader(),
      cache: "no-store"
    });
    if (!response.ok || !response.body) {
      return new NextResponse(await response.text(), { status: response.status });
    }
    return new NextResponse(response.body, {
      status: response.status,
      headers: {
        "cache-control": "no-cache, no-transform",
        connection: "keep-alive",
        "content-type": "text/event-stream"
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
