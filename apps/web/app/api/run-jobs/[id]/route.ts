import { NextResponse } from "next/server";

const apiBaseUrl = process.env.USB_AGENTS_API_URL ?? "http://127.0.0.1:8765";
const apiToken = process.env.USB_AGENTS_API_TOKEN;

export async function GET(_request: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  return proxy(`${apiBaseUrl}/api/run-jobs/${id}`);
}

export async function DELETE(_request: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  return proxy(`${apiBaseUrl}/api/run-jobs/${id}`, { method: "DELETE" });
}

async function proxy(url: string, init?: RequestInit) {
  try {
    const headers = new Headers(init?.headers);
    if (apiToken) {
      headers.set("authorization", `Bearer ${apiToken}`);
    }
    const response = await fetch(url, {
      ...init,
      headers,
      cache: "no-store"
    });
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
