import { NextResponse } from "next/server";

const apiBaseUrl = process.env.USB_AGENTS_API_URL ?? "http://127.0.0.1:8765";
const apiToken = process.env.USB_AGENTS_API_TOKEN;

export async function GET() {
  return proxy(`${apiBaseUrl}/api/runs`);
}

export async function POST(request: Request) {
  const body = await request.text();
  return proxy(`${apiBaseUrl}/api/runs`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body
  });
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
