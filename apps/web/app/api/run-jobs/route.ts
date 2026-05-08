import { NextResponse } from "next/server";

const apiBaseUrl = process.env.USB_AGENTS_API_URL ?? "http://127.0.0.1:8765";
const apiToken = process.env.USB_AGENTS_API_TOKEN;

export async function POST(request: Request) {
  const body = await request.text();
  try {
    const response = await fetch(`${apiBaseUrl}/api/run-jobs`, {
      method: "POST",
      headers: { "content-type": "application/json", ...authHeader() },
      body,
      cache: "no-store"
    });
    return proxyJson(response);
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

async function proxyJson(response: Response) {
  const text = await response.text();
  return new NextResponse(text, {
    status: response.status,
    headers: {
      "content-type": response.headers.get("content-type") ?? "application/json"
    }
  });
}
