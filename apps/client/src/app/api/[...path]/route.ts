// Runtime proxy to the FastAPI backend (apps/api).
//
// The browser hits same-origin `/api/*` (so there are no CORS round-trips); this handler forwards
// the request server-side to API_PROXY_TARGET. Unlike next.config `rewrites()` — whose destination
// Next bakes into the build — this reads the target at request time, so the deployed backend is a
// plain runtime env var (the internal API Service on the cluster; localhost in dev). No rebuild to
// repoint it.

import { type NextRequest } from "next/server";

// force-dynamic: never statically evaluate/cache — run per request so process.env is read live.
// nodejs runtime: needed to reach in-cluster Service DNS (the edge runtime can't).
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const apiTarget = (): string => process.env.API_PROXY_TARGET ?? "http://localhost:8000";

export async function GET(request: NextRequest): Promise<Response> {
  // Forward the incoming path (already `/api/...`) and query string verbatim to the backend.
  const url = `${apiTarget()}${request.nextUrl.pathname}${request.nextUrl.search}`;

  const upstream = await fetch(url, {
    headers: { Accept: request.headers.get("accept") ?? "application/json" },
    cache: "no-store", // The API owns freshness; don't cache at the proxy.
  });

  // Stream the body through, preserving status and content type.
  return new Response(upstream.body, {
    status: upstream.status,
    headers: { "content-type": upstream.headers.get("content-type") ?? "application/json" },
  });
}
