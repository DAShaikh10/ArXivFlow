// HTTP client for the ArXivFlow API.
//
// In the browser we hit same-origin `/api/*`, which the Route Handler at `src/app/api/[...path]`
// proxies server-side to the FastAPI backend (so there are no CORS round-trips). Set NEXT_PUBLIC_API_BASE
// only to bypass the proxy and call a backend host directly from the browser (then CORS applies).

import {
  type AtlasResponse,
  type Category,
  type NeighborsResponse,
  type PaperDetail,
  type PaperListParams,
  type PaperListResponse,
  type Recommender,
  type SearchResponse,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** A fetched result carrying the measured client-side round-trip time. */
export interface Timed<T> {
  data: T;
  tookMs: number;
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<Timed<T>> {
  const started = performance.now();
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { Accept: "application/json" },
    ...(signal ? { signal } : {}),
  });
  const tookMs = performance.now() - started;

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // non-JSON error body; keep statusText
    }
    throw new ApiError(detail, response.status);
  }

  return { data: (await response.json()) as T, tookMs };
}

function queryString(params: Record<string, string | number | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== "") search.set(key, String(value));
  }
  const qs = search.toString();
  return qs ? `?${qs}` : "";
}

export const api = {
  listPapers(params: PaperListParams = {}, signal?: AbortSignal): Promise<Timed<PaperListResponse>> {
    return getJson<PaperListResponse>(`/api/papers${queryString({ ...params })}`, signal);
  },

  getPaper(id: string, signal?: AbortSignal): Promise<Timed<PaperDetail>> {
    return getJson<PaperDetail>(`/api/papers/${encodeURIComponent(id)}`, signal);
  },

  getNeighbors(
    id: string,
    k = 8,
    recommender: Recommender = "dense",
    signal?: AbortSignal,
  ): Promise<Timed<NeighborsResponse>> {
    return getJson<NeighborsResponse>(
      `/api/papers/${encodeURIComponent(id)}/neighbors${queryString({ k, recommender })}`,
      signal,
    );
  },

  search(q: string, k = 20, signal?: AbortSignal): Promise<Timed<SearchResponse>> {
    return getJson<SearchResponse>(`/api/search${queryString({ q, k })}`, signal);
  },

  getCategories(signal?: AbortSignal): Promise<Timed<Category[]>> {
    return getJson<Category[]>(`/api/categories`, signal);
  },

  getYears(signal?: AbortSignal): Promise<Timed<number[]>> {
    return getJson<number[]>(`/api/years`, signal);
  },

  getAtlas(signal?: AbortSignal): Promise<Timed<AtlasResponse>> {
    return getJson<AtlasResponse>(`/api/atlas`, signal);
  },
};
