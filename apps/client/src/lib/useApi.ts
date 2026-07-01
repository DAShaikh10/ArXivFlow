// A tiny data-fetching hook: tracks loading / error / data and the measured round-trip latency,
// re-running whenever `deps` change and aborting the in-flight request on change or unmount.

"use client";

import { useEffect, useState } from "react";

import { type Timed } from "./api";

export interface ApiState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  /** Real client-side round-trip time of the most recent successful fetch, in ms. */
  tookMs: number | null;
  /** True while a refetch (after the first load) is in flight — drives the "computing…" badge. */
  refreshing: boolean;
}

export function useApi<T>(
  fetcher: (signal: AbortSignal) => Promise<Timed<T>>,
  deps: ReadonlyArray<unknown>,
): ApiState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [tookMs, setTookMs] = useState<number | null>(null);
  // State (not a ref) so it can be read during render for `refreshing`; set inside the async
  // resolution below, which the set-state-in-effect rule does not flag.
  const [hasLoaded, setHasLoaded] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    // Synchronous reset on dep change: show the spinner and clear the stale error before the refetch
    // resolves. The only rule-satisfying alternative is a Suspense rewrite, out of scope for this hook.
    /* eslint-disable react-hooks/set-state-in-effect */
    setLoading(true);
    setError(null);
    /* eslint-enable react-hooks/set-state-in-effect */

    fetcher(controller.signal)
      .then(({ data: result, tookMs: ms }) => {
        setData(result);
        setTookMs(ms);
        setHasLoaded(true);
      })
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setError(err instanceof Error ? err.message : "Request failed");
      })
      .finally(() => setLoading(false));

    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return {
    data,
    error,
    loading,
    tookMs,
    refreshing: loading && hasLoaded,
  };
}
