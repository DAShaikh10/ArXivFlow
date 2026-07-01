"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { PaperCard } from "@/components/feed/PaperCard";
import { Icon } from "@/components/shared/Icon";
import { LatencyBadge } from "@/components/shared/LatencyBadge";
import { ErrorState, FeedSkeleton } from "@/components/shared/States";
import { api } from "@/lib/api";
import { useCategories } from "@/lib/categories";
import { useTheme } from "@/lib/theme";
import { type SortKey } from "@/lib/types";
import { useApi } from "@/lib/useApi";

const SORTS: Array<[SortKey, string]> = [
  ["cited", "Most cited"],
  ["newest", "Newest"],
  ["title", "Title A-Z"],
];

const PAGE_SIZE = 25;

export default function DiscoverPage() {
  const theme = useTheme();
  const { list: categories } = useCategories();

  const [sort, setSort] = useState<SortKey>("cited");
  const [clusterId, setClusterId] = useState<string>("all");
  const [yearFrom, setYearFrom] = useState<number>(0);
  const [limit, setLimit] = useState<number>(PAGE_SIZE);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const [years, setYears] = useState<number[]>([]);
  useEffect(() => {
    const controller = new AbortController();
    api
      .getYears(controller.signal)
      .then(({ data }) => setYears(data))
      .catch(() => {});
    return () => controller.abort();
  }, []);

  const params = useMemo(
    () => ({
      sort,
      limit,
      ...(clusterId !== "all" ? { cluster_id: clusterId } : {}),
      ...(yearFrom ? { year_from: yearFrom } : {}),
    }),
    [sort, clusterId, yearFrom, limit],
  );

  const { data, error, loading, tookMs, refreshing } = useApi(
    useCallback((signal: AbortSignal) => api.listPapers(params, signal), [params]),
    [params],
  );

  const items = data?.items ?? [];
  const total = data?.total ?? 0;
  const filtered = clusterId !== "all" || yearFrom > 0;

  const allOpen = items.length > 0 && items.every((paper) => expanded.has(paper.id));
  const toggleAll = () => setExpanded(allOpen ? new Set() : new Set(items.map((paper) => paper.id)));
  const toggleOne = (id: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  return (
    <div className="feed-wrap">
      <div className="feed-head">
        <div className="feed-head-top">
          <div>
            <div className="feed-eyebrow">
              <span className="feed-eyebrow-dot" />
              {total ? `${total} arXiv NLP papers indexed` : "ArXivFlow corpus"}
            </div>
            <h1 className="feed-h1">Discover</h1>
            <p className="feed-lede">
              Browse the indexed corpus. Open any paper to see its nearest neighbours by SPECTER2 embedding similarity.
            </p>
          </div>
          <LatencyBadge
            tookMs={tookMs}
            serverMs={data?.took_ms ?? null}
            busy={refreshing}
            show={theme.showLatency}
            variant="inline"
            label={`listed ${items.length} papers`}
          />
        </div>

        <div className="feed-controls">
          <div className="feed-sorts">
            <Icon name="sort" s={15} c="var(--text-faint)" />
            {SORTS.map(([key, label]) => (
              <button key={key} className={`feed-sort${sort === key ? " on" : ""}`} onClick={() => setSort(key)}>
                {label}
              </button>
            ))}
          </div>

          <div className="feed-selects">
            <label className="feed-sel">
              <span>Cluster</span>
              <select value={clusterId} onChange={(event) => setClusterId(event.target.value)}>
                <option value="all">All clusters</option>
                {categories.map((category) => (
                  <option key={category.id} value={category.id}>
                    {category.name} ({category.count})
                  </option>
                ))}
              </select>
            </label>
            <label className="feed-sel">
              <span>Since</span>
              <select value={yearFrom} onChange={(event) => setYearFrom(Number(event.target.value))}>
                <option value={0}>All years</option>
                {years.map((year) => (
                  <option key={year} value={year}>
                    {year}
                  </option>
                ))}
              </select>
            </label>
            <button className="af-ghost" onClick={toggleAll} disabled={items.length === 0}>
              <Icon name={allOpen ? "collapse" : "expand"} s={14} />
              {allOpen ? "Collapse all" : "Expand all"}
            </button>
          </div>
        </div>

        <div className="feed-count">
          <b>{total}</b> papers{filtered ? " · filtered" : ""}
        </div>
      </div>

      {error ? (
        <ErrorState message={error} />
      ) : loading && items.length === 0 ? (
        <FeedSkeleton />
      ) : items.length === 0 ? (
        <div className="feed-empty">No papers match these filters.</div>
      ) : (
        <>
          <div className="feed-list">
            {items.map((paper) => (
              <PaperCard
                key={paper.id}
                paper={paper}
                expanded={expanded.has(paper.id)}
                onToggle={() => toggleOne(paper.id)}
              />
            ))}
          </div>
          {items.length < total && (
            <div style={{ display: "flex", justifyContent: "center", marginTop: 24 }}>
              <button className="af-btn" onClick={() => setLimit((value) => value + PAGE_SIZE)} disabled={refreshing}>
                {refreshing ? "Loading…" : `Load more (${total - items.length} left)`}
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
