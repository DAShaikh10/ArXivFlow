// Paper detail: metadata, collapsible Abstract / Similar papers (embedding neighbours) / References,
// plus a sticky rail with the arXiv link, an atlas deep-link, and a nearest-neighbour shortlist.

"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";

import { CatPill } from "@/components/shared/CatPill";
import { Icon } from "@/components/shared/Icon";
import { LatencyBadge } from "@/components/shared/LatencyBadge";
import { ErrorState, Loading } from "@/components/shared/States";
import { api } from "@/lib/api";
import { arxivUrl, authorStr, dateStr, topicFieldLabel } from "@/lib/format";
import { RECOMMENDERS, useTheme } from "@/lib/theme";
import { type Neighbor } from "@/lib/types";
import { useApi } from "@/lib/useApi";

type RecSort = "similarity" | "newest" | "cited";

const REC_SORTS: Array<[RecSort, string]> = [
  ["similarity", "Similarity"],
  ["newest", "Newest"],
  ["cited", "Most cited"],
];

interface SectionProps {
  title: string;
  count?: number;
  open: boolean;
  onToggle: () => void;
  accent?: React.ReactNode;
  children: React.ReactNode;
}

function Section({ title, count, open, onToggle, accent, children }: SectionProps) {
  // A div (not a <button>) because the header can contain its own interactive accent (the latency
  // badge button) — nesting buttons is invalid HTML and breaks hydration.
  return (
    <section className={`dsec${open ? " open" : ""}`}>
      <div
        className="dsec-h"
        role="button"
        tabIndex={0}
        onClick={onToggle}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            onToggle();
          }
        }}
      >
        <Icon
          name="chevR"
          s={16}
          c="var(--text-dim)"
          style={{ transform: open ? "rotate(90deg)" : "none", transition: ".2s" }}
        />
        <span className="dsec-t">{title}</span>
        {count != null && <span className="dsec-c">{count}</span>}
        {accent}
      </div>
      {open && <div className="dsec-body">{children}</div>}
    </section>
  );
}

function RecRow({
  neighbor,
  showLat,
  onOpen,
}: {
  neighbor: Neighbor;
  showLat: boolean;
  onOpen: (id: string) => void;
}) {
  return (
    <button className="rec-row" onClick={() => onOpen(neighbor.id)}>
      <div className="rec-sim">
        <div className="rec-sim-num">{neighbor.similarity.toFixed(2)}</div>
        <div className="rec-sim-bar">
          <i style={{ width: `${Math.max(0, Math.min(1, neighbor.similarity)) * 100}%` }} />
        </div>
        <div className="rec-sim-lbl">similarity</div>
      </div>
      <div className="rec-main">
        <div className="rec-cats">
          <CatPill clusterId={neighbor.cluster_id} />
          <span className="rec-id">arXiv:{neighbor.id}</span>
          {showLat && neighbor.influential_citations > 0 && (
            <span className="rec-time">
              <Icon name="clock" s={11} c="var(--text-faint)" />
              cited by {neighbor.influential_citations}
            </span>
          )}
        </div>
        <h4 className="rec-title">{neighbor.title}</h4>
        {neighbor.authors.length > 0 && <div className="rec-auth">{authorStr(neighbor.authors, 3)}</div>}
      </div>
      <Icon name="chevR" s={18} c="var(--text-faint)" style={{ flexShrink: 0, alignSelf: "center" }} />
    </button>
  );
}

export default function PaperDetailPage() {
  const params = useParams<{ id: string }>();
  const id = decodeURIComponent(params.id);
  const router = useRouter();
  const theme = useTheme();

  const [recSort, setRecSort] = useState<RecSort>("similarity");
  const [open, setOpen] = useState({ abs: true, recs: true, refs: false });

  const detail = useApi(
    useCallback((signal: AbortSignal) => api.getPaper(id, signal), [id]),
    [id],
  );
  const neighbors = useApi(
    useCallback(
      (signal: AbortSignal) => api.getNeighbors(id, theme.recCount, theme.recommender, signal),
      [id, theme.recCount, theme.recommender],
    ),
    [id, theme.recCount, theme.recommender],
  );

  // Reset scroll when navigating between papers.
  useEffect(() => {
    document.querySelector(".stage-scroll")?.scrollTo(0, 0);
  }, [id]);

  const recs = useMemo(() => {
    const list = [...(neighbors.data?.neighbors ?? [])];
    if (recSort === "newest") list.sort((a, b) => (b.published_date ?? "").localeCompare(a.published_date ?? ""));
    else if (recSort === "cited") list.sort((a, b) => b.influential_citations - a.influential_citations);
    else list.sort((a, b) => b.similarity - a.similarity);
    return list;
  }, [neighbors.data, recSort]);

  const openPaper = (paperId: string) => router.push(`/paper/${encodeURIComponent(paperId)}`);

  if (detail.error)
    return (
      <div className="detail-wrap">
        <ErrorState message={detail.error} />
      </div>
    );
  if (detail.loading || !detail.data)
    return (
      <div className="detail-wrap">
        <Loading message="Loading paper…" />
      </div>
    );

  const paper = detail.data;
  const setSection = (key: keyof typeof open) => setOpen((prev) => ({ ...prev, [key]: !prev[key] }));
  const allOpen = open.abs && open.recs && open.refs;
  const toggleAll = () =>
    setOpen(allOpen ? { abs: false, recs: false, refs: false } : { abs: true, recs: true, refs: true });

  return (
    <div className="detail-wrap">
      <Link className="detail-back" href="/">
        <Icon name="arrowL" s={16} />
        Back to Discover
      </Link>

      <div className="detail-grid">
        <div className="detail-main">
          <div className="detail-cats">
            <CatPill clusterId={paper.cluster_id} />
            <span className="detail-id">arXiv:{paper.id}</span>
            <span className="detail-date">{dateStr(paper.published_date)}</span>
          </div>
          <h1 className="detail-title">{paper.title}</h1>
          {paper.authors.length > 0 && <div className="detail-auth">{paper.authors.join(", ")}</div>}

          <div className="detail-metarow">
            <span className="detail-stat">
              <b>{paper.influential_citations}</b> citations
            </span>
            <span className="detail-stat">
              <b>{neighbors.data?.neighbors.length ?? theme.recCount}</b> related
            </span>
            <span className="detail-stat">
              <b>{paper.references.length}</b> references
            </span>
          </div>

          {paper.topics.length > 0 && (
            <div className="detail-topics">
              {paper.topics.map((topic) => (
                <span
                  key={`${topic.field}:${topic.value}`}
                  className="af-cat"
                  style={{ background: "var(--surface-2)", color: "var(--text-dim)" }}
                  title={topicFieldLabel(topic.field)}
                >
                  {topic.value}
                </span>
              ))}
            </div>
          )}

          <div className="detail-toolbar">
            <button className="af-ghost" onClick={toggleAll}>
              <Icon name={allOpen ? "collapse" : "expand"} s={14} />
              {allOpen ? "Collapse all" : "Expand all"}
            </button>
          </div>

          <Section title="Abstract" open={open.abs} onToggle={() => setSection("abs")}>
            <p className="detail-abs">{paper.abstract}</p>
          </Section>

          <Section
            title="Similar papers"
            count={recs.length}
            open={open.recs}
            onToggle={() => setSection("recs")}
            accent={
              <span className="dsec-right" onClick={(event) => event.stopPropagation()}>
                <LatencyBadge
                  tookMs={neighbors.tookMs}
                  serverMs={neighbors.data?.took_ms ?? null}
                  busy={neighbors.refreshing}
                  show={theme.showLatency}
                  variant="inline"
                  label="recommended"
                />
              </span>
            }
          >
            <div className="rec-controls">
              <span className="rec-controls-lbl">Recommender</span>
              <div className="af-seg">
                {RECOMMENDERS.map((rec) => (
                  <button
                    key={rec.key}
                    className={theme.recommender === rec.key ? "on" : ""}
                    title={rec.hint}
                    onClick={() => theme.set("recommender", rec.key)}
                  >
                    {rec.label}
                  </button>
                ))}
              </div>
              <span className="rec-controls-lbl">Sort by</span>
              {REC_SORTS.map(([key, label]) => (
                <button
                  key={key}
                  className={`rec-sortbtn${recSort === key ? " on" : ""}`}
                  onClick={() => setRecSort(key)}
                >
                  {label}
                </button>
              ))}
            </div>
            {neighbors.error ? (
              <ErrorState message={neighbors.error} />
            ) : neighbors.loading && recs.length === 0 ? (
              <Loading message="Scoring neighbours…" />
            ) : (
              <div className="rec-list">
                {recs.map((neighbor) => (
                  <RecRow key={neighbor.id} neighbor={neighbor} showLat={theme.showLatency} onOpen={openPaper} />
                ))}
              </div>
            )}
          </Section>

          <Section
            title="References"
            count={paper.references.length}
            open={open.refs}
            onToggle={() => setSection("refs")}
          >
            <ol className="ref-list">
              {paper.references.map((ref, index) => (
                <li key={`${ref.arxiv_id ?? "noid"}-${index}`} className="ref-item">
                  <span className="ref-num">{index + 1}</span>
                  <div className="ref-body">
                    <div className="ref-title">{ref.title}</div>
                    <div className="ref-meta">
                      {ref.in_corpus && ref.arxiv_id && (
                        <button className="ref-jump" onClick={() => openPaper(ref.arxiv_id as string)}>
                          open in app
                        </button>
                      )}
                      {ref.url && (
                        <a className="ref-jump" href={ref.url} target="_blank" rel="noreferrer">
                          {ref.arxiv_id ? `arXiv:${ref.arxiv_id}` : "source"}
                        </a>
                      )}
                    </div>
                  </div>
                </li>
              ))}
            </ol>
          </Section>
        </div>

        <aside className="detail-rail">
          <div className="rail-card">
            <div className="rail-act">
              <a className="af-btn pri" href={arxivUrl(paper.id, paper.url)} target="_blank" rel="noreferrer">
                <Icon name="ext" s={15} c="var(--on-accent)" />
                Read on arXiv
              </a>
            </div>
            <Link className="rail-atlas" href={`/atlas?sel=${encodeURIComponent(paper.id)}`}>
              <Icon name="scatter" s={16} c="var(--accent)" />
              <span>Locate in Embedding Atlas</span>
              <Icon name="chevR" s={15} c="var(--text-faint)" />
            </Link>
          </div>

          <div className="rail-card rail-neighbors">
            <div className="rail-h">Nearest neighbors</div>
            {(neighbors.data?.neighbors ?? []).slice(0, 6).map((neighbor) => (
              <button key={neighbor.id} className="rail-nb" onClick={() => openPaper(neighbor.id)}>
                <span className="rail-nb-bar">
                  <i style={{ width: `${Math.max(0, Math.min(1, neighbor.similarity)) * 100}%` }} />
                </span>
                <span className="rail-nb-sim">{neighbor.similarity.toFixed(2)}</span>
                <span className="rail-nb-t">{neighbor.title}</span>
              </button>
            ))}
          </div>
        </aside>
      </div>
    </div>
  );
}
