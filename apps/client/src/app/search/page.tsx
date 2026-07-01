"use client";

import Link from "next/link";
import { useCallback, useState } from "react";

import { CatPill } from "@/components/shared/CatPill";
import { Icon } from "@/components/shared/Icon";
import { EmptyState, ErrorState, Loading } from "@/components/shared/States";
import { api } from "@/lib/api";
import { authorStr, dateStr, fmtMs } from "@/lib/format";
import { type SearchResponse } from "@/lib/types";
import { useApi } from "@/lib/useApi";

const SAMPLE_QUERIES = [
  "low-resource neural machine translation",
  "contrastive sentence representations",
  "retrieval-augmented question answering",
];

const EMPTY: SearchResponse = { query: "", recommender: "rrf", items: [], total: 0, took_ms: 0 };

export default function SearchPage() {
  const [input, setInput] = useState("");
  const [query, setQuery] = useState("");

  const { data, error, loading } = useApi(
    useCallback(
      (signal: AbortSignal) =>
        query.trim() ? api.search(query.trim(), 20, signal) : Promise.resolve({ data: EMPTY, tookMs: 0 }),
      [query],
    ),
    [query],
  );

  const submit = (next: string) => {
    setInput(next);
    setQuery(next);
  };

  const results = data?.items ?? [];
  const hasQuery = query.trim().length > 0;
  const searching = hasQuery && loading;

  return (
    <div className="sv2-wrap">
      <Link className="detail-back" href="/">
        <Icon name="arrowL" s={16} />
        Back to Discover
      </Link>

      <div className="sv2-hero">
        <div className="sv2-eyebrow">
          <span className="sv2-eyebrow-dot" />
          Search over the indexed corpus
        </div>
        <h1 className="sv2-h1">
          <span className="sv2-h1-a">Describe the idea.</span>
          <span className="sv2-h1-b">We&rsquo;ll find the paper.</span>
        </h1>
        <p className="sv2-lede">
          Type a natural-language query and we&rsquo;ll rank the corpus by fusing two signals — SPECTER2 semantic
          similarity and BM25 keyword relevance — so results match on meaning <em>and</em> wording.
        </p>

        <form
          className="sv2-searchbox"
          onSubmit={(event) => {
            event.preventDefault();
            submit(input);
          }}
        >
          <Icon name="search" s={20} c="var(--text-faint)" />
          <input
            className="sv2-input"
            type="search"
            value={input}
            autoFocus
            placeholder="diffusion models for protein folding with limited data"
            onChange={(event) => setInput(event.target.value)}
          />
          <button className="sv2-go" type="submit" disabled={!input.trim()}>
            Search
          </button>
        </form>

        <div className="sv2-chips">
          {SAMPLE_QUERIES.map((sample) => (
            <button key={sample} className="sv2-chip" type="button" onClick={() => submit(sample)}>
              {sample}
            </button>
          ))}
        </div>
      </div>

      {hasQuery && (
        <div className="sv2-preview">
          <div className="sv2-preview-bar">
            <span>
              {searching ? "Searching…" : `${results.length} result${results.length === 1 ? "" : "s"}`}
              {!searching && results.length > 0 ? ` for “${query.trim()}”` : ""}
            </span>
            {!searching && data ? (
              <span className="sv2-lat">
                {data.recommender} · {fmtMs(data.took_ms)} ms
              </span>
            ) : null}
          </div>

          {error ? (
            <ErrorState message={error} />
          ) : searching ? (
            <Loading message="Ranking the corpus…" />
          ) : results.length === 0 ? (
            <EmptyState message={`No papers matched “${query.trim()}”. Try different or broader terms.`} />
          ) : (
            <div className="sv2-results sv2-live">
              {results.map((paper, index) => (
                <Link key={paper.id} className="sv2-res" href={`/paper/${encodeURIComponent(paper.id)}`}>
                  <span className="sv2-rank">#{index + 1}</span>
                  <div className="sv2-res-score">{paper.score.toFixed(2)}</div>
                  <div className="sv2-res-body">
                    <div className="sv2-res-meta">
                      <CatPill clusterId={paper.cluster_id} />
                      <span>{dateStr(paper.published_date)}</span>
                      <span>cited by {paper.influential_citations}</span>
                    </div>
                    <h4>{paper.title}</h4>
                    {paper.authors.length > 0 && <div className="sv2-res-auth">{authorStr(paper.authors)}</div>}
                    <p>{paper.abstract.slice(0, 220)}…</p>
                  </div>
                </Link>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
