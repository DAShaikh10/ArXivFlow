// A single paper card in the Discover feed. The abstract expands inline; everything else links into
// the detail view (where embedding neighbours are fetched on demand).

"use client";

import Link from "next/link";

import { CatPill } from "@/components/shared/CatPill";
import { Icon } from "@/components/shared/Icon";
import { arxivUrl, authorStr, dateStr } from "@/lib/format";
import { type PaperSummary } from "@/lib/types";

interface PaperCardProps {
  paper: PaperSummary;
  expanded: boolean;
  onToggle: () => void;
}

export function PaperCard({ paper, expanded, onToggle }: PaperCardProps) {
  const detailHref = `/paper/${encodeURIComponent(paper.id)}`;

  return (
    <article className="feed-card">
      <div className="feed-cats">
        <CatPill clusterId={paper.cluster_id} />
        <span className="feed-id">arXiv:{paper.id}</span>
        <span className="feed-date">{dateStr(paper.published_date)}</span>
      </div>

      <Link className="feed-title" href={detailHref}>
        {paper.title}
      </Link>

      {paper.authors.length > 0 && <div className="feed-auth">{authorStr(paper.authors)}</div>}

      <div className={`feed-abs${expanded ? " open" : ""}`}>
        <p>{paper.abstract}</p>
      </div>

      <div className="feed-acts">
        <button className="af-link" onClick={onToggle}>
          <Icon name="chevD" s={14} style={{ transform: expanded ? "rotate(180deg)" : "none", transition: ".2s" }} />
          {expanded ? "Hide abstract" : "Abstract"}
        </button>
        <Link className="af-link" href={detailHref}>
          <Icon name="target" s={14} />
          Similar papers
        </Link>
        <Link className="af-link" href={detailHref}>
          <Icon name="list" s={14} />
          {paper.reference_count} references
        </Link>
        <a className="af-link" href={arxivUrl(paper.id, paper.url)} target="_blank" rel="noreferrer">
          <Icon name="ext" s={13} />
          arXiv
        </a>
        <span className="feed-cited">cited by {paper.influential_citations}</span>
        <span className="feed-match">
          prominence <b>{paper.prominence.toFixed(3)}</b>
        </span>
      </div>
    </article>
  );
}
