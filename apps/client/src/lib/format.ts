// Small presentation helpers shared across views.

/** "2019-01-25T05:57:24Z" -> "Jan 25, 2019". Tolerates null/invalid dates. */
export function dateStr(iso: string | null | undefined): string {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

/** Year as a number, or null. */
export function yearOf(iso: string | null | undefined): number | null {
  if (!iso || iso.length < 4) return null;
  const year = Number(iso.slice(0, 4));
  return Number.isInteger(year) ? year : null;
}

/** Format milliseconds the way the latency badge expects: "<1", "7.4", "120". */
export function fmtMs(ms: number | null): string {
  if (ms === null) return "—";
  if (ms < 1) return "<1";
  if (ms < 10) return ms.toFixed(1);
  return Math.round(ms).toString();
}

/** The arXiv abstract URL for an id (uses the stored url when present). */
export function arxivUrl(id: string, stored?: string | null): string {
  return stored ?? `https://arxiv.org/abs/${id}`;
}

/** Join an author list, truncating to `max` names with a "+N" overflow. Empty list -> "". */
export function authorStr(authors: string[] | null | undefined, max = 4): string {
  if (!authors || authors.length === 0) return "";
  if (authors.length <= max) return authors.join(", ");
  return `${authors.slice(0, max).join(", ")} +${authors.length - max}`;
}

/** Turn a canonical topic field key ("target_nlp_task") into a readable label. */
export function topicFieldLabel(field: string): string {
  return field
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}
