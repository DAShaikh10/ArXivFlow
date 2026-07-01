// Wire types mirroring the FastAPI responses (apps/api/src/schemas.py). Field names stay snake_case
// to match the JSON exactly — no client-side remapping.

export interface Category {
  id: string;
  name: string;
  color: string;
  count: number;
}

export interface Reference {
  arxiv_id: string | null;
  title: string;
  url: string | null;
  in_corpus: boolean;
}

export interface Topic {
  field: string;
  value: string;
}

export interface PaperSummary {
  id: string;
  title: string;
  abstract: string;
  authors: string[];
  published_date: string | null;
  influential_citations: number;
  reference_count: number;
  url: string | null;
  cluster_id: string | null;
  /** Citation-authority score (0..1): log-normalised influential citations. A real standing signal,
   *  not relevance — see store._prominence. */
  prominence: number;
}

export interface PaperDetail extends PaperSummary {
  references: Reference[];
  topics: Topic[];
}

export interface Neighbor {
  id: string;
  title: string;
  authors: string[];
  similarity: number;
  published_date: string | null;
  influential_citations: number;
  cluster_id: string | null;
}

/** Which "More Like This" recommender to query. dense = SPECTER2 cosine; bm25 = lexical over Title+Abstract. */
export type Recommender = "dense" | "bm25";

export interface NeighborsResponse {
  source_id: string;
  /** Echoes the recommender that produced these results. */
  recommender: Recommender;
  neighbors: Neighbor[];
  took_ms: number;
}

export interface SearchResult extends PaperSummary {
  /** Per-query-normalised relevance (0..1): the top hit is 1.0 and the rest descend. */
  score: number;
}

export interface SearchResponse {
  query: string;
  /** Echoes the recommender that produced these results (currently "bm25"). */
  recommender: string;
  items: SearchResult[];
  total: number;
  took_ms: number;
}

export interface PaperListResponse {
  items: PaperSummary[];
  total: number;
  limit: number;
  offset: number;
  took_ms: number;
}

export interface AtlasPoint {
  id: string;
  x: number;
  y: number;
  cluster_id: string;
  title: string;
  published_date: string | null;
  influential_citations: number;
}

export interface AtlasResponse {
  points: AtlasPoint[];
  categories: Category[];
  count: number;
}

export type SortKey = "cited" | "newest" | "title";

export interface PaperListParams {
  sort?: SortKey;
  year_from?: number;
  year_to?: number;
  cluster_id?: string;
  q?: string;
  limit?: number;
  offset?: number;
}
