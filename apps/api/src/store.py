"""
In-memory data layer for the ArXivFlow API.

Loads three artifacts once at startup and serves every request from memory:

  1. The enriched paper dataset (JSONL)         -> listings, detail, references.
  2. The SPECTER2 ChromaDB collection           -> embedding nearest-neighbour queries.
  3. The precomputed atlas projection (JSON)    -> 2D coordinates + cluster colours (optional).

Canonical NER topics come from the boolean Chroma metadata keys ("{field}:{canonical}") written by papervec.

`@author`: DAShaikh10
"""

import json
import math
import re
import time
from typing import Dict, List, Optional

import chromadb
import pandas as pd
from rank_bm25 import BM25Okapi

from . import config
from .query_encoder import query_encoder

# Trailing arXiv version suffix ("2007.10310v3" -> "2007.10310"). Our paper ids carry a version,
# but scraped reference ids do not, so the two never match without stripping it.
_VERSION_SUFFIX = re.compile(r"v\d+$")

# Lowercase alphanumeric tokens for the BM25 lexical recommender — deliberately simple and identical
# to compass's `recommend.signals._tokenize`, so the served BM25 matches the one compass evaluated.
_BM25_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Tokenize a document/query into lowercase alphanumeric terms for BM25."""

    return _BM25_TOKEN.findall(text.lower())


def _normalize_id(arxiv_id: Optional[str]) -> Optional[str]:
    """Strip the version suffix so a versioned paper id and an unversioned reference id compare equal."""

    if not arxiv_id:
        return None
    return _VERSION_SUFFIX.sub("", arxiv_id.strip())


def _year_of(published_date: Optional[str]) -> Optional[int]:
    """Extract the 4-digit year from an ISO published date, tolerating missing values."""

    if not published_date or len(published_date) < 4 or not published_date[:4].isdigit():
        return None
    return int(published_date[:4])


class DataStore:
    """Owns the loaded corpus, embedding collection, and atlas projection."""

    def __init__(self) -> None:
        self._papers: List[dict] = []
        self._by_id: Dict[str, dict] = {}
        # Version-stripped id -> paper, so unversioned reference ids can resolve to a corpus paper.
        self._by_norm_id: Dict[str, dict] = {}
        self._cluster_of: Dict[str, str] = {}
        self._atlas_points: List[dict] = []
        self._categories: List[dict] = []
        self._collection: Optional[chromadb.Collection] = None
        # Corpus-wide max of log1p(influential_citations); the denominator for the authority score.
        self._max_log_cite: float = 0.0
        # BM25 lexical recommender, built over the same documents SPECTER2 embedded (see _build_bm25).
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_ids: List[str] = []
        self._bm25_tokens: List[List[str]] = []
        self._bm25_row: Dict[str, int] = {}

    def load(self) -> None:
        """Load every artifact. Raises if the dataset or embeddings are missing; atlas is optional."""

        self._load_dataset()
        self._load_embeddings()
        self._build_bm25()
        self._load_atlas()

    def _load_dataset(self) -> None:
        if not config.ENRICHED_DATASET_FILE.exists():
            raise FileNotFoundError(
                f"Enriched dataset not found at {config.ENRICHED_DATASET_FILE}. "
                "Run the `crucible` package to produce it."
            )

        frame = pd.read_json(config.ENRICHED_DATASET_FILE, lines=True)
        self._papers = frame.to_dict(orient="records")
        for paper in self._papers:
            refs = paper.get("references")
            paper["_references"] = refs if isinstance(refs, list) else []
            paper["_year"] = _year_of(paper.get("published_date"))
            # `authors` arrives once crucible's arxiv parser captures it; absent in older datasets.
            authors = paper.get("authors")
            paper["_authors"] = [a for a in authors if a] if isinstance(authors, list) else []
            self._by_id[paper["arxiv_id"]] = paper
            norm = _normalize_id(paper["arxiv_id"])
            if norm:
                self._by_norm_id[norm] = paper

        # Precompute the authority denominator: log-scaling tames the log-normal citation
        # distribution so a 1000-citation paper isn't treated as 1000x a
        # 1-citation one. Normalising by the corpus max maps every paper to [0, 1].
        self._max_log_cite = max(
            (math.log1p(int(p.get("influential_citations") or 0)) for p in self._papers),
            default=0.0,
        )

    def _load_embeddings(self) -> None:
        if not config.EMBEDDING_DATABASE_PATH.exists():
            raise FileNotFoundError(
                f"ChromaDB store not found at {config.EMBEDDING_DATABASE_PATH}. "
                "Run the `papervec` package to generate embeddings."
            )

        client = chromadb.PersistentClient(path=str(config.EMBEDDING_DATABASE_PATH))
        self._collection = client.get_collection(name=config.EMBEDDING_COLLECTION_NAME)

    def _build_bm25(self) -> None:
        """
        Build the BM25 lexical recommender index.

        Documents are the "Title[SEP]Abstract" strings read back from the Chroma collection — the exact
        text SPECTER2 embedded — so the served lexical recommender is identical to the BM25 signal
        compass evaluated (which beats dense ~30% on the honest tag-overlap yardstick). Built once at
        startup; queries reuse the cached token lists.
        """

        if self._collection is None:
            return

        record = self._collection.get(include=["documents"])
        self._bm25_ids = record["ids"]
        documents = record.get("documents") or [""] * len(self._bm25_ids)
        self._bm25_tokens = [_tokenize(doc or "") for doc in documents]
        self._bm25_row = {paper_id: row for row, paper_id in enumerate(self._bm25_ids)}
        self._bm25 = BM25Okapi(self._bm25_tokens)

    def _load_atlas(self) -> None:
        """
        Load the optional 2D projection. The API stays fully functional without it (no clusters).
        """

        if not config.ATLAS_PROJECTION_FILE.exists():
            return

        with open(config.ATLAS_PROJECTION_FILE, "r", encoding="utf-8") as handle:
            atlas = json.load(handle)

        self._categories = atlas.get("categories", [])
        for point in atlas.get("points", []):
            self._cluster_of[point["id"]] = point["cluster_id"]
            paper = self._by_id.get(point["id"])
            if paper is None:
                continue
            self._atlas_points.append(
                {
                    "id": point["id"],
                    "x": point["x"],
                    "y": point["y"],
                    "cluster_id": point["cluster_id"],
                    "title": paper.get("title", ""),
                    "published_date": paper.get("published_date"),
                    "influential_citations": int(paper.get("influential_citations") or 0),
                }
            )

    @property
    def atlas_ready(self) -> bool:
        return bool(self._atlas_points)

    def counts(self) -> tuple[int, int]:
        embeddings = self._collection.count() if self._collection else 0
        return len(self._papers), embeddings

    def _prominence(self, paper: dict) -> float:
        """
        Citation-authority score in [0, 1]: `log1p(citations)` normalised by the corpus max.

        This is a real signal (the influential-citation count from Semantic Scholar), not a relevance
        score — a context-free listing has no query or source paper to be "relevant" to. It ranks the
        default feed by standing in the field. True item-to-item relevance lives on the neighbours
        endpoint; a personalised relevance score arrives with the recommender (PLAN.md WP8).
        """

        if self._max_log_cite <= 0:
            return 0.0
        cites = int(paper.get("influential_citations") or 0)
        return round(math.log1p(cites) / self._max_log_cite, 3)

    def _summary(self, paper: dict) -> dict:
        return {
            "id": paper["arxiv_id"],
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "authors": paper["_authors"],
            "published_date": paper.get("published_date"),
            "influential_citations": int(paper.get("influential_citations") or 0),
            "reference_count": len(paper["_references"]),
            "url": paper.get("url"),
            "cluster_id": self._cluster_of.get(paper["arxiv_id"]),
            "prominence": self._prominence(paper),
        }

    def categories(self) -> List[dict]:
        return self._categories

    def years(self) -> List[int]:
        """
        Distinct publication years actually present in the corpus, ascending — the source for the
        'Since' filter options, so the UI can never offer a year with no papers behind it.
        """

        return sorted({paper["_year"] for paper in self._papers if paper.get("_year")})

    def list_papers(
        self,
        sort: str = "cited",
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        cluster_id: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 25,
        offset: int = 0,
    ) -> dict:
        """
        Filter, sort and paginate the corpus listing. `query` is a plain title substring match.
        """

        started = time.perf_counter()
        q_lower = query.strip().lower() if query else None

        rows = self._papers
        if cluster_id:
            rows = [p for p in rows if self._cluster_of.get(p["arxiv_id"]) == cluster_id]
        if year_from is not None:
            rows = [p for p in rows if (p["_year"] or 0) >= year_from]
        if year_to is not None:
            rows = [p for p in rows if (p["_year"] or 9999) <= year_to]
        if q_lower:
            rows = [p for p in rows if q_lower in p.get("title", "").lower()]

        if sort == "title":
            rows = sorted(rows, key=lambda p: p.get("title", "").lower())
        elif sort == "newest":
            rows = sorted(rows, key=lambda p: p.get("published_date") or "", reverse=True)
        else:  # "cited" (default). No personalised "relevance" sort exists yet: a context-free feed
            # has nothing to be relevant to, so ranking by influential citations is the honest default
            # until the recommender lands (PLAN.md WP8).
            rows = sorted(rows, key=lambda p: int(p.get("influential_citations") or 0), reverse=True)

        total = len(rows)
        page = rows[offset : offset + limit]
        return {
            "items": [self._summary(p) for p in page],
            "total": total,
            "limit": limit,
            "offset": offset,
            "took_ms": (time.perf_counter() - started) * 1000,
        }

    def search(self, query: str, k: int = 20, recommender: str = "rrf") -> dict:
        """
        Free-text corpus search (Semantic Search v2), via the chosen recommender.

          * "rrf" (default)   — reciprocal-rank fusion of the dense and BM25 rankings. The most balanced
                                signal on the query->doc eval (compass `search_eval`): wins on citation
                                overlap and stays competitive with BM25 on tag overlap, never losing badly.
          * "dense"           — SPECTER2 ad-hoc query embedding: the query is encoded with the
                                `allenai/specter2_adhoc_query` adapter (raw text, [CLS] pooling) and ranked
                                by cosine against the stored proximity document vectors. Semantic, by meaning.
          * "bm25"            — BM25 lexical relevance over the "Title[SEP]Abstract" documents; the same
                                lexical index that powers the BM25 "More Like This" recommender. Model-free.
                                Strongest single signal on the honest tag-overlap yardstick.

        All paths emit the same shape. `score` is per-result-set normalized to [0, 1] (top hit 1.0,
        rest descend) — only the ordering is strictly meaningful. Empty result set when nothing matches.
        """

        if recommender == "bm25":
            return self._bm25_search(query, k)
        if recommender == "dense":
            return self._dense_search(query, k)
        return self._rrf_search(query, k)

    def _rrf_search(self, query: str, k: int) -> dict:
        """
        Reciprocal-rank fusion of the dense (SPECTER2 ad-hoc query) and BM25 rankings.

        Fuses the top `config.SEARCH_FUSION_POOL` of each list by summing 1/(SEARCH_RRF_K + rank); the
        fused score is rank-based, so the two signals need no score calibration. Falls back cleanly to
        whichever list is non-empty (BM25 with no dense pool, or vice versa).
        """

        started = time.perf_counter()
        if self._collection is None or self._bm25 is None or not (query or "").strip():
            return {"query": query, "recommender": "rrf", "items": [], "total": 0,
                    "took_ms": (time.perf_counter() - started) * 1000}

        pool = config.SEARCH_FUSION_POOL

        # Dense ranking — order is all RRF needs.
        vector = query_encoder.embed(query)
        n = min(pool, self._collection.count())
        dense_ids = self._collection.query(query_embeddings=[vector], n_results=n, include=["distances"])["ids"][0]

        # BM25 ranking over the same corpus.
        scores = self._bm25.get_scores(_tokenize(query))
        bm25_rows = sorted(
            (i for i in range(len(scores)) if scores[i] > 0.0), key=lambda i: scores[i], reverse=True
        )[:pool]
        bm25_ids = [self._bm25_ids[i] for i in bm25_rows]

        # Reciprocal-rank fusion, then keep only corpus papers.
        fused: Dict[str, float] = {}
        for ranking in (dense_ids, bm25_ids):
            for rank, paper_id in enumerate(ranking, start=1):
                fused[paper_id] = fused.get(paper_id, 0.0) + 1.0 / (config.SEARCH_RRF_K + rank)
        ranked = [(pid, s) for pid, s in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
                  if pid in self._by_id][:k]

        # Min-max rescale within the result set (consistent with the dense/bm25 badges).
        values = [s for _, s in ranked]
        hi, lo = (max(values), min(values)) if values else (0.0, 0.0)
        span = hi - lo
        items: List[dict] = []
        for paper_id, score in ranked:
            summary = self._summary(self._by_id[paper_id])
            summary["score"] = round((score - lo) / span, 4) if span > 0 else 1.0
            items.append(summary)

        return {
            "query": query,
            "recommender": "rrf",
            "items": items,
            "total": len(items),
            "took_ms": (time.perf_counter() - started) * 1000,
        }

    def _dense_search(self, query: str, k: int) -> dict:
        """
        SPECTER2 ad-hoc query nearest neighbours: embed the query, rank the corpus by cosine.
        """

        started = time.perf_counter()
        if self._collection is None or not (query or "").strip():
            return {"query": query, "recommender": "dense", "items": [], "total": 0,
                    "took_ms": (time.perf_counter() - started) * 1000}

        vector = query_encoder.embed(query)
        result = self._collection.query(query_embeddings=[vector], n_results=k, include=["distances"])
        ids = result["ids"][0]
        distances = result["distances"][0]

        # Chroma returns cosine distance in [0, 2]; similarity = 1 - distance. Cross-adapter cosines sit
        # in a narrow high band (~0.8), so we min-max rescale within the result set — top hit -> 1.0,
        # last -> 0.0 — to convey ranking confidence in the badge (as BM25's top-normalization does).
        kept = [(paper_id, 1.0 - float(distance)) for paper_id, distance in zip(ids, distances)
                if paper_id in self._by_id]
        sims = [sim for _, sim in kept]
        hi, lo = (max(sims), min(sims)) if sims else (0.0, 0.0)
        span = hi - lo

        items: List[dict] = []
        for paper_id, sim in kept:
            summary = self._summary(self._by_id[paper_id])
            summary["score"] = round((sim - lo) / span, 4) if span > 0 else 1.0
            items.append(summary)

        return {
            "query": query,
            "recommender": "dense",
            "items": items,
            "total": len(items),
            "took_ms": (time.perf_counter() - started) * 1000,
        }

    def _bm25_search(self, query: str, k: int) -> dict:
        """
        BM25 lexical corpus search. Scores are normalized per query by the top hit so the best result is
        1.0 and the rest descend; only the ordering is strictly meaningful. Returns an empty result set
        when the query has no scorable terms or no paper shares a term with it.
        """

        started = time.perf_counter()
        tokens = _tokenize(query or "")
        if self._bm25 is None or not tokens:
            return {"query": query, "recommender": "bm25", "items": [], "total": 0,
                    "took_ms": (time.perf_counter() - started) * 1000}

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            (i for i in range(len(scores)) if scores[i] > 0.0),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        top = float(scores[ranked[0]]) if ranked else 0.0
        items: List[dict] = []
        for i in ranked:
            paper = self._by_id.get(self._bm25_ids[i])
            if paper is None:
                continue
            summary = self._summary(paper)
            summary["score"] = round(float(scores[i]) / top, 4) if top else 0.0
            items.append(summary)

        return {
            "query": query,
            "recommender": "bm25",
            "items": items,
            "total": len(items),
            "took_ms": (time.perf_counter() - started) * 1000,
        }

    def get_paper(self, arxiv_id: str) -> Optional[dict]:
        """Full detail for one paper, with corpus-membership flags on references and canonical topics."""

        paper = self._by_id.get(arxiv_id)
        if paper is None:
            return None

        references = []
        for ref in paper["_references"]:
            if not ref.get("title"):
                continue
            # Reference ids are unversioned; match on the normalized id, then expose the canonical
            # versioned id so the UI's "open in app" deep-link resolves in the store and in Chroma.
            corpus_paper = self._by_norm_id.get(_normalize_id(ref.get("arxiv_id")) or "")
            references.append(
                {
                    "arxiv_id": corpus_paper["arxiv_id"] if corpus_paper else ref.get("arxiv_id"),
                    "title": ref.get("title", ""),
                    "url": ref.get("url"),
                    "in_corpus": corpus_paper is not None,
                }
            )

        detail = self._summary(paper)
        detail["references"] = references
        detail["topics"] = self._topics_for(arxiv_id)
        return detail

    def _topics_for(self, arxiv_id: str) -> List[dict]:
        """
        Read canonical NER tags from Chroma metadata ('{field}:{canonical}' boolean keys).
        """

        if self._collection is None:
            return []

        record = self._collection.get(ids=[arxiv_id], include=["metadatas"])
        metadatas = record.get("metadatas") or []
        if not metadatas or not metadatas[0]:
            return []

        topics: List[dict] = []
        for key, present in metadatas[0].items():
            if present and ":" in key:
                field, value = key.split(":", 1)
                topics.append({"field": field, "value": value})
        topics.sort(key=lambda t: (t["field"], t["value"]))
        return topics

    def _neighbor_entry(self, neighbor_id: str, similarity: float) -> Optional[dict]:
        """
        Build one neighbour card, or None if the id is not in the loaded corpus.
        """

        paper = self._by_id.get(neighbor_id)
        if paper is None:
            return None
        return {
            "id": neighbor_id,
            "title": paper.get("title", ""),
            "authors": paper["_authors"],
            "similarity": round(similarity, 4),
            "published_date": paper.get("published_date"),
            "influential_citations": int(paper.get("influential_citations") or 0),
            "cluster_id": self._cluster_of.get(neighbor_id),
        }

    def neighbors(self, arxiv_id: str, k: int = 8, recommender: str = "dense") -> Optional[dict]:
        """
        Top-k related papers for a paper, via the chosen recommender.

          * "dense" (default) — SPECTER2 cosine nearest neighbours (the deployed Tier 0 recommender).
          * "bm25"            — BM25 lexical similarity over Title+Abstract; the strongest signal on the
                                honest tag-overlap yardstick (~30% higher nDCG than dense).

        Returns None if the paper is unknown. Both paths emit the same shape; `similarity` is cosine
        in [0, 1] for dense and per-query-normalized BM25 in (0, 1] for bm25 (only the order is meaningful).
        """

        if arxiv_id not in self._by_id:
            return None
        if recommender == "bm25":
            return self._bm25_neighbors(arxiv_id, k)
        return self._dense_neighbors(arxiv_id, k)

    def _dense_neighbors(self, arxiv_id: str, k: int) -> Optional[dict]:
        """
        Cosine nearest neighbours for a paper, straight from the SPECTER2 HNSW index.
        """

        if self._collection is None:
            return None

        started = time.perf_counter()
        record = self._collection.get(ids=[arxiv_id], include=["embeddings"])
        embeddings = record.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return None

        # Fetch k+1 because the closest hit is the paper itself (distance 0).
        result = self._collection.query(
            query_embeddings=[embeddings[0]],
            n_results=k + 1,
            include=["distances"],
        )
        ids = result["ids"][0]
        distances = result["distances"][0]

        neighbors: List[dict] = []
        for neighbor_id, distance in zip(ids, distances):
            if neighbor_id == arxiv_id:
                continue
            # Chroma returns cosine distance in [0, 2]; similarity = 1 - distance.
            entry = self._neighbor_entry(neighbor_id, 1.0 - float(distance))
            if entry is not None:
                neighbors.append(entry)
            if len(neighbors) >= k:
                break

        return {
            "source_id": arxiv_id,
            "recommender": "dense",
            "neighbors": neighbors,
            "took_ms": (time.perf_counter() - started) * 1000,
        }

    def _bm25_neighbors(self, arxiv_id: str, k: int) -> Optional[dict]:
        """
        BM25 lexical neighbours: score the corpus with the source paper's own document as the query.
        """

        row = self._bm25_row.get(arxiv_id)
        if self._bm25 is None or row is None:
            return None

        started = time.perf_counter()
        scores = self._bm25.get_scores(self._bm25_tokens[row])

        # Rank by score, dropping the source itself and any zero-overlap papers (mirrors compass).
        ranked = sorted(
            (i for i in range(len(scores)) if i != row and scores[i] > 0.0),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        # BM25 is unbounded, but the client expects a similarity-like field; normalize per query by the
        # top hit so the best neighbour is 1.0 and the rest descend. Only the ordering is meaningful.
        top = float(scores[ranked[0]]) if ranked else 0.0
        neighbors: List[dict] = []
        for i in ranked:
            entry = self._neighbor_entry(self._bm25_ids[i], float(scores[i]) / top if top else 0.0)
            if entry is not None:
                neighbors.append(entry)

        return {
            "source_id": arxiv_id,
            "recommender": "bm25",
            "neighbors": neighbors,
            "took_ms": (time.perf_counter() - started) * 1000,
        }

    def atlas(self) -> dict:
        return {
            "points": self._atlas_points,
            "categories": self._categories,
            "count": len(self._atlas_points),
        }


# Module-level singleton, populated by the FastAPI lifespan handler.
store = DataStore()
