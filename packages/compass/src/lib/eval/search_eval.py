"""
Query -> document search evaluation: SPECTER2 ad-hoc query (dense) vs BM25 (lexical) vs their RRF fusion.

The Tier harnesses evaluate the *doc -> doc* regime (a paper's own vector/document is the
query), where BM25 leads on tag overlap. Free-text **search** is a different regime: a short
natural-language query -> a topical set of papers. This is exactly what the SPECTER2 `adhoc_query`
adapter is trained for and where BM25 has few terms to match, so the doc->doc verdict does not carry
over — this harness measures it directly, to justify (or not) making dense the default search recommender.

Ground truth (the whole experiment): query = paper P; **positives = P's tag-overlap neighbours from the
human gold** (`build_tag_positives`, shared >= TAG_MIN_SHARED), **with P excluded**. This is the same
embedding-independent label the headline doc->doc test uses (honest for dense *and* bm25 alike), and
dropping P removes the known-item exact-match leak (P's title is embedded verbatim in P's own document).
Queries are built from title/abstract text, never from the tag vocabulary, so the label stays independent
of the query — the circularity rule `recommend.eval` enforces.

Two query sets over the same positives:
  * title      — query = P's title. Zero-cost, every eligible paper. A short concept phrase.
  * paraphrase — query = an LLM paraphrase of P's information need (natural language, not title n-grams),
                 read from SEARCH_QUERIES_FILE for a fixed sample. The cleaner test of semantic value.
                 When that file is absent, this module writes the sample's title+abstract to
                 SEARCH_SAMPLE_FILE so the paraphrases can be authored, then skips the paraphrase table.

Faithfulness to the shipped API: `apps/api` embeds queries with the same base+adhoc_query adapter and
[CLS] pooling; it omits the explicit `set_active_adapters` and L2-norm that `embed.py` does, but both were
measured to be no-ops for ranking (explicit-vs-implicit activation cos=1.0; L2-norm cannot change cosine
*order*). So these rankings equal the deployed dense-search path — do not "reconcile" that difference away.

`@author`: DAShaikh10
"""

import json
import os
import random
import time
from typing import Dict, List, Sequence, Set, Tuple

import chromadb

from src.utils import logger, resolve_path

from ..recommend import config as rec_config
from ..recommend.corpus import Corpus
from ..recommend.embed import get_embedder
from ..recommend.fusion import rrf
from ..recommend.signals import Signals, _tokenize  # _tokenize: the exact tokenizer the BM25 index uses
from . import config, groundtruth
from .main import evaluate_signal

# Fixed sample size for the LLM-paraphrase query set (the title set uses every eligible paper).
SAMPLE_SIZE = int(os.getenv("SEARCH_SAMPLE_SIZE", "150"))
SAMPLE_SEED = int(os.getenv("RANDOM_SEED", "42"))
# Authored natural-language paraphrase queries (JSONL: {"arxiv_id": ..., "query": ...}).
SEARCH_QUERIES_FILE = os.getenv("SEARCH_QUERIES_FILE", "search_paraphrase_queries.jsonl")
# Where the sample's title+abstract is dumped when the paraphrase file is missing.
SEARCH_SAMPLE_FILE = os.getenv("SEARCH_SAMPLE_FILE", "search_paraphrase_sample.jsonl")
SEARCH_REPORT_FILE = os.getenv("SEARCH_REPORT_FILE", "search_eval_report.json")


def _load_paper_text(dataset_path: str) -> Dict[str, Tuple[str, str]]:
    """
    Map version-stripped arXiv id -> (title, abstract) from the enriched dataset.
    """

    text: Dict[str, Tuple[str, str]] = {}
    with open(dataset_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            paper = json.loads(line)
            norm = groundtruth.normalize_id(paper.get("arxiv_id"))
            if norm:
                text[norm] = (paper.get("title") or "", paper.get("abstract") or "")
    return text


def _title_of(paper_id: str, text: Dict[str, Tuple[str, str]]) -> str:
    return text.get(groundtruth.normalize_id(paper_id) or "", ("", ""))[0]


def _select_sample(sources: Sequence[str], n: int) -> List[str]:
    """
    Deterministic sample of eligible source ids (seeded, id-sorted for reproducibility).
    """

    ordered = sorted(sources)
    if len(ordered) <= n:
        return ordered
    return sorted(random.Random(SAMPLE_SEED).sample(ordered, n))


# pylint: disable=too-many-locals


def _rank_ids(
    signals: Signals,
    corpus: Corpus,
    queries: Dict[str, str],
    max_k: int,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Rank every query through dense, bm25 and their RRF fusion; return {signal: {source_id: [ranked ids]}}.

    The source paper is excluded from its own results (it is not a positive, and its title is embedded
    verbatim in its own document — leaving it in would just be a known-item distractor). Dense queries are
    batch-encoded once through the adhoc_query adapter for speed.
    """

    source_ids = list(queries.keys())
    rows = [corpus.index[pid] for pid in source_ids]

    embedder = get_embedder("adhoc_query")
    logger.info("Embedding %d queries through the SPECTER2 adhoc_query adapter...", len(source_ids))
    vectors = embedder.encode([queries[pid] for pid in source_ids])

    dense: Dict[str, List[str]] = {}
    lexical: Dict[str, List[str]] = {}
    fused: Dict[str, List[str]] = {}
    for source_id, row, vector in zip(source_ids, rows, vectors):
        dense_ranked = signals.dense_from_vector(vector, limit=max_k, exclude=row)
        bm25_ranked = signals.lexical_from_tokens(_tokenize(queries[source_id]), limit=max_k, exclude=row)
        rrf_ranked = rrf([[r for r, _ in dense_ranked], [r for r, _ in bm25_ranked]], k=rec_config.RRF_K)[:max_k]

        dense[source_id] = [corpus.ids[r] for r, _ in dense_ranked]
        lexical[source_id] = [corpus.ids[r] for r, _ in bm25_ranked]
        fused[source_id] = [corpus.ids[r] for r, _ in rrf_ranked]

    return {"dense": dense, "bm25": lexical, "rrf": fused}


# pylint: enable=too-many-locals


def _evaluate_query_set(
    signals: Signals,
    corpus: Corpus,
    queries: Dict[str, str],
    positives: Dict[str, Set[str]],
    k_values: List[int],
) -> dict:
    """
    Rank all queries and score each signal against the shared tag-overlap positives.
    """

    rankings = _rank_ids(signals, corpus, queries, max(k_values))
    return {
        "n_queries": len(queries),
        "signals": {name: evaluate_signal(ranked, positives, k_values) for name, ranked in rankings.items()},
    }


def _print_table(title: str, result: dict) -> None:
    """
    Console table for one query set: one column per signal.
    """

    signals = result["signals"]
    names = list(signals.keys())
    metric_keys = list(next(iter(signals.values()))["metrics"].keys())

    logger.info("=== %s (%d queries) ===", title, result["n_queries"])
    header = f"{'metric':<14}" + "".join(f"{name:>12}" for name in names)
    logger.info(header)
    logger.info("-" * len(header))
    for key in metric_keys:
        row = f"{key:<14}" + "".join(f"{signals[n]['metrics'][key]:>12.4f}" for n in names)
        logger.info(row)


def _dump_sample(sample: List[str], text: Dict[str, Tuple[str, str]], sample_path: str) -> None:
    """
    Write the paraphrase sample's title+abstract so natural-language queries can be authored.
    """

    with open(sample_path, "w", encoding="utf-8") as handle:
        for paper_id in sample:
            title, abstract = text.get(groundtruth.normalize_id(paper_id) or "", ("", ""))
            handle.write(json.dumps({"arxiv_id": paper_id, "title": title, "abstract": abstract}) + "\n")
    logger.info(
        "Wrote %d sample papers to %s — author paraphrase queries into %s, then re-run.",
        len(sample),
        sample_path,
        SEARCH_QUERIES_FILE,
    )


def _load_paraphrase_queries(queries_path: str, eligible: Set[str]) -> Dict[str, str]:
    """
    Load authored paraphrase queries (JSONL), keeping only eligible sources with a non-empty query.
    """

    queries: Dict[str, str] = {}
    with open(queries_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            paper_id, query = record.get("arxiv_id"), (record.get("query") or "").strip()
            if paper_id in eligible and query:
                queries[paper_id] = query
    return queries


# pylint: disable=too-many-locals


def main() -> None:
    """
    Run the query->doc search eval (title always; paraphrase when its queries file exists).
    """

    current_dir = os.path.dirname(__file__)
    database_path = resolve_path(current_dir, config.EMBEDDING_DATABASE_NAME)
    dataset_path = resolve_path(current_dir, config.ENRICHED_DATASET_FILE)
    annotation_path = resolve_path(current_dir, config.ANNOTATION_FILE)  # human gold
    canonical_map_path = resolve_path(current_dir, config.CANONICAL_MAP_FILE)
    queries_path = resolve_path(current_dir, SEARCH_QUERIES_FILE)
    sample_path = resolve_path(current_dir, SEARCH_SAMPLE_FILE)
    report_path = resolve_path(current_dir, SEARCH_REPORT_FILE)

    logger.info("Loading Chroma collection '%s' from %s", config.EMBEDDING_COLLECTION_NAME, database_path)
    client = chromadb.PersistentClient(path=database_path)
    collection = client.get_collection(name=config.EMBEDDING_COLLECTION_NAME)
    corpus = Corpus.load(collection, dataset_path, annotation_path, canonical_map_path)
    logger.info("Corpus: %d papers", len(corpus.ids))

    # Ground truth: tag-overlap positives from the human gold (self excluded by construction).
    tag_sets = groundtruth.tag_sets_from_annotations(annotation_path, canonical_map_path, corpus.ids)
    positives = groundtruth.build_tag_positives(tag_sets, config.TAG_MIN_SHARED, config.TAG_JACCARD_MIN)
    eligible = {pid for pid in corpus.ids if positives.get(pid)}
    logger.info("Eligible sources (>=1 tag-overlap positive, shared>=%d): %d", config.TAG_MIN_SHARED, len(eligible))

    text = _load_paper_text(dataset_path)
    signals = Signals(corpus)

    # --- Query set 1: title (zero cost, every eligible paper) ---
    title_queries = {pid: _title_of(pid, text) for pid in eligible if _title_of(pid, text)}
    started = time.perf_counter()
    title_result = _evaluate_query_set(signals, corpus, title_queries, positives, config.K_VALUES)
    logger.info("Title query set evaluated in %.1fs", time.perf_counter() - started)

    query_sets: Dict[str, dict] = {"title": title_result}

    # --- Query set 2: paraphrase (fixed sample; needs authored queries) ---
    sample = _select_sample(sorted(eligible), SAMPLE_SIZE)
    if os.path.exists(queries_path):
        paraphrase_queries = _load_paraphrase_queries(queries_path, eligible)
        logger.info("Loaded %d paraphrase queries from %s", len(paraphrase_queries), queries_path)
        if paraphrase_queries:
            query_sets["paraphrase"] = _evaluate_query_set(
                signals, corpus, paraphrase_queries, positives, config.K_VALUES
            )
    else:
        _dump_sample(sample, text, sample_path)

    report = {
        "task": "query->document search",
        "ground_truth": f"tag-overlap human gold (shared>={config.TAG_MIN_SHARED}), source paper excluded",
        "recommenders": ["dense (SPECTER2 adhoc_query)", "bm25 (lexical)", "rrf (dense+bm25)"],
        "corpus_size": len(corpus.ids),
        "eligible_sources": len(eligible),
        "k_values": config.K_VALUES,
        "tag_min_shared": config.TAG_MIN_SHARED,
        "query_sets": query_sets,
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logger.info("Wrote report to %s", report_path)

    for name, result in query_sets.items():
        _print_table(name, result)


# pylint: enable=too-many-locals

if __name__ == "__main__":
    main()
