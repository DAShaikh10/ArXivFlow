"""
Recommender evaluation — entry point.

`@author`: DAShaikh10
"""

import json
import os
import time
from typing import Dict, List, Sequence, Set

import chromadb

from src.utils import logger, resolve_path

from . import config, groundtruth, metrics


def _mean(values: List[float]) -> float:
    """
    Arithmetic mean, 0.0 for an empty sequence.
    """

    return sum(values) / len(values) if values else 0.0


def rank_neighbors(
    collection: chromadb.Collection,
    ids: Sequence[str],
    embeddings: Sequence,
    max_k: int,
) -> Dict[str, List[str]]:
    """
    Top-`max_k` ranked neighbour ids per paper, via the same cosine HNSW query the API serves.

    Fetches `max_k + 1` because the closest hit is the paper itself (distance 0), which is dropped.
    """

    rankings: Dict[str, List[str]] = {}
    for paper_id, embedding in zip(ids, embeddings):
        vector = embedding.tolist() if hasattr(embedding, "tolist") else embedding
        result = collection.query(query_embeddings=[vector], n_results=max_k + 1)
        ranked = [neighbor_id for neighbor_id in result["ids"][0] if neighbor_id != paper_id]
        rankings[paper_id] = ranked[:max_k]
    return rankings


def evaluate_signal(
    rankings: Dict[str, List[str]],
    positives: Dict[str, Set[str]],
    k_values: List[int],
) -> dict:
    """
    Macro-average every metric over the source papers that have at least one positive.
    """

    sources = [source for source in rankings if positives.get(source)]

    scores: Dict[str, float] = {}
    for k in k_values:
        scores[f"precision@{k}"] = _mean([metrics.precision_at_k(rankings[s], positives[s], k) for s in sources])
        scores[f"recall@{k}"] = _mean([metrics.recall_at_k(rankings[s], positives[s], k) for s in sources])
        scores[f"ndcg@{k}"] = _mean([metrics.ndcg_at_k(rankings[s], positives[s], k) for s in sources])
    scores["mrr"] = _mean([metrics.reciprocal_rank(rankings[s], positives[s]) for s in sources])

    return {
        "coverage": len(sources),
        "avg_positives": _mean([float(len(positives[s])) for s in sources]),
        "metrics": {key: round(value, 4) for key, value in scores.items()},
    }


def _print_summary(report: dict) -> None:
    """
    Console table: one column per signal.
    """

    signals = report["signals"]
    metric_keys = list(next(iter(signals.values()))["metrics"].keys())
    names = list(signals.keys())

    header = f"{'metric':<14}" + "".join(f"{name:>20}" for name in names)
    logger.info(header)
    logger.info("-" * len(header))
    cov = f"{'coverage':<14}" + "".join(f"{signals[n]['coverage']:>20}" for n in names)
    logger.info(cov)
    avg = f"{'avg_positives':<14}" + "".join(f"{signals[n]['avg_positives']:>20.2f}" for n in names)
    logger.info(avg)
    for key in metric_keys:
        row = f"{key:<14}" + "".join(f"{signals[n]['metrics'][key]:>20.4f}" for n in names)
        logger.info(row)


def main() -> None:  # pylint: disable=too-many-locals
    """
    Run the Tier 0 ground-truth eval across signals and write the report under `data/`.
    """

    current_dir = os.path.dirname(__file__)
    database_path = resolve_path(current_dir, config.EMBEDDING_DATABASE_NAME)
    dataset_path = resolve_path(current_dir, config.ENRICHED_DATASET_FILE)
    annotation_path = resolve_path(current_dir, config.ANNOTATION_FILE)
    canonical_map_path = resolve_path(current_dir, config.CANONICAL_MAP_FILE)

    logger.info("Loading Chroma collection '%s' from %s", config.EMBEDDING_COLLECTION_NAME, database_path)
    client = chromadb.PersistentClient(path=database_path)
    collection = client.get_collection(name=config.EMBEDDING_COLLECTION_NAME)

    record = collection.get(include=["embeddings"])
    ids = record["ids"]
    embeddings = record["embeddings"]
    logger.info("Corpus: %d papers", len(ids))

    # Ground truth. Tags come from the gold human annotations, not Chroma metadata.
    tag_sets = groundtruth.tag_sets_from_annotations(annotation_path, canonical_map_path, ids)
    tagged = sum(1 for tags in tag_sets.values() if tags)
    logger.info("Papers carrying >=1 canonical tag: %d", tagged)

    tag_positives = groundtruth.build_tag_positives(tag_sets, config.TAG_MIN_SHARED, config.TAG_JACCARD_MIN)
    citation_positives = groundtruth.build_citation_positives(dataset_path, ids)
    logger.info(
        "Positive coverage — tag-overlap (shared>=%d, J>=%.2f): %d papers, citation-overlap: %d papers",
        config.TAG_MIN_SHARED,
        config.TAG_JACCARD_MIN,
        sum(1 for s in tag_positives if tag_positives[s]),
        sum(1 for s in citation_positives if citation_positives[s]),
    )

    # Rankings from the actual served query path.
    max_k = max(config.K_VALUES)
    started = time.perf_counter()
    rankings = rank_neighbors(collection, ids, embeddings, max_k)
    elapsed = time.perf_counter() - started
    logger.info("Ranked %d papers (top-%d each) in %.2fs", len(rankings), max_k, elapsed)

    report = {
        "recommender": "specter2-proximity cosine kNN (Chroma HNSW)",
        "corpus_size": len(ids),
        "papers_tagged": tagged,
        "k_values": config.K_VALUES,
        "tag_min_shared": config.TAG_MIN_SHARED,
        "tag_jaccard_min": config.TAG_JACCARD_MIN,
        "signals": {
            "tag_overlap": evaluate_signal(rankings, tag_positives, config.K_VALUES),
            "citation_overlap": evaluate_signal(rankings, citation_positives, config.K_VALUES),
        },
    }

    report_path = resolve_path(current_dir, config.EVAL_REPORT_FILE)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logger.info("Wrote report to %s", report_path)

    _print_summary(report)


if __name__ == "__main__":
    main()
