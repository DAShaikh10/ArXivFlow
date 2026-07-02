"""
Stage 2 reranker evaluation.

Measures the lift a cross-encoder reranker gives over a Stage 1 candidate pool, on a sample of source
papers (reranking is too expensive to run over all 992 × pool pairs for a quick check). For each source
the cross-encoder scores (source-document, candidate-document) pairs and reorders the pool; we compare
the reranked top-k against the Stage 1 (no-op) top-k.

A reranker can only reorder what Stage 1 retrieved, so we also report **Stage-1 recall@pool** — the
ceiling the reranker is working under. Citation-overlap is the headline ground truth here (it is not
circular with any retrieval signal and, post-rebuild, is where dense is strongest).

Run: `REC_CROSS_ENCODER_MODEL=... uv run -m src.lib.recommend.rerank_eval`
     (or `moon run compass:evaluate-rerank`).

`@author`: DAShaikh10
"""

import os
import time
from typing import Dict, List

import chromadb

from src.utils import logger, resolve_path

from ..eval import config as eval_config
from ..eval import groundtruth
from ..eval.main import _mean, evaluate_signal
from . import config
from .corpus import Corpus
from .pipeline import Recommender, RetrievalConfig
from .rerank import CrossEncoderReranker, NoOpReranker

# Stage 1 backbone to rerank. Default to the fused pool so we measure the real two-stage system (Stage1+2
# vs Stage1) with a high recall ceiling; set REC_RERANK_STAGE1=dense to rerank dense-only instead.
_POOL = int(os.getenv("REC_RERANK_POOL", "50"))
_STAGE1 = RetrievalConfig(pool_size=_POOL, use_fusion=os.getenv("REC_RERANK_STAGE1", "fused") != "dense")
_SAMPLE = int(os.getenv("REC_RERANK_SAMPLE", "120"))


def _recall_at_pool(recommender: Recommender, sample: List[str], positives: Dict[str, set]) -> float:
    """
    Macro recall of the candidate pool itself — the ceiling any reranker is bounded by.
    """

    recalls = []
    for paper_id in sample:
        row = recommender.corpus.index[paper_id]
        pool = {recommender.corpus.ids[c] for c in recommender.candidate_rows(row, _STAGE1)}
        relevant = positives.get(paper_id) or set()
        if relevant:
            recalls.append(len(pool & relevant) / len(relevant))
    return _mean(recalls)


def main() -> None:  # pylint: disable=too-many-locals
    """
    Evaluate the configured reranker over the candidate pool and report pool/rerank recall.
    """
    current_dir = os.path.dirname(__file__)
    client = chromadb.PersistentClient(path=resolve_path(current_dir, config.EMBEDDING_DATABASE_NAME))
    collection = client.get_collection(name=config.EMBEDDING_COLLECTION_NAME)

    corpus = Corpus.load(
        collection,
        resolve_path(current_dir, config.ENRICHED_DATASET_FILE),
        resolve_path(current_dir, config.REC_ANNOTATION_FILE),
        resolve_path(current_dir, config.CANONICAL_MAP_FILE),
        tag_source=config.REC_TAG_SOURCE,
    )

    gold_tags = groundtruth.tag_sets_from_annotations(
        resolve_path(current_dir, eval_config.ANNOTATION_FILE),
        resolve_path(current_dir, config.CANONICAL_MAP_FILE),
        corpus.ids,
    )
    dataset_path = resolve_path(current_dir, config.ENRICHED_DATASET_FILE)
    ground_truth = {
        "tag_overlap": groundtruth.build_tag_positives(gold_tags, config.TAG_MIN_SHARED, config.TAG_JACCARD_MIN),
        "citation_overlap": groundtruth.build_citation_positives(dataset_path, corpus.ids),
        "cocitation_overlap": groundtruth.build_cocitation_positives(
            dataset_path, corpus.ids, config.COCITE_MIN_SHARED
        ),
    }

    # Deterministic sample: papers with a citation positive (the headline GT), in corpus order.
    citation_pos = ground_truth["citation_overlap"]
    sample = [pid for pid in corpus.ids if citation_pos.get(pid)][:_SAMPLE]
    logger.info(
        "Reranker: %s | sample: %d sources | Stage-1: %s, pool=%d",
        config.CROSS_ENCODER_MODEL,
        len(sample),
        "fused" if _STAGE1.use_fusion else "dense",
        _STAGE1.pool_size,
    )

    baseline = Recommender(corpus, reranker=NoOpReranker())
    started = time.perf_counter()
    reranker = Recommender(corpus, signals=baseline.signals, reranker=CrossEncoderReranker(config.CROSS_ENCODER_MODEL))
    logger.info("Loaded cross-encoder in %.1fs", time.perf_counter() - started)

    max_k = max(config.K_VALUES)
    started = time.perf_counter()
    base_rank = baseline.rank_all(_STAGE1, max_k, sources=sample)
    reranked = reranker.rank_all(_STAGE1, max_k, sources=sample)
    logger.info("Ranked %d sources (Stage-1 + reranked) in %.1fs", len(sample), time.perf_counter() - started)

    logger.info("%-18s%-12s%12s%12s%12s", "ground_truth", "config", f"recall@{max_k // 2}", f"ndcg@{max_k // 2}", "mrr")
    logger.info("-" * 66)
    report: Dict[str, dict] = {}
    k = config.K_VALUES[1] if len(config.K_VALUES) > 1 else config.K_VALUES[0]
    for gt_name, positives in ground_truth.items():
        ceiling = _recall_at_pool(baseline, sample, positives)
        base_eval = evaluate_signal(base_rank, positives, config.K_VALUES)
        rr_eval = evaluate_signal(reranked, positives, config.K_VALUES)
        report[gt_name] = {"recall_at_pool": round(ceiling, 4), "stage1": base_eval, "reranked": rr_eval}
        for label, result in (("stage1", base_eval), ("reranked", rr_eval)):
            m = result["metrics"]
            logger.info("%-18s%-12s%12.4f%12.4f%12.4f", gt_name, label, m[f"recall@{k}"], m[f"ndcg@{k}"], m["mrr"])
        logger.info("%-18s%-12s%12.4f", gt_name, "pool-ceiling", ceiling)

    logger.info("(reranker only reorders the pool; recall@pool is its ceiling. Sample-based — not full corpus.)")


if __name__ == "__main__":
    main()
