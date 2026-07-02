"""
Tier 1 ablation evaluation.

Scores each Stage 1 signal in isolation, plus the RRF fusion of all of them, against both proxy
ground-truth signals — so the fusion lift over the Tier 0 dense baseline is measurable.

Eval-circularity rule (enforced in the report, not just documented): a signal must be judged on the
ground truth it does *not* mechanically satisfy. The entity signal ranks by canonical-tag overlap and
the tag-overlap ground truth *defines* positives by canonical-tag overlap, so that pairing is circular
and flagged; the entity signal's honest score is its number on citation-overlap. Likewise the citation
signal is circular against citation-overlap and honest against tag-overlap. Each config's `circular_vs`
field names the ground truth(s) to disregard for it.

`@author`: DAShaikh10
"""

import json
import os
import time
from typing import Dict

import chromadb

from src.utils import logger, resolve_path

from ..eval import config as eval_config
from ..eval import groundtruth
from ..eval.main import evaluate_signal
from . import config
from .corpus import Corpus
from .pipeline import DEFAULT_SIGNALS, Recommender, RetrievalConfig

# Stage 1 configurations to ablate, and the ground truth each is circular against (mechanically inflated,
# so disregard that column for that row).
# `circular_vs` lists the ground truths a config mechanically satisfies (disregard those columns). The
# citation signal ranks by bibliographic coupling + direct citation, so it is circular against BOTH
# citation_overlap (direct) and coupling_overlap (shared refs); the entity signal is circular vs tag.
_CONFIGS: Dict[str, dict] = {
    "dense (tier0)": {"rc": RetrievalConfig(use_fusion=False), "circular_vs": []},
    "lexical": {"rc": RetrievalConfig(signals=("lexical",)), "circular_vs": []},
    "citation": {
        "rc": RetrievalConfig(signals=("citation",)),
        "circular_vs": ["citation_overlap", "coupling_overlap"],
    },
    "entity": {"rc": RetrievalConfig(signals=("entity",)), "circular_vs": ["tag_overlap"]},
    "fused (all)": {
        "rc": RetrievalConfig(signals=DEFAULT_SIGNALS),
        "circular_vs": ["tag_overlap", "citation_overlap", "coupling_overlap"],
    },
    # Held-out fusions: drop the signal that mechanically satisfies a ground truth so fusion's lift on
    # *that* ground truth is honestly measurable. Read each only in its non-circular column.
    "fused -entity": {
        "rc": RetrievalConfig(signals=("dense", "lexical", "citation")),
        "circular_vs": ["citation_overlap", "coupling_overlap"],
    },
    "fused -citation": {
        "rc": RetrievalConfig(signals=("dense", "lexical", "entity")),
        "circular_vs": ["tag_overlap"],
    },
    # Weighted-sum fusion (coupling-dominant). RRF discards score magnitude and weights every signal
    # equally; weighted sum keeps magnitude and lets the citation signal lead. Same circularity applies —
    # any config carrying the citation signal is circular vs citation/coupling, so read it on tag and
    # co-citation; any carrying entity is circular vs tag. The two held-out variants mirror the RRF rows
    # above so the weighted-vs-RRF delta is readable in a single non-circular column.
    "weighted (all)": {
        "rc": RetrievalConfig(signals=DEFAULT_SIGNALS, fusion_method="weighted"),
        "circular_vs": ["tag_overlap", "citation_overlap", "coupling_overlap"],
    },
    "weighted -entity": {
        "rc": RetrievalConfig(signals=("dense", "lexical", "citation"), fusion_method="weighted"),
        "circular_vs": ["citation_overlap", "coupling_overlap"],
    },
    "weighted -citation": {
        "rc": RetrievalConfig(signals=("dense", "lexical", "entity"), fusion_method="weighted"),
        "circular_vs": ["tag_overlap"],
    },
}


def _print_summary(report: dict) -> None:
    """
    Console table: one row per config, recall@10 / ndcg@10 / mrr per ground truth, '*' = circular.
    """

    k = report["k_values"][1] if len(report["k_values"]) > 1 else report["k_values"][0]
    signals = list(next(iter(report["configs"].values()))["signals"].keys())
    header = f"{'config':<16}" + "".join(
        f"{s + ' R@' + str(k):>22}{' nDCG@' + str(k):>12}{' MRR':>10}" for s in signals
    )
    logger.info(header)
    logger.info("-" * len(header))
    for name, result in report["configs"].items():
        row = f"{name:<16}"
        for signal in signals:
            metrics = result["signals"][signal]["metrics"]
            flag = "*" if signal in result["circular_vs"] else " "
            row += f"{metrics[f'recall@{k}']:>21.4f}{flag}{metrics[f'ndcg@{k}']:>12.4f}{metrics['mrr']:>10.4f}"
        logger.info(row)
    logger.info("(* = circular: signal mechanically satisfies this ground truth; disregard that column)")


# pylint: disable=too-many-locals


def main() -> None:
    """
    Run the recommender ablation end-to-end and write the report under `data/`.
    """

    current_dir = os.path.dirname(__file__)
    database_path = resolve_path(current_dir, config.EMBEDDING_DATABASE_NAME)
    dataset_path = resolve_path(current_dir, config.ENRICHED_DATASET_FILE)
    serving_annotation_path = resolve_path(current_dir, config.REC_ANNOTATION_FILE)
    gold_annotation_path = resolve_path(current_dir, eval_config.ANNOTATION_FILE)
    canonical_map_path = resolve_path(current_dir, config.CANONICAL_MAP_FILE)

    logger.info("Loading Chroma collection '%s' from %s", config.EMBEDDING_COLLECTION_NAME, database_path)
    client = chromadb.PersistentClient(path=database_path)
    collection = client.get_collection(name=config.EMBEDDING_COLLECTION_NAME)

    logger.info("Building corpus (entity-signal tag source: '%s')", config.REC_TAG_SOURCE)
    corpus = Corpus.load(
        collection, dataset_path, serving_annotation_path, canonical_map_path, tag_source=config.REC_TAG_SOURCE
    )
    tagged = sum(1 for tags in corpus.tag_sets if tags)
    logger.info("Corpus: %d papers (%d carry entity tags)", len(corpus.ids), tagged)

    # Ground truth. Tag-overlap uses the held-out human gold, kept separate from the serving tags above.
    gold_tags = groundtruth.tag_sets_from_annotations(gold_annotation_path, canonical_map_path, corpus.ids)
    tag_positives = groundtruth.build_tag_positives(gold_tags, config.TAG_MIN_SHARED, config.TAG_JACCARD_MIN)
    citation_positives = groundtruth.build_citation_positives(dataset_path, corpus.ids)
    cocitation_positives = groundtruth.build_cocitation_positives(dataset_path, corpus.ids, config.COCITE_MIN_SHARED)
    coupling_positives = groundtruth.build_coupling_positives(dataset_path, corpus.ids, config.COUPLE_MIN_SHARED)
    ground_truth = {
        "tag_overlap": tag_positives,
        "citation_overlap": citation_positives,
        "cocitation_overlap": cocitation_positives,
        "coupling_overlap": coupling_positives,
    }
    logger.info(
        "Positive coverage — tag: %d, citation: %d, co-citation: %d, coupling: %d papers",
        sum(1 for s in tag_positives if tag_positives[s]),
        sum(1 for s in citation_positives if citation_positives[s]),
        sum(1 for s in cocitation_positives if cocitation_positives[s]),
        sum(1 for s in coupling_positives if coupling_positives[s]),
    )

    recommender = Recommender(corpus)  # no-op reranker: Stage 1 isolation
    max_k = max(config.K_VALUES)

    configs_report: Dict[str, dict] = {}
    for name, spec in _CONFIGS.items():
        started = time.perf_counter()
        rankings = recommender.rank_all(spec["rc"], max_k)
        elapsed = time.perf_counter() - started
        configs_report[name] = {
            "circular_vs": spec["circular_vs"],
            "signals": {
                gt: evaluate_signal(rankings, positives, config.K_VALUES) for gt, positives in ground_truth.items()
            },
        }
        logger.info("Ranked config '%s' (%d sources, top-%d) in %.2fs", name, len(rankings), max_k, elapsed)

    report = {
        "recommender": "tier1 fusion (dense + bm25 + citation + entity), RRF + weighted-sum ablated",
        "corpus_size": len(corpus.ids),
        "serving_annotation": config.REC_ANNOTATION_FILE,
        "eval_annotation": eval_config.ANNOTATION_FILE,
        "k_values": config.K_VALUES,
        "pool_size": config.POOL_SIZE,
        "rrf_k": config.RRF_K,
        "fusion_weights": config.FUSION_WEIGHTS,
        "configs": configs_report,
    }

    report_path = resolve_path(current_dir, config.REC_REPORT_FILE)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logger.info("Wrote report to %s", report_path)

    _print_summary(report)


# pylint: enable=too-many-locals

if __name__ == "__main__":
    main()
