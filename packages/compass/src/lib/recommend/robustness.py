"""
Track B — robustness of the *conclusions*, not the system.

Three verdicts are banked in the recommender's evaluation history: BM25 ≥ dense, RRF ≥ weighted fusion,
and the cross-encoder reranker hurts. Each was read off a single point estimate against a *proxy* ground
truth at a fixed k. This harness asks whether those verdicts survive contact with uncertainty:

  * B1 — paired bootstrap CIs on the metric *gap* between the two configs in each verdict. Resampling the
         source papers gives a distribution of the gap; if its 95% CI straddles zero the verdict is not
         statistically supported by the current ground truth. Bootstrapping is *paired* (the same
         resampled sources score both configs) so the CI reflects the correlation between them.
  * B2 — knob-sweep winner stability. Re-reads each verdict while sweeping the ground-truth thresholds
         (tag/coupling/cocitation shared-count floors), the cut-off k, and the RRF constant. If the winner
         flips as a knob moves, the verdict was an artifact of that knob.
  * B3 — the reranker verdict, bootstrapped in its own harness (rerank_eval's Stage-1 pool + cross-encoder),
         since it lives outside the Stage-1 ablation.

Circularity is honored throughout: a verdict is only ever read on a ground truth that neither of its two
configs mechanically satisfies (the `circular_vs` sets from `recommend.eval`). A pair whose only honest
column has near-zero coverage is reported as *unfalsifiable with current ground truth* — itself a finding.

Run: `uv run -m src.lib.recommend.robustness`  (or `moon run compass:robustness`).
Set `ROBUSTNESS_RERANKER=1` to include B3 (loads the cross-encoder). `ROBUSTNESS_BOOT` sets resamples.

`@author`: DAShaikh10
"""

import json
import os
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import chromadb
import numpy as np

from src.utils import logger, resolve_path

from ..eval import config as eval_config
from ..eval import groundtruth, metrics
from . import config
from .corpus import Corpus
from .eval import _CONFIGS
from .pipeline import Recommender, RetrievalConfig

GROUND_TRUTHS = ("tag_overlap", "citation_overlap", "cocitation_overlap", "coupling_overlap")

# The verdicts to stress-test: (label, config A, config B). "A wins" means A scores higher. Honest ground
# truths are derived from each config's `circular_vs` in `_CONFIGS`, so circularity is single-sourced.
VERDICTS: Tuple[Tuple[str, str, str], ...] = (
    ("BM25 >= dense", "lexical", "dense (tier0)"),
    ("RRF >= weighted (all signals)", "fused (all)", "weighted (all)"),
    ("RRF >= weighted (content only, -citation)", "fused -citation", "weighted -citation"),
)

BOOT = int(os.getenv("ROBUSTNESS_BOOT", "2000"))
SEED = int(os.getenv("ROBUSTNESS_SEED", "20260701"))
# A pair's only-honest column must cover at least this many sources to be considered falsifiable.
MIN_COVERAGE = int(os.getenv("ROBUSTNESS_MIN_COVERAGE", "30"))


def per_source_recall(rankings: Dict[str, List[str]], positives: Dict[str, Set[str]], k: int) -> Dict[str, float]:
    """
    recall@k for every source that has at least one positive — the unit the bootstrap resamples.
    """

    return {
        source: metrics.recall_at_k(ranked, positives[source], k)
        for source, ranked in rankings.items()
        if positives.get(source)
    }


def paired_bootstrap(
    scores_a: Dict[str, float], scores_b: Dict[str, float], rng: np.random.Generator, n_boot: int = BOOT
) -> Dict[str, float]:
    """
    Bootstrap the mean gap (A − B) over the sources scored by *both* configs.

    Resamples source indices with replacement, recomputing the paired mean difference each time, and
    returns the observed gap, its 95% percentile CI, and P(gap > 0) — the share of resamples where A leads.
    A CI that straddles 0 means the current ground truth does not distinguish the two configs.
    """

    common = sorted(set(scores_a) & set(scores_b))
    if not common:
        return {
            "observed": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_a_gt_b": float("nan"),
            "coverage": 0,
        }

    arr_a = np.array([scores_a[s] for s in common])
    arr_b = np.array([scores_b[s] for s in common])
    diff = arr_a - arr_b
    observed = float(diff.mean())

    n = len(common)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_gaps = diff[idx].mean(axis=1)
    return {
        "observed": round(observed, 4),
        "ci_low": round(float(np.percentile(boot_gaps, 2.5)), 4),
        "ci_high": round(float(np.percentile(boot_gaps, 97.5)), 4),
        "p_a_gt_b": round(float((boot_gaps > 0).mean()), 4),
        "coverage": n,
    }


def _honest_gts(config_a: str, config_b: str) -> List[str]:
    """
    Ground truths neither config mechanically satisfies (union of their `circular_vs`, removed).
    """

    circular = set(_CONFIGS[config_a]["circular_vs"]) | set(_CONFIGS[config_b]["circular_vs"])
    return [gt for gt in GROUND_TRUTHS if gt not in circular]


# pylint: disable=too-many-locals


def bootstrap_verdicts(
    recommender: Recommender, ground_truth: Dict[str, Dict[str, Set[str]]], k: int, max_k: int
) -> List[dict]:
    """For each verdict, bootstrap the gap on every honest ground truth; flag unfalsifiable pairs."""

    rng = np.random.default_rng(SEED)
    ranking_cache: Dict[str, Dict[str, List[str]]] = {}

    def rank(name: str) -> Dict[str, List[str]]:
        if name not in ranking_cache:
            ranking_cache[name] = recommender.rank_all(_CONFIGS[name]["rc"], max_k)
        return ranking_cache[name]

    results: List[dict] = []
    for label, name_a, name_b in VERDICTS:
        honest = _honest_gts(name_a, name_b)
        rank_a, rank_b = rank(name_a), rank(name_b)
        columns: Dict[str, dict] = {}
        for gt in honest:
            boot = paired_bootstrap(
                per_source_recall(rank_a, ground_truth[gt], k),
                per_source_recall(rank_b, ground_truth[gt], k),
                rng,
            )
            boot["falsifiable"] = boot["coverage"] >= MIN_COVERAGE
            # Verdict holds when the CI excludes 0 in A's favour; note "ns" (not significant) otherwise.
            boot["supports_verdict"] = boot["falsifiable"] and boot["ci_low"] > 0
            columns[gt] = boot
        results.append(
            {
                "verdict": label,
                "config_a": name_a,
                "config_b": name_b,
                "honest_gts": honest,
                "columns": columns,
            }
        )
    return results


# pylint: enable=too-many-locals
# pylint: disable=too-many-arguments,too-many-positional-arguments


def _build_gt(dataset_path: str, corpus_ids: Sequence[str], gold_tags, tag_min: int, couple_min: int, cocite_min: int):
    """
    Rebuild the four proxy positive sets at the given thresholds (rankings stay fixed — this is cheap).
    """

    return {
        "tag_overlap": groundtruth.build_tag_positives(gold_tags, tag_min, config.TAG_JACCARD_MIN),
        "citation_overlap": groundtruth.build_citation_positives(dataset_path, corpus_ids),
        "cocitation_overlap": groundtruth.build_cocitation_positives(dataset_path, corpus_ids, cocite_min),
        "coupling_overlap": groundtruth.build_coupling_positives(dataset_path, corpus_ids, couple_min),
    }


def _winner(rank_a, rank_b, name_a, name_b, positives, k) -> dict:
    """
    Which config leads on this positive set, and by how much (macro recall@k gap).
    """

    sa = per_source_recall(rank_a, positives, k)
    sb = per_source_recall(rank_b, positives, k)
    common = sorted(set(sa) & set(sb))
    if not common:
        return {"winner": "n/a", "gap": float("nan"), "coverage": 0}
    gap = float(np.mean([sa[s] - sb[s] for s in common]))
    return {"winner": name_a if gap > 0 else name_b, "gap": round(abs(gap), 4), "coverage": len(common)}


# pylint: disable=too-many-locals


def sweep_thresholds(
    recommender: Recommender, dataset_path: str, gold_tags, k_values: List[int], max_k: int
) -> Dict[str, List[dict]]:
    """
    Re-read each verdict across ground-truth threshold and k sweeps. Rankings are computed once per config
    and re-scored against every rebuilt ground truth, so the whole sweep costs one ranking pass per config.
    """

    ranking_cache: Dict[str, Dict[str, List[str]]] = {}

    def rank(name: str) -> Dict[str, List[str]]:
        if name not in ranking_cache:
            ranking_cache[name] = recommender.rank_all(_CONFIGS[name]["rc"], max_k)
        return ranking_cache[name]

    # (knob label, gt each setting rebuilds, list of (setting-label, kwargs override)).
    tag_settings = [(f"tag_min={m}", {"tag_min": m}) for m in (1, 2, 3)]
    couple_settings = [(f"couple_min={m}", {"couple_min": m}) for m in (2, 3, 5)]
    cocite_settings = [(f"cocite_min={m}", {"cocite_min": m}) for m in (1, 2)]
    defaults = {
        "tag_min": config.TAG_MIN_SHARED,
        "couple_min": config.COUPLE_MIN_SHARED,
        "cocite_min": config.COCITE_MIN_SHARED,
    }

    out: Dict[str, List[dict]] = {}
    for label, name_a, name_b in VERDICTS:
        honest = _honest_gts(name_a, name_b)
        rank_a, rank_b = rank(name_a), rank(name_b)
        rows: List[dict] = []

        # Threshold sweeps — only the ground truth a knob controls, and only if it is an honest column here.
        knob_map = [
            ("tag_overlap", tag_settings),
            ("coupling_overlap", couple_settings),
            ("cocitation_overlap", cocite_settings),
        ]
        for gt_name, settings in knob_map:
            if gt_name not in honest:
                continue
            for setting_label, override in settings:
                positives = _build_gt(dataset_path, recommender.corpus.ids, gold_tags, **{**defaults, **override})[
                    gt_name
                ]
                res = _winner(
                    rank_a, rank_b, name_a, name_b, positives, k_values[1] if len(k_values) > 1 else k_values[0]
                )
                rows.append({"knob": setting_label, "gt": gt_name, **res})

        # k sweep — free, on every honest column at default thresholds.
        base_gt = _build_gt(dataset_path, recommender.corpus.ids, gold_tags, **defaults)
        for k in k_values:
            for gt_name in honest:
                res = _winner(rank_a, rank_b, name_a, name_b, base_gt[gt_name], k)
                rows.append({"knob": f"k={k}", "gt": gt_name, **res})

        out[label] = rows
    return out


# pylint: enable=too-many-locals,too-many-arguments,too-many-positional-arguments


def sweep_rrf_k(recommender: Recommender, ground_truth, k: int, max_k: int) -> List[dict]:
    """
    RRF-vs-weighted on its honest cocitation column across the RRF rank constant (re-ranks per value).
    """

    name_a, name_b = "fused (all)", "weighted (all)"
    honest = _honest_gts(name_a, name_b)
    gt_name = honest[0] if honest else "cocitation_overlap"
    positives = ground_truth[gt_name]
    rank_b = recommender.rank_all(_CONFIGS[name_b]["rc"], max_k)

    rows: List[dict] = []
    for rrf_k in (10, 30, 60, 100):
        rank_a = recommender.rank_all(RetrievalConfig(signals=_CONFIGS[name_a]["rc"].signals, rrf_k=rrf_k), max_k)
        res = _winner(rank_a, rank_b, name_a, name_b, positives, k)
        rows.append({"knob": f"rrf_k={rrf_k}", "gt": gt_name, **res})
    return rows


# pylint: disable=too-many-locals


def bootstrap_reranker(recommender: Recommender, ground_truth, k: int, max_k: int) -> Optional[dict]:
    """
    Bootstrap the reranked-vs-Stage-1 recall@k gap on citation_overlap (rerank_eval's headline, non-circular
    ground truth). A CI that sits at or below 0 confirms the 'reranker does not help / hurts' verdict.
    Loads the cross-encoder; sample-based, mirroring rerank_eval.
    """

    from .rerank import CrossEncoderReranker, NoOpReranker  # pylint: disable=import-outside-toplevel

    pool = int(os.getenv("REC_RERANK_POOL", "50"))
    sample_size = int(os.getenv("REC_RERANK_SAMPLE", "120"))
    stage1 = RetrievalConfig(pool_size=pool, use_fusion=True)
    positives = ground_truth["citation_overlap"]
    sample = [pid for pid in recommender.corpus.ids if positives.get(pid)][:sample_size]

    started = time.perf_counter()
    reranker = Recommender(
        recommender.corpus, signals=recommender.signals, reranker=CrossEncoderReranker(config.CROSS_ENCODER_MODEL)
    )
    logger.info("Loaded cross-encoder in %.1fs; reranking %d sources", time.perf_counter() - started, len(sample))

    base = Recommender(recommender.corpus, signals=recommender.signals, reranker=NoOpReranker())
    base_rank = base.rank_all(stage1, max_k, sources=sample)
    rr_rank = reranker.rank_all(stage1, max_k, sources=sample)

    rng = np.random.default_rng(SEED)
    boot = paired_bootstrap(per_source_recall(rr_rank, positives, k), per_source_recall(base_rank, positives, k), rng)
    boot["falsifiable"] = boot["coverage"] >= MIN_COVERAGE
    # Verdict = "reranker helps"; it is SUPPORTED only if the reranker's gap CI is > 0. Otherwise the
    # banked "reranker hurts / no help" stands (and ci_high <= 0 is positive evidence of harm).
    boot["reranker_helps"] = boot["falsifiable"] and boot["ci_low"] > 0
    boot["gt"] = "citation_overlap"
    return boot


# pylint: enable=too-many-locals


def _print_report(report: dict) -> None:
    k = report["k"]
    logger.info("=== B1: verdict bootstrap (%d resamples, recall@%d gap A−B) ===", report["n_boot"], k)
    for v in report["verdicts"]:
        logger.info("• %s   [A=%s, B=%s]", v["verdict"], v["config_a"], v["config_b"])
        for gt, b in v["columns"].items():
            verdict = (
                "SUPPORTED"
                if b["supports_verdict"]
                else ("unfalsifiable" if not b["falsifiable"] else "not significant")
            )
            logger.info(
                "    %-20s gap=%+.4f  CI[%+.4f,%+.4f]  P(A>B)=%.3f  n=%d  -> %s",
                gt,
                b["observed"],
                b["ci_low"],
                b["ci_high"],
                b["p_a_gt_b"],
                b["coverage"],
                verdict,
            )

    logger.info(
        "=== B2: winner stability across knob sweeps (per honest column; * = n<%d, ignored) ===", report["min_coverage"]
    )

    def _report_sweep(header: str, rows: List[dict]) -> None:
        # Stability is judged *within each ground-truth column* (a knob flipping the winner on the same
        # proxy is the instability signal; different proxies favouring different configs is not). Columns
        # with fewer than min_coverage sources are too sparse to judge and are excluded from the verdict.
        by_gt: Dict[str, List[dict]] = {}
        for r in rows:
            by_gt.setdefault(r["gt"], []).append(r)
        flips = [
            gt
            for gt, grp in by_gt.items()
            if len({r["winner"] for r in grp if r["coverage"] >= report["min_coverage"]}) > 1
        ]
        verdict = f"FLIPS on {', '.join(flips)}" if flips else "STABLE (every honest column keeps one winner)"
        logger.info("• %s  -> %s", header, verdict)
        for r in rows:
            mark = "*" if r["coverage"] < report["min_coverage"] else " "
            logger.info(
                "    %-14s %-20s winner=%-18s gap=%.4f n=%d%s",
                r["knob"],
                r["gt"],
                r["winner"],
                r["gap"],
                r["coverage"],
                mark,
            )

    for label, rows in report["sweeps"].items():
        _report_sweep(label, rows)
    if report.get("rrf_k_sweep"):
        _report_sweep("RRF-vs-weighted across rrf_k", report["rrf_k_sweep"])

    if report.get("reranker"):
        b = report["reranker"]
        logger.info("=== B3: reranker verdict (reranked − Stage-1, recall@%d on %s) ===", k, b["gt"])
        verdict = "reranker HELPS" if b["reranker_helps"] else "reranker does NOT help (banked verdict stands)"
        logger.info(
            "    gap=%+.4f  CI[%+.4f,%+.4f]  P(help>0)=%.3f  n=%d  -> %s",
            b["observed"],
            b["ci_low"],
            b["ci_high"],
            b["p_a_gt_b"],
            b["coverage"],
            verdict,
        )


def main() -> None:  # pylint: disable=too-many-locals
    """
    Run B1 + B2 (+ optional B3), write the report, and print a verdict-by-verdict summary.
    """

    current_dir = os.path.dirname(__file__)
    client = chromadb.PersistentClient(path=resolve_path(current_dir, config.EMBEDDING_DATABASE_NAME))
    collection = client.get_collection(name=config.EMBEDDING_COLLECTION_NAME)
    dataset_path = resolve_path(current_dir, config.ENRICHED_DATASET_FILE)
    canonical_map_path = resolve_path(current_dir, config.CANONICAL_MAP_FILE)

    corpus = Corpus.load(
        collection,
        dataset_path,
        resolve_path(current_dir, config.REC_ANNOTATION_FILE),
        canonical_map_path,
        tag_source=config.REC_TAG_SOURCE,
    )
    recommender = Recommender(corpus)
    logger.info("Corpus: %d papers, %d tagged", len(corpus.ids), sum(1 for t in corpus.tag_sets if t))

    gold_tags = groundtruth.tag_sets_from_annotations(
        resolve_path(current_dir, eval_config.ANNOTATION_FILE), canonical_map_path, corpus.ids
    )
    ground_truth = _build_gt(
        dataset_path,
        corpus.ids,
        gold_tags,
        config.TAG_MIN_SHARED,
        config.COUPLE_MIN_SHARED,
        config.COCITE_MIN_SHARED,
    )
    logger.info(
        "Coverage — tag:%d citation:%d cocitation:%d coupling:%d",
        *[sum(1 for s in ground_truth[gt] if ground_truth[gt][s]) for gt in GROUND_TRUTHS],
    )

    k = config.K_VALUES[1] if len(config.K_VALUES) > 1 else config.K_VALUES[0]
    max_k = max(config.K_VALUES)

    started = time.perf_counter()
    verdicts = bootstrap_verdicts(recommender, ground_truth, k, max_k)
    logger.info("B1 verdict bootstrap in %.1fs", time.perf_counter() - started)

    started = time.perf_counter()
    sweeps = sweep_thresholds(recommender, dataset_path, gold_tags, config.K_VALUES, max_k)
    rrf_k_sweep = sweep_rrf_k(recommender, ground_truth, k, max_k)
    logger.info("B2 knob sweeps in %.1fs", time.perf_counter() - started)

    reranker = None
    if os.getenv("ROBUSTNESS_RERANKER", "").lower() in ("1", "true", "yes"):
        started = time.perf_counter()
        reranker = bootstrap_reranker(recommender, ground_truth, k, max_k)
        logger.info("B3 reranker bootstrap in %.1fs", time.perf_counter() - started)

    report = {
        "analysis": "track B — conclusion robustness (bootstrap CIs + knob-sweep stability)",
        "corpus_size": len(corpus.ids),
        "k": k,
        "n_boot": BOOT,
        "seed": SEED,
        "min_coverage": MIN_COVERAGE,
        "verdicts": verdicts,
        "sweeps": sweeps,
        "rrf_k_sweep": rrf_k_sweep,
        "reranker": reranker,
    }
    report_path = resolve_path(current_dir, os.getenv("ROBUSTNESS_REPORT_FILE", "robustness_report.json"))
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logger.info("Wrote report to %s", report_path)
    _print_report(report)


if __name__ == "__main__":
    main()
