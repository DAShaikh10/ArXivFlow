"""
Track A — recommender robustness under query perturbation.

Asks: if the query wobbles slightly, does the top-k wobble a lot? A recommender people trust should
return nearly the same papers for near-identical queries. This harness perturbs the query in each
signal's *native* representation — Gaussian noise on the dense vector, token dropout on the BM25 query,
reference-key dropout on the citation query, tag dropout on the entity query — re-ranks through the
identical scoring path (`Signals.*_from_*`), and measures two things against the unperturbed baseline:

  * stability   — how much the ranking *moved*: Jaccard@k (membership churn), a top-weighted rank
                  agreement, and Kendall's tau over the survivors (reordering).
  * retention   — how much the ranking got *worse*: Δrecall@k against the signal's honest ground truth.
                  A ranking can churn heavily while holding recall (it swapped equally-relevant papers),
                  so stability alone under-reports or over-reports quality loss; retention disentangles it.

The dense perturbation is parameterized by the *resulting* cosine between the original and perturbed
query (0.99 / 0.95 / 0.90) rather than a noise magnitude in σ-units, because "results at 0.95 query
similarity" is legible and "ε = 0.1σ" is not. Gaussian noise in 768-dim does not correspond to any real
text edit — this probes the kNN's conditioning, not query realism; for real typo/paraphrase edits on the
free-text door run with `PERTURB_TEXT=1` (see `embed.py`), which re-embeds through SPECTER2.

Everything is seeded (`PERTURB_SEED`) and sampled (`PERTURB_SAMPLE` sources × `PERTURB_TRIALS` trials).

Run: `uv run -m src.lib.recommend.perturb`  (or `moon run compass:perturb`).

`@author`: DAShaikh10
"""

import json
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import chromadb
import numpy as np
from scipy.stats import kendalltau

from src.utils import logger, resolve_path

from ..eval import config as eval_config
from ..eval import groundtruth, metrics
from . import config
from .corpus import Corpus
from .fusion import rrf
from .signals import Signals, _tokenize

# Sweep points. Dense is keyed by target cosine(q, q'); the set-dropout signals by member drop rate.
DENSE_COSINES: Tuple[float, ...] = (0.99, 0.95, 0.90)
DROP_RATES: Tuple[float, ...] = (0.1, 0.2, 0.3)

SAMPLE = int(os.getenv("PERTURB_SAMPLE", "150"))
TRIALS = int(os.getenv("PERTURB_TRIALS", "5"))
SEED = int(os.getenv("PERTURB_SEED", "20260701"))
_LETTERS = "abcdefghijklmnopqrstuvwxyz"

# The honest (non-circular) ground truth each signal's retention is scored against — mirrors the
# circularity rule in `recommend.eval`: never score a signal on the ground truth it mechanically satisfies.
HONEST_GT: Dict[str, str] = {
    "dense": "citation_overlap",
    "lexical": "citation_overlap",
    "citation": "cocitation_overlap",  # citation signal is circular vs citation/coupling
    "entity": "coupling_overlap",  # entity signal is circular vs tag
    "fused": "cocitation_overlap",  # only column honest for a fusion carrying every signal
}


def perturb_to_cosine(query: np.ndarray, target_cos: float, rng: np.random.Generator) -> np.ndarray:
    """
    A unit vector at exactly `target_cos` cosine from `query`: q' = c·q + √(1-c²)·u, where u is a random
    unit vector in the hyperplane orthogonal to q. Exact by construction (q'·q = c since u⊥q, both unit).
    """

    q = query / (np.linalg.norm(query) or 1.0)
    gaussian = rng.standard_normal(q.shape).astype(q.dtype)
    perp = gaussian - (gaussian @ q) * q  # project out the component along q
    perp /= np.linalg.norm(perp) or 1.0
    return target_cos * q + np.sqrt(max(0.0, 1.0 - target_cos**2)) * perp


def drop_members(items: Sequence, rate: float, rng: np.random.Generator) -> list:
    """
    Independently drop each member with probability `rate` (Bernoulli token/key/tag dropout).
    """

    if not items:
        return list(items)
    keep = rng.random(len(items)) >= rate
    return [item for item, k in zip(items, keep) if k]


def inject_typos(text: str, rate: float, rng: np.random.Generator) -> str:
    """
    Per-character keyboard-noise: each char is, with probability `rate`, deleted / substituted /
    transposed with its neighbour / preceded by a random letter. Models real free-text query sloppiness.
    """

    chars = list(text)
    out: List[str] = []
    i = 0
    while i < len(chars):
        char = chars[i]
        if char.isalpha() and rng.random() < rate:
            op = rng.integers(0, 4)
            if op == 0:  # delete
                i += 1
                continue
            if op == 1:  # substitute
                out.append(_LETTERS[rng.integers(0, 26)])
                i += 1
                continue
            if op == 2 and i + 1 < len(chars):  # transpose with next
                out.append(chars[i + 1])
                out.append(char)
                i += 2
                continue
            out.append(_LETTERS[rng.integers(0, 26)])  # insert before
        out.append(char)
        i += 1
    return "".join(out)


def drop_words(text: str, rate: float, rng: np.random.Generator) -> str:
    """
    Independently drop each whitespace token with probability `rate` (query truncation / omission).
    """

    return " ".join(drop_members(text.split(), rate, rng))


def _kendall_over_survivors(baseline: List[str], perturbed: List[str], k: int) -> Optional[float]:
    """
    Kendall's tau on the items present in *both* top-k, ranked by their position in each list. None if
    fewer than two survive (tau undefined) — averaged only over the trials where it is defined.
    """

    shared = [item for item in baseline[:k] if item in set(perturbed[:k])]
    if len(shared) < 2:
        return None
    pos_perturbed = {item: rank for rank, item in enumerate(perturbed[:k])}
    tau, _ = kendalltau(range(len(shared)), [pos_perturbed[item] for item in shared])
    return None if tau is None or np.isnan(tau) else float(tau)


def _score_pair(baseline: List[str], perturbed: List[str], relevant: Set[str], k: int) -> Dict[str, Optional[float]]:
    """
    Stability + retention numbers for one (baseline, perturbed) top-k pair.
    """

    return {
        "jaccard": metrics.jaccard_at_k(baseline, perturbed, k),
        "rank_agreement": metrics.rank_agreement_at_k(baseline, perturbed, k),
        "kendall": _kendall_over_survivors(baseline, perturbed, k),
        "delta_recall": metrics.recall_at_k(perturbed, relevant, k) - metrics.recall_at_k(baseline, relevant, k),
    }


def _aggregate(pairs: List[Dict[str, Optional[float]]]) -> Dict[str, float]:
    """
    Mean each metric across trials, ignoring None (undefined Kendall tau).
    """

    out: Dict[str, float] = {}
    for key in ("jaccard", "rank_agreement", "kendall", "delta_recall"):
        vals = [p[key] for p in pairs if p[key] is not None]
        out[key] = round(sum(vals) / len(vals), 4) if vals else float("nan")
    return out


# A perturber turns (signal-native baseline query for a row, level, rng) into a perturbed ranking.
Ranker = Callable[[int, float, np.random.Generator], List[int]]


def _rankers(signals: Signals) -> Dict[str, Tuple[Ranker, Tuple[float, ...], Callable[[int], bool]]]:
    """
    Per signal: a perturbed-ranking function, its sweep levels, and a predicate for eligible source rows.
    """

    corpus = signals.corpus
    k_pool = max(config.K_VALUES)

    def dense(row: int, level: float, rng: np.random.Generator) -> List[int]:
        q = perturb_to_cosine(corpus.embeddings[row], level, rng)
        return [c for c, _ in signals.dense_from_vector(q, limit=k_pool, exclude=row)]

    def lexical(row: int, level: float, rng: np.random.Generator) -> List[int]:
        tokens = drop_members(_tokenize(corpus.documents[row]), level, rng)
        return [c for c, _ in signals.lexical_from_tokens(tokens, limit=k_pool, exclude=row)]

    def citation(row: int, level: float, rng: np.random.Generator) -> List[int]:
        keys = set(drop_members(sorted(corpus.ref_keys[row]), level, rng))
        return [c for c, _ in signals.citation_from(keys, corpus.cites[row], limit=k_pool, exclude=row)]

    def entity(row: int, level: float, rng: np.random.Generator) -> List[int]:
        tags = set(drop_members(sorted(corpus.tag_sets[row]), level, rng))
        return [c for c, _ in signals.entity_from(tags, limit=k_pool, exclude=row)]

    return {
        "dense": (dense, DENSE_COSINES, lambda r: True),
        "lexical": (lexical, DROP_RATES, lambda r: bool(corpus.documents[r])),
        "citation": (citation, DROP_RATES, lambda r: len(corpus.ref_keys[r]) >= 2),
        "entity": (entity, DROP_RATES, lambda r: len(corpus.tag_sets[r]) >= 2),
    }


def _baseline_ranking(signals: Signals, name: str, row: int, k_pool: int) -> List[int]:
    """
    Unperturbed top-k rows for the signal (the row-form query).
    """

    return [c for c, _ in getattr(signals, name)(row, limit=k_pool)]


def run_signal(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    signals: Signals,
    name: str,
    ranker: Ranker,
    levels: Tuple[float, ...],
    eligible: Callable[[int], bool],
    positives: Dict[str, Set[str]],
    sample_rows: List[int],
    k: int,
) -> Dict[str, dict]:
    """
    Sweep one signal over its perturbation levels; aggregate stability + retention across sourcesxtrials.
    """

    corpus = signals.corpus
    ids = corpus.ids
    k_pool = max(config.K_VALUES)
    rows = [r for r in sample_rows if eligible(r)]

    baselines = {r: [ids[c] for c in _baseline_ranking(signals, name, r, k_pool)] for r in rows}

    per_level: Dict[str, dict] = {}
    for level in levels:
        pairs: List[Dict[str, Optional[float]]] = []
        for r in rows:
            relevant = positives.get(ids[r]) or set()
            # Seed per (row, level) so trials are reproducible and independent across levels.
            rng = np.random.default_rng((SEED, r, int(level * 1000)))
            for _ in range(TRIALS):
                perturbed = [ids[c] for c in ranker(r, level, rng)]
                pairs.append(_score_pair(baselines[r], perturbed, relevant, k))
        per_level[f"{level}"] = {"n_pairs": len(pairs), **_aggregate(pairs)}
    return {"honest_gt": HONEST_GT[name], "n_sources": len(rows), "levels": per_level}


# pylint: disable=too-many-arguments,too-many-positional-arguments


def _fused_rows(
    signals: Signals, row: int, k_pool: int, rng: Optional[np.random.Generator], cos: float, drop: float
) -> List[int]:
    """
    RRF fusion of the four signals for one source. `rng=None` → unperturbed baseline; otherwise the
    dense query is pushed to `cos` cosine and each set signal is thinned at `drop` rate off the same rng.
    """

    corpus = signals.corpus
    if rng is None:
        lists = [
            signals.dense(row, k_pool),
            signals.lexical(row, k_pool),
            signals.citation(row, k_pool),
            signals.entity(row, k_pool),
        ]
    else:
        q = perturb_to_cosine(corpus.embeddings[row], cos, rng)
        tokens = drop_members(_tokenize(corpus.documents[row]), drop, rng)
        keys = set(drop_members(sorted(corpus.ref_keys[row]), drop, rng))
        tags = set(drop_members(sorted(corpus.tag_sets[row]), drop, rng))
        lists = [
            signals.dense_from_vector(q, k_pool, exclude=row),
            signals.lexical_from_tokens(tokens, k_pool, exclude=row),
            signals.citation_from(keys, corpus.cites[row], k_pool, exclude=row),
            signals.entity_from(tags, k_pool, exclude=row),
        ]
    fused = rrf([[c for c, _ in ranking] for ranking in lists], k=config.RRF_K)
    return [c for c, _ in fused[:k_pool]]


# pylint: enable=too-many-arguments,too-many-positional-arguments


def run_fused(signals: Signals, positives: Dict[str, Set[str]], sample_rows: List[int], k: int) -> Dict[str, dict]:
    """
    Stability of the *shipped* RRF-fused ranking under a matched mild perturbation of every query input
    at once (dense at each swept cosine, the set signals at a fixed light dropout). This is the number
    that speaks to what a user would actually see move.
    """

    ids = signals.corpus.ids
    k_pool = max(config.K_VALUES)
    weak_drop = DROP_RATES[0]
    baselines = {r: [ids[c] for c in _fused_rows(signals, r, k_pool, None, 0.0, 0.0)] for r in sample_rows}

    per_level: Dict[str, dict] = {}
    for cos in DENSE_COSINES:
        pairs: List[Dict[str, Optional[float]]] = []
        for r in sample_rows:
            relevant = positives.get(ids[r]) or set()
            rng = np.random.default_rng((SEED, r, int(cos * 1000), 999))
            for _ in range(TRIALS):
                perturbed = [ids[c] for c in _fused_rows(signals, r, k_pool, rng, cos, weak_drop)]
                pairs.append(_score_pair(baselines[r], perturbed, relevant, k))
        per_level[f"{cos}"] = {"n_pairs": len(pairs), **_aggregate(pairs)}
    return {"honest_gt": HONEST_GT["fused"], "n_sources": len(sample_rows), "levels": per_level}


# (perturbation name, fn, sweep levels). Char-typo rate and word-drop rate are different scales by design.
TEXT_PERTURBATIONS = (
    ("typo", inject_typos, (0.02, 0.05, 0.10)),
    ("word_drop", drop_words, (0.1, 0.2, 0.3)),
)

# pylint: disable=too-many-locals


def run_text(signals: Signals, positives: Dict[str, Set[str]], sample_rows: List[int], k: int) -> Dict[str, dict]:
    """
    Real free-text query perturbation on the SPECTER2 adhoc-query door.

    For each sampled paper the query is its "Title[SEP]Abstract" text; the baseline ranking re-embeds the
    *clean* text through the adhoc-query adapter (not the stored proximity vector), so the measured drift
    is the perturbation's effect alone, not an adapter/version mismatch. Each perturbed variant is embedded
    the same way and ranked against the corpus. Requires the `perturb-text` dependency group.
    """

    from .embed import get_embedder  # pylint: disable=import-outside-toplevel  # lazy: heavy model dep

    corpus = signals.corpus
    ids = corpus.ids
    k_pool = max(config.K_VALUES)
    embedder = get_embedder("adhoc_query")

    clean_texts = [corpus.documents[r] for r in sample_rows]
    clean_vectors = embedder.encode(clean_texts)
    baselines = {
        r: [ids[c] for c, _ in signals.dense_from_vector(clean_vectors[i], limit=k_pool, exclude=r)]
        for i, r in enumerate(sample_rows)
    }

    results: Dict[str, dict] = {}
    for pert_name, pert_fn, levels in TEXT_PERTURBATIONS:
        per_level: Dict[str, dict] = {}
        for level in levels:
            # Build every (source × trial) perturbed text, embed in one batched pass, then rank.
            variants: List[str] = []
            index: List[int] = []  # sample_rows position for each variant
            for i, r in enumerate(sample_rows):
                rng = np.random.default_rng((SEED, r, int(level * 1000), hash(pert_name) & 0xFFFF))
                for _ in range(TRIALS):
                    variants.append(pert_fn(clean_texts[i], level, rng))
                    index.append(i)
            vectors = embedder.encode(variants)
            pairs: List[Dict[str, Optional[float]]] = []
            for pos, i in enumerate(index):
                r = sample_rows[i]
                relevant = positives.get(ids[r]) or set()
                perturbed = [ids[c] for c, _ in signals.dense_from_vector(vectors[pos], limit=k_pool, exclude=r)]
                pairs.append(_score_pair(baselines[r], perturbed, relevant, k))
            per_level[f"{level}"] = {"n_pairs": len(pairs), **_aggregate(pairs)}
        results[pert_name] = {"honest_gt": HONEST_GT["dense"], "n_sources": len(sample_rows), "levels": per_level}
    return results


# pylint: enable=too-many-locals


def _print_summary(report: dict) -> None:
    k = report["k"]
    logger.info("Perturbation robustness — %d sources/signal × %d trials, k=%d", report["sample"], report["trials"], k)
    logger.info(
        "%-15s%-8s%8s%10s%12s%10s%14s",
        "signal",
        "level",
        "n",
        "jaccard",
        "rank_agree",
        "kendall",
        f"Δrecall@{k}",
    )
    logger.info("-" * 77)
    for name, result in report["signals"].items():
        for level, m in result["levels"].items():
            logger.info(
                "%-15s%-8s%8d%10.3f%12.3f%10s%14.4f",
                name,
                level,
                m["n_pairs"],
                m["jaccard"],
                m["rank_agreement"],
                "nan" if m["kendall"] != m["kendall"] else f"{m['kendall']:.3f}",
                m["delta_recall"],
            )
    logger.info("(dense level = target cos(q,q'); others = member drop rate. Δrecall vs each signal's honest GT.)")


# pylint: disable=too-many-locals


def main() -> None:
    """
    Run the query-perturbation sweep over every signal + the fused pipeline, and write a report.
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
    signals = Signals(corpus)
    logger.info("Corpus: %d papers, %d tagged", len(corpus.ids), sum(1 for t in corpus.tag_sets if t))

    gold_tags = groundtruth.tag_sets_from_annotations(
        resolve_path(current_dir, eval_config.ANNOTATION_FILE), canonical_map_path, corpus.ids
    )
    ground_truth = {
        "tag_overlap": groundtruth.build_tag_positives(gold_tags, config.TAG_MIN_SHARED, config.TAG_JACCARD_MIN),
        "citation_overlap": groundtruth.build_citation_positives(dataset_path, corpus.ids),
        "cocitation_overlap": groundtruth.build_cocitation_positives(
            dataset_path, corpus.ids, config.COCITE_MIN_SHARED
        ),
        "coupling_overlap": groundtruth.build_coupling_positives(dataset_path, corpus.ids, config.COUPLE_MIN_SHARED),
    }

    k = config.K_VALUES[1] if len(config.K_VALUES) > 1 else config.K_VALUES[0]
    rng = np.random.default_rng(SEED)
    sample_rows = sorted(rng.choice(len(corpus.ids), size=min(SAMPLE, len(corpus.ids)), replace=False).tolist())

    rankers = _rankers(signals)
    results: Dict[str, dict] = {}
    for name, (ranker, levels, eligible) in rankers.items():
        started = time.perf_counter()
        results[name] = run_signal(
            signals, name, ranker, levels, eligible, ground_truth[HONEST_GT[name]], sample_rows, k
        )
        logger.info("Swept '%s' (%d levels) in %.1fs", name, len(levels), time.perf_counter() - started)

    started = time.perf_counter()
    results["fused"] = run_fused(signals, ground_truth[HONEST_GT["fused"]], sample_rows, k)
    logger.info("Swept fused pipeline in %.1fs", time.perf_counter() - started)

    # Real free-text perturbation (SPECTER2 adhoc-query door) — opt-in, needs the perturb-text deps and a
    # model load, so it runs on a (usually smaller) sample only when explicitly enabled.
    if os.getenv("PERTURB_TEXT", "").lower() in ("1", "true", "yes"):
        text_sample = sorted(
            rng.choice(
                len(corpus.ids), size=min(int(os.getenv("PERTURB_TEXT_SAMPLE", "60")), len(corpus.ids)), replace=False
            ).tolist()
        )
        started = time.perf_counter()
        for name, result in run_text(signals, ground_truth[HONEST_GT["dense"]], text_sample, k).items():
            results[f"text:{name}"] = result
        logger.info(
            "Swept real text perturbation (%d sources) in %.1fs", len(text_sample), time.perf_counter() - started
        )

    report = {
        "analysis": "track A — query-perturbation robustness (model-free, signal-native)",
        "corpus_size": len(corpus.ids),
        "sample": SAMPLE,
        "trials": TRIALS,
        "seed": SEED,
        "k": k,
        "signals": results,
    }
    report_path = resolve_path(current_dir, os.getenv("PERTURB_REPORT_FILE", "perturbation_report.json"))
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logger.info("Wrote report to %s", report_path)
    _print_summary(report)


# pylint: enable=too-many-locals

if __name__ == "__main__":
    main()
