"""
Candidate-list fusion: Reciprocal Rank Fusion and weighted-sum.

Two ways to combine the Stage 1 signals into one ranking:

  * `rrf`      — rank-based; needs no per-signal calibration and is robust to signals on wildly different
                 scales (cosine vs BM25 vs Jaccard). The right default before any labelled data exists.
  * `weighted` — score-based; min-max normalizes each signal's scores then sums them under per-signal
                 weights. Unlike RRF it keeps score *magnitude* (the gap between rank 1 and rank 2, not
                 just their order) and lets one signal dominate. With a coupling-heavy weight vector this
                 expresses "shared references are observed evidence, so trust them over a predicted
                 cosine" — a candidate absent from the coupling list simply contributes 0 there and still
                 ranks on the dense/lexical signals, so dense carries the papers coupling can't see.

`@author`: DAShaikh10
"""

from collections import defaultdict
from typing import Dict, List, Mapping, Sequence, Tuple


def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> List[Tuple[int, float]]:
    """
    Fuse `rankings` (each a best-first list of candidate rows) into one ranked `[(row, score), ...]`.

    A candidate's score is the sum over the lists it appears in of `1 / (k + rank)`, rank starting at 1.
    Larger `k` flattens the contribution of top positions; 60 is the common default.
    """

    scores: defaultdict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, row in enumerate(ranking, start=1):
            scores[row] += 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def weighted(
    scored_rankings: Sequence[Tuple[str, Sequence[Tuple[int, float]]]],
    weights: Mapping[str, float],
) -> List[Tuple[int, float]]:
    """
    Fuse per-signal scored lists into one ranking by a weighted sum of min-max-normalized scores.

    `scored_rankings` is `[(signal_name, [(row, score), ...]), ...]`; `weights` maps a signal name to its
    weight (a missing or zero weight drops the signal). Each signal's raw scores are min-max scaled to
    [0, 1] over its own candidates before weighting, so cosine, BM25 and Jaccard become comparable; when a
    signal's scores are all equal they map to 1.0. A candidate not present in a signal's list contributes
    nothing for that signal — it is not penalized below zero — so a heavily-weighted coupling signal lifts
    coupled papers above equally-dense uncoupled ones without burying papers coupling never reached.
    """

    totals: Dict[int, float] = defaultdict(float)
    for name, ranking in scored_rankings:
        weight = weights.get(name, 0.0)
        if weight == 0.0 or not ranking:
            continue
        values = [score for _, score in ranking]
        low, high = min(values), max(values)
        span = high - low
        for row, score in ranking:
            normalized = (score - low) / span if span > 0.0 else 1.0
            totals[row] += weight * normalized

    return sorted(totals.items(), key=lambda pair: pair[1], reverse=True)
