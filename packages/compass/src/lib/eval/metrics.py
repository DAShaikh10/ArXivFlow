"""
Pure ranking metrics for top-k recommendation evaluation.

Every function takes a ranked list of candidate ids (best first, source excluded) and the set of
relevant ids for that source, and returns a float. No I/O, no global state — unit-testable in
isolation and reusable by every later recommender tier.

Relevance is binary (a candidate is relevant or it is not), which is what proxy ground-truth signals
like tag-overlap and citation-overlap give us.

`@author`: DAShaikh10
"""

import math
from typing import Sequence, Set


def precision_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    """
    Fraction of the top-k that are relevant.
    """

    if k <= 0:
        return 0.0
    top_k = ranked[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    """
    Fraction of all relevant items that appear in the top-k.
    """

    if not relevant:
        return 0.0
    top_k = ranked[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def dcg_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    """
    Discounted cumulative gain at k with binary gains (position 1 -> discount log2(2) = 1).
    """

    return sum(1.0 / math.log2(rank + 2) for rank, item in enumerate(ranked[:k]) if item in relevant)


def ndcg_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    """
    nDCG@k: DCG normalised by the ideal DCG (all relevant items ranked first).
    """

    if not relevant:
        return 0.0
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(ranked, relevant, k) / idcg


def reciprocal_rank(ranked: Sequence[str], relevant: Set[str]) -> float:
    """Reciprocal of the 1-based rank of the first relevant item; 0.0 if none are ranked."""

    for rank, item in enumerate(ranked):
        if item in relevant:
            return 1.0 / (rank + 1)
    return 0.0


# --- Rank-agreement metrics (perturbation analysis) -------------------------------------------------
# These compare two *rankings of the same candidate universe* — a baseline top-k against the top-k a
# perturbed query produced — rather than a ranking against a relevance set. They quantify how much the
# recommender's own output moved, independent of whether it moved toward or away from ground truth.


def jaccard_at_k(a: Sequence[str], b: Sequence[str], k: int) -> float:
    """
    Set overlap of two top-k rankings: |A ∩ B| / |A ∪ B|.

    Order-insensitive *membership* stability — how many of the top-k survived the perturbation, ignoring
    where they moved. 1.0 = identical membership, 0.0 = disjoint. Pair with a rank-correlation (Kendall
    tau over the survivors) to capture the reordering this metric is blind to. Two empty lists count as
    identical (1.0).
    """

    if k <= 0:
        return 0.0
    set_a, set_b = set(a[:k]), set(b[:k])
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 1.0


def rank_agreement_at_k(a: Sequence[str], b: Sequence[str], k: int) -> float:
    """
    Top-weighted [0, 1] agreement between two top-k rankings.

    A finite-depth, depth-normalized rank-biased-overlap variant: the running set-overlap at each depth
    1..k is averaged with geometric weights so a disagreement at rank 1 costs far more than one at rank k,
    then divided by the identical-lists value so identical prefixes score exactly 1.0 and disjoint lists
    score 0.0 (the raw finite-depth RBO tops out below 1, which misreads as instability). Complements
    `jaccard_at_k`: this is position-sensitive, Jaccard is not.
    """

    depth = min(k, max(len(a), len(b))) if k > 0 else max(len(a), len(b))
    if depth <= 0:
        return 1.0

    p = 0.9  # persistence: ~86% of the weight lands in the first 20 positions
    seen_a: Set[str] = set()
    seen_b: Set[str] = set()
    weighted, normalizer, weight = 0.0, 0.0, 1.0
    for d in range(1, depth + 1):
        seen_a.add(a[d - 1] if d - 1 < len(a) else f"\0a{d}")  # sentinels keep absent slots from matching
        seen_b.add(b[d - 1] if d - 1 < len(b) else f"\0b{d}")
        weighted += weight * (len(seen_a & seen_b) / d)
        normalizer += weight  # identical lists would score len/d == 1 at every depth
        weight *= p
    return weighted / normalizer if normalizer else 1.0
