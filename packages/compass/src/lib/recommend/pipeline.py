"""
The two-stage recommender pipeline.

Stage 1 gathers candidates from the enabled signals and fuses them with RRF (or falls back to
dense-only, i.e. the Tier 0 baseline). Stage 2 reorders the pool with the configured reranker.

Retrieval knobs live in `RetrievalConfig` so the eval harness can ablate signals and toggle fusion;
the reranker is injected at construction so the heavy model is built at most once. Signals operate in
integer row-space; ids are restored only at the API boundary.

`@author`: DAShaikh10
"""

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import config
from .corpus import Corpus
from .fusion import rrf, weighted
from .rerank import NoOpReranker, Reranker
from .signals import Signals

DEFAULT_SIGNALS: Tuple[str, ...] = ("dense", "lexical", "citation", "entity")


@dataclass
class RetrievalConfig:
    """
    Stage 1 knobs. `signals` is fused with `fusion_method`; with `use_fusion=False` only dense is used
    (the Tier 0 baseline). `fusion_method` is "rrf" (rank-based) or "weighted" (min-max-normalized
    weighted sum); `weights` overrides the per-signal weights the weighted method uses.
    """

    signals: Tuple[str, ...] = DEFAULT_SIGNALS
    use_fusion: bool = True
    pool_size: int = config.POOL_SIZE
    rrf_k: int = config.RRF_K
    fusion_method: str = config.FUSION_METHOD
    weights: Optional[Mapping[str, float]] = None


@dataclass
class Recommender:
    """
    Stage 1 retrieve + Stage 2 rerank over a loaded `Corpus`.
    """

    corpus: Corpus
    signals: Signals = field(default=None)  # type: ignore[assignment]
    reranker: Reranker = field(default_factory=NoOpReranker)

    def __post_init__(self) -> None:
        if self.signals is None:
            self.signals = Signals(self.corpus)

    def candidate_rows(self, row: int, rc: RetrievalConfig) -> List[int]:
        """
        Stage 1: per-signal candidate lists fused (RRF or weighted), or dense-only, truncated to the pool.
        """

        if not rc.use_fusion:
            ranked = self.signals.dense(row, limit=rc.pool_size)
            return [cand for cand, _ in ranked]

        scored = [(name, getattr(self.signals, name)(row, limit=rc.pool_size)) for name in rc.signals]

        if rc.fusion_method == "weighted":
            fused = weighted(scored, rc.weights or config.FUSION_WEIGHTS)
        else:
            fused = rrf([[cand for cand, _ in ranking] for _, ranking in scored], k=rc.rrf_k)
        return [cand for cand, _ in fused[: rc.pool_size]]

    def rank_source(self, row: int, rc: RetrievalConfig, depth: int) -> List[int]:
        """
        Full pipeline for one source row: Stage 1 candidates -> reranker -> top-`depth` rows.
        """

        rows = self.candidate_rows(row, rc)
        candidates = [(cand, self.corpus.documents[cand]) for cand in rows]
        reranked = self.reranker.rerank(self.corpus.documents[row], candidates, depth)
        return [cand for cand, _ in reranked]

    def rank_all(
        self, rc: RetrievalConfig, depth: int, sources: Optional[Sequence[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Ranked neighbour ids for every source paper (or `sources`), for batch evaluation.
        """

        source_ids = list(sources) if sources is not None else self.corpus.ids
        rankings: Dict[str, List[str]] = {}
        for paper_id in source_ids:
            row = self.corpus.index.get(paper_id)
            if row is None:
                continue
            rankings[paper_id] = [self.corpus.ids[cand] for cand in self.rank_source(row, rc, depth)]
        return rankings

    def recommend(self, arxiv_id: str, rc: Optional[RetrievalConfig] = None, k: int = config.TOP_K) -> List[dict]:
        """
        Top-`k` recommendations for a paper as `[{"id", "score"}, ...]` (click "More Like This").
        """

        row = self.corpus.index.get(arxiv_id)
        if row is None:
            return []

        rc = rc or RetrievalConfig()
        rows = self.candidate_rows(row, rc)
        candidates = [(cand, self.corpus.documents[cand]) for cand in rows]
        reranked = self.reranker.rerank(self.corpus.documents[row], candidates, k)
        return [{"id": self.corpus.ids[cand], "score": round(score, 4)} for cand, score in reranked]
