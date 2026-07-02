"""
Stage 2 rerankers.

A reranker reorders the Stage 1 candidate pool by reading the query and each candidate *together*,
which a bi-encoder cannot. All backends share one interface so the pipeline can swap them per door:

  * none           — passthrough; keeps Stage 1 order (the Tier 1-only configuration)
  * cross-encoder  — sentence-transformers CrossEncoder (e.g. bge-reranker); natural for the search door
  * llm-listwise   — (Phase 4) Qwen listwise reranker; natural for the paper<->paper click door

The cross-encoder dependency is imported lazily so the package stays usable for Stage 1 / evaluation
without the heavier model stack installed.

`@author`: DAShaikh10
"""

from typing import List, Protocol, Sequence, Tuple

# These strategy classes intentionally expose only `rerank` (the shared interface); a single public
# method is the point, not a smell.
# pylint: disable=too-few-public-methods


class Reranker(Protocol):
    """
    Reorder candidates given the query text; return `[(row, score), ...]` best-first, top-k.
    """

    def rerank(self, query: str, candidates: Sequence[Tuple[int, str]], k: int) -> List[Tuple[int, float]]:
        """
        Return the top-k `(row, score)` pairs, best-first, for `query` over `candidates`.
        """


class NoOpReranker:
    """
    Keep Stage 1 order, attaching a descending positional score. Used when no reranker is configured.
    """

    def rerank(
        self, query: str, candidates: Sequence[Tuple[int, str]], k: int  # pylint: disable=unused-argument
    ) -> List[Tuple[int, float]]:
        """
        Passthrough: keep Stage 1 order, scoring by position (`query` is ignored by design).
        """
        return [(row, 1.0 / (rank + 1)) for rank, (row, _doc) in enumerate(candidates[:k])]


class CrossEncoderReranker:
    """
    sentence-transformers CrossEncoder scoring (query, candidate-document) pairs.
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder  # pylint: disable=import-outside-toplevel  # lazy: heavy dep

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: Sequence[Tuple[int, str]], k: int) -> List[Tuple[int, float]]:
        """
        Score every `(query, document)` pair with the CrossEncoder and return the top-k, best-first.
        """
        if not candidates:
            return []
        scores = self._model.predict([(query, doc) for _row, doc in candidates])
        ranked = sorted(
            zip((row for row, _ in candidates), (float(s) for s in scores)), key=lambda p: p[1], reverse=True
        )
        return ranked[:k]


def build_reranker(name: str, cross_encoder_model: str) -> Reranker:
    """
    Construct the reranker named by config ("none" | "cross-encoder"); "llm-listwise" lands in Phase 4.
    """

    if name == "cross-encoder":
        return CrossEncoderReranker(cross_encoder_model)
    if name in ("none", ""):
        return NoOpReranker()
    raise ValueError(f"Unknown reranker backend: {name!r} (expected 'none' or 'cross-encoder')")
