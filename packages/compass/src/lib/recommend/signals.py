"""
Stage 1 retrieval signals.

Four independent views of "related to paper X", each returning a ranked candidate list
`[(row, score), ...]` best-first with the source row excluded:

  * dense    — SPECTER2 cosine (the Tier 0 signal)
  * lexical  — BM25 over "Title[SEP]Abstract" (catches rare proper nouns dense vectors blur)
  * citation — bibliographic coupling (shared references) + a direct-citation boost
  * entity   — canonical NER-tag Jaccard

`Signals` precomputes the heavy structures once (normalized matrix, BM25 index, inverted indices) so
both the batch evaluator and single-source serving reuse them. Citation and entity restrict candidates
to papers sharing at least one reference key / tag via an inverted index, so neither pays the full N^2.

`@author`: DAShaikh10
"""

import re
from typing import Dict, List, Set, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from .corpus import Corpus

# Direct corpus-internal citations are a stronger relatedness cue than a shared external reference, so
# a directly-cited (or citing) paper gets this added to its coupling Jaccard before ranking.
_DIRECT_CITATION_BOOST = 1.0

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """
    Lowercase alphanumeric tokens — deliberately simple, matched by query and document.
    """

    return _TOKEN.findall(text.lower())


def _invert(member_sets: List[set]) -> Dict[object, List[int]]:
    """
    token/key -> rows that contain it, for candidate generation without an N^2 scan.
    """

    inverted: Dict[object, List[int]] = {}
    for row, members in enumerate(member_sets):
        for member in members:
            inverted.setdefault(member, []).append(row)
    return inverted


class Signals:
    """
    Precomputed Stage 1 signals over a `Corpus`.
    """

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self._bm25 = BM25Okapi([_tokenize(doc) for doc in corpus.documents])
        self._ref_inverted = _invert(corpus.ref_keys)
        self._tag_inverted = _invert(corpus.tag_sets)

    # Each signal has two forms: the row form (`dense`, `lexical`, ...) queries with a corpus paper's own
    # representation — what serving and evaluation use — and delegates to the `*_from_*` form, which takes
    # an explicit query representation. The split lets the perturbation harness feed a noised embedding,
    # a dropped-token query, or a thinned reference/tag set through the identical scoring path, with no
    # forked ranking logic. `exclude` drops the source row from its own result (unset for a foreign query).

    def dense(self, row: int, limit: int = 0) -> List[Tuple[int, float]]:
        """
        Cosine nearest neighbours via the normalized embedding matrix.
        """

        return self.dense_from_vector(self.corpus.embeddings[row], limit=limit, exclude=row)

    def dense_from_vector(self, query: np.ndarray, limit: int = 0, exclude: int = -1) -> List[Tuple[int, float]]:
        """
        Cosine nearest neighbours to an explicit (already L2-normalized) query vector.
        """

        sims = self.corpus.embeddings @ query
        order = np.argsort(-sims)
        ranked = [(int(j), float(sims[j])) for j in order if j != exclude]
        return ranked[:limit] if limit else ranked

    def lexical(self, row: int, limit: int = 0) -> List[Tuple[int, float]]:
        """
        BM25 with the source paper's own document as the query.
        """

        return self.lexical_from_tokens(_tokenize(self.corpus.documents[row]), limit=limit, exclude=row)

    def lexical_from_tokens(self, tokens: List[str], limit: int = 0, exclude: int = -1) -> List[Tuple[int, float]]:
        """
        BM25 scored against an explicit token list (already tokenized, e.g. a perturbed query).
        """

        scores = self._bm25.get_scores(tokens)
        order = np.argsort(-scores)
        ranked = [(int(j), float(scores[j])) for j in order if j != exclude and scores[j] > 0.0]
        return ranked[:limit] if limit else ranked

    def citation(self, row: int, limit: int = 0) -> List[Tuple[int, float]]:
        """
        Bibliographic-coupling Jaccard over reference keys, plus a direct-citation boost.
        """

        return self.citation_from(self.corpus.ref_keys[row], self.corpus.cites[row], limit=limit, exclude=row)

    def citation_from(
        self, source_keys: Set[str], source_cites: Set[int], limit: int = 0, exclude: int = -1
    ) -> List[Tuple[int, float]]:
        """
        Coupling Jaccard + direct-citation boost for an explicit reference-key / citation-row set.
        """

        candidates = {cand for key in source_keys for cand in self._ref_inverted.get(key, []) if cand != exclude}
        candidates |= {cand for cand in source_cites if cand != exclude}

        scored: List[Tuple[int, float]] = []
        for cand in candidates:
            cand_keys = self.corpus.ref_keys[cand]
            union = len(source_keys | cand_keys)
            jaccard = len(source_keys & cand_keys) / union if union else 0.0
            directed = cand in source_cites or exclude in self.corpus.cites[cand]
            score = jaccard + (_DIRECT_CITATION_BOOST if directed else 0.0)
            if score > 0.0:
                scored.append((cand, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit] if limit else scored

    def entity(self, row: int, limit: int = 0) -> List[Tuple[int, float]]:
        """
        Canonical NER-tag Jaccard over the serving annotation set.
        """

        return self.entity_from(self.corpus.tag_sets[row], limit=limit, exclude=row)

    def entity_from(self, source_tags: Set[str], limit: int = 0, exclude: int = -1) -> List[Tuple[int, float]]:
        """
        Canonical NER-tag Jaccard for an explicit tag set.
        """

        if not source_tags:
            return []
        candidates = {cand for tag in source_tags for cand in self._tag_inverted.get(tag, []) if cand != exclude}

        scored: List[Tuple[int, float]] = []
        for cand in candidates:
            cand_tags = self.corpus.tag_sets[cand]
            union = len(source_tags | cand_tags)
            jaccard = len(source_tags & cand_tags) / union if union else 0.0
            if jaccard > 0.0:
                scored.append((cand, jaccard))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit] if limit else scored
