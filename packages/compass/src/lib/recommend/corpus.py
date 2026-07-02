"""
In-memory corpus for the recommender.

Joins the three sources a paper recommender needs, all keyed by the canonical (versioned) arXiv id:

  * embeddings + documents — from the Chroma collection (SPECTER2 proximity vectors, "Title[SEP]Abstract")
  * reference keys + corpus-internal citations — from the enriched JSONL
  * canonical NER tags — from an annotation JSON (the Chroma store carries no metadata)

Everything is held as plain lists aligned to `ids` (with an `index` lookup), so the signal layer can
work in fast integer row-space and only convert back to ids at the edges. 992 papers fit comfortably in
memory, so the same object backs both batch evaluation and single-source serving.

`@author`: DAShaikh10
"""

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

import chromadb
import numpy as np

from ..eval import groundtruth

# Wrapping characters stripped from a reference title before hashing it into a coupling key; matches the
# span normalization in eval.groundtruth so an id-less reference still couples consistently.
_WRAP_CHARS = " \t\n\r\"'`“”‘’()[]{}.,;:"


def _normalize_title(text: str) -> str:
    """
    NFKC + lowercase + whitespace-collapse + wrapping-punctuation strip.
    """

    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(_WRAP_CHARS)


def _ref_key(ref: dict) -> Optional[str]:
    """
    Stable identity for a reference, used for bibliographic coupling.

    Prefer the version-stripped arXiv id; fall back to the normalized title so references that lack an
    id still couple two papers. Returns None for an empty reference.
    """

    norm_id = groundtruth.normalize_id(ref.get("arxiv_id"))
    if norm_id:
        return f"id:{norm_id}"
    title = ref.get("title")
    if title:
        norm_title = _normalize_title(title)
        if norm_title:
            return f"t:{norm_title}"
    return None


@dataclass
class Corpus:
    """
    Aligned per-paper data plus an id -> row index lookup.
    """

    ids: List[str]
    index: Dict[str, int]
    embeddings: np.ndarray  # (N, D), L2-normalized float32 — row dot product == cosine similarity
    documents: List[str]
    ref_keys: List[Set[str]]  # all outgoing reference keys (id- or title-based), for coupling
    cites: List[Set[int]]  # corpus-internal outgoing citation row indices, for direct-citation boost
    tag_sets: List[Set[str]]  # canonical "{field}:{canonical}" tags from the serving annotation set

    @classmethod
    def load(
        cls,
        collection: chromadb.Collection,
        dataset_path: str,
        annotation_path: str,
        canonical_map_path: str,
        tag_source: str = "chroma",
    ) -> "Corpus":
        """
        Build the corpus from a Chroma collection, the enriched dataset, and a tag source.

        `tag_source` selects where the entity-signal tags come from: "chroma" reads the
        `{field}:{canonical}` boolean keys baked into the collection metadata (the single source of truth,
        consistent with the API), while "annotation" parses them from the annotation JSON directly (the
        fallback used before the store carried metadata). The eval *ground truth* always reads the human
        gold separately — see `recommend.eval`.
        """

        record = collection.get(include=["embeddings", "documents", "metadatas"])
        ids: List[str] = list(record["ids"])
        index = {paper_id: row for row, paper_id in enumerate(ids)}

        embeddings = np.asarray(record["embeddings"], dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        embeddings = embeddings / norms

        documents = [doc or "" for doc in (record.get("documents") or [""] * len(ids))]

        ref_keys, cites = cls._load_references(dataset_path, ids, index)
        tag_sets = cls._load_tags(tag_source, record, ids, annotation_path, canonical_map_path)

        return cls(ids, index, embeddings, documents, ref_keys, cites, tag_sets)

    @staticmethod
    def _load_tags(
        tag_source: str,
        record: dict,
        ids: Sequence[str],
        annotation_path: str,
        canonical_map_path: str,
    ) -> List[Set[str]]:
        """
        Per-paper canonical-tag sets from Chroma metadata ("chroma") or the annotation JSON ("annotation").
        """

        if tag_source == "chroma":
            # Each metadata dict holds boolean `{field}:{canonical}` keys (plus a non-tag "authors" string);
            # keep only the True-valued, colon-bearing keys. Mirrors api/store._topics_for.
            metadatas = record.get("metadatas") or [None] * len(ids)
            return [{key for key, value in (md or {}).items() if value is True and ":" in key} for md in metadatas]

        if tag_source == "annotation":
            # LLM serving sets store spans under "predictions"; fall back to human "annotations".
            tag_map = groundtruth.tag_sets_from_annotations(
                annotation_path, canonical_map_path, ids, blocks=("predictions", "annotations")
            )
            return [tag_map.get(paper_id, set()) for paper_id in ids]

        raise ValueError(f"Unknown tag_source: {tag_source!r} (expected 'chroma' or 'annotation')")

    @staticmethod
    def _load_references(
        dataset_path: str,
        ids: Sequence[str],
        index: Dict[str, int],
    ) -> tuple[List[Set[str]], List[Set[int]]]:
        """
        Per-paper reference-key sets (for coupling) and corpus-internal citation rows (for the boost).
        """

        norm_to_row = {
            groundtruth.normalize_id(pid): row for row, pid in enumerate(ids) if groundtruth.normalize_id(pid)
        }

        ref_keys: List[Set[str]] = [set() for _ in ids]
        cites: List[Set[int]] = [set() for _ in ids]

        with open(dataset_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                paper = json.loads(line)
                row = index.get(paper.get("arxiv_id"))
                if row is None:
                    norm = groundtruth.normalize_id(paper.get("arxiv_id"))
                    row = norm_to_row.get(norm) if norm else None
                if row is None:
                    continue
                for ref in paper.get("references") or []:
                    key = _ref_key(ref)
                    if key:
                        ref_keys[row].add(key)
                    target = norm_to_row.get(groundtruth.normalize_id(ref.get("arxiv_id")) or "")
                    if target is not None and target != row:
                        cites[row].add(target)

        return ref_keys, cites
