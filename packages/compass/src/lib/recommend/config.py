"""
Configuration for the Tier 1 + Tier 2 recommender.

`@author`: DAShaikh10
"""

import os

from dotenv import load_dotenv

# Reuse the harness inputs (Chroma name/collection, enriched dataset, canonical map, eval knobs).
from ..eval.config import (
    CANONICAL_MAP_FILE,
    EMBEDDING_COLLECTION_NAME,
    EMBEDDING_DATABASE_NAME,
    ENRICHED_DATASET_FILE,
    K_VALUES,
    TAG_JACCARD_MIN,
    TAG_MIN_SHARED,
)

# Re-exported for convenience so callers can import every recommender setting from this one module.
__all__ = [
    "CANONICAL_MAP_FILE",
    "EMBEDDING_COLLECTION_NAME",
    "EMBEDDING_DATABASE_NAME",
    "ENRICHED_DATASET_FILE",
    "K_VALUES",
    "TAG_JACCARD_MIN",
    "TAG_MIN_SHARED",
    "REC_TAG_SOURCE",
    "REC_ANNOTATION_FILE",
    "POOL_SIZE",
    "RRF_K",
    "FUSION_METHOD",
    "FUSION_WEIGHTS",
    "COCITE_MIN_SHARED",
    "COUPLE_MIN_SHARED",
    "RERANKER",
    "CROSS_ENCODER_MODEL",
    "TOP_K",
    "REC_REPORT_FILE",
]

load_dotenv(override=True)

# Where the entity signal reads its tags. "chroma" (default) uses the {field}:{canonical} keys baked into
# the collection metadata — the single source of truth, consistent with the API. "annotation" parses them
# from REC_ANNOTATION_FILE instead (the fallback used before the store carried metadata).
REC_TAG_SOURCE = os.getenv("REC_TAG_SOURCE", "chroma")
# Annotation set feeding the entity signal only when REC_TAG_SOURCE="annotation". An LLM set is the
# default there because it covers all 992 papers; the human gold is reserved for evaluation.
REC_ANNOTATION_FILE = os.getenv("REC_ANNOTATION_FILE", "claude-opus-4.8-annotations.json")

# Stage 1 — candidate pool handed to the reranker, and the RRF rank constant.
POOL_SIZE = int(os.getenv("REC_POOL_SIZE", "150"))
RRF_K = int(os.getenv("REC_RRF_K", "60"))

# Stage 1 fusion method: "rrf" (rank-based, no calibration) or "weighted" (min-max-normalized weighted
# sum, keeps score magnitude and lets one signal dominate).
FUSION_METHOD = os.getenv("REC_FUSION_METHOD", "rrf")
# Per-signal weights for the "weighted" method. Coupling-dominant by default: shared references are
# observed evidence of relatedness, so the citation signal outranks the predicted dense/lexical/entity
# signals; dense is the strongest predictor and carries papers coupling never reaches. Override any
# single weight with REC_W_<SIGNAL> (e.g. REC_W_CITATION=2.0).
FUSION_WEIGHTS = {
    "citation": float(os.getenv("REC_W_CITATION", "1.0")),
    "dense": float(os.getenv("REC_W_DENSE", "0.6")),
    "lexical": float(os.getenv("REC_W_LEXICAL", "0.3")),
    "entity": float(os.getenv("REC_W_ENTITY", "0.3")),
}

# Co-citation ground truth: a pair is positive when >= this many corpus papers cite both. Independent of
# every retrieval signal, so it scores fusion/reranking without circularity (but sparse — see eval).
COCITE_MIN_SHARED = int(os.getenv("REC_COCITE_MIN_SHARED", "1"))
# Bibliographic-coupling ground truth: a pair is positive when they share >= this many references (the
# shared paper may be external). High coverage; circular with the citation signal, honest for the rest.
COUPLE_MIN_SHARED = int(os.getenv("REC_COUPLE_MIN_SHARED", "2"))

# Stage 2 — reranker backend ("none" | "cross-encoder" | "llm-listwise") and final cut-off.
RERANKER = os.getenv("REC_RERANKER", "none")
CROSS_ENCODER_MODEL = os.getenv("REC_CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3")
TOP_K = int(os.getenv("REC_TOP_K", "10"))

# Output of the ablation harness (written under data/).
REC_REPORT_FILE = os.getenv("REC_REPORT_FILE", "recommender_eval_report.json")
