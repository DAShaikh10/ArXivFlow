"""
Load configuration for the Tier 0 evaluation harness.

`@author`: DAShaikh10
"""

import os

from dotenv import load_dotenv

# Load environment variables and override any existing system/shell variables.
load_dotenv(override=True)

# Inputs (produced by papervec + crucible + annotate into data/).
EMBEDDING_DATABASE_NAME = os.getenv("EMBEDDING_DATABASE_NAME", "embeddings")
EMBEDDING_COLLECTION_NAME = os.getenv("EMBEDDING_COLLECTION_NAME", "ArXivFlow")
ENRICHED_DATASET_FILE = os.getenv("ENRICHED_DATASET_FILE", "raw_dataset_enriched_cleaned.jsonl")

# Gold human annotations + canonical map: the source for the tag-overlap ground truth.
ANNOTATION_FILE = os.getenv("ANNOTATION_FILE", "human-annotations.json")
CANONICAL_MAP_FILE = os.getenv("CANONICAL_MAP_FILE", "canonical_map.json")

# Evaluation knobs.
K_VALUES = [int(k) for k in os.getenv("EVAL_K_VALUES", "5,10,20").split(",") if k.strip()]
# A tag-overlap positive must share at least this many distinct canonical tags (primary guard against
# single-generic-tag coincidences). TAG_JACCARD_MIN is an optional extra tightener (0 = off).
TAG_MIN_SHARED = int(os.getenv("TAG_MIN_SHARED", "2"))
TAG_JACCARD_MIN = float(os.getenv("TAG_JACCARD_MIN", "0.0"))

# Output.
EVAL_REPORT_FILE = os.getenv("EVAL_REPORT_FILE", "tier0_eval_report.json")
