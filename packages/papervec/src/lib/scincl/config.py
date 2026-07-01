"""
Load configuration for the SciNCL module.

`@author`: DAShaikh10
"""

import os

from dotenv import load_dotenv

# Load environment variables and override any existing system/shell variables.
load_dotenv(override=True)

# ---------- SciNCL model + destination collection ----------
# Plain (adapter-free) encoder; defaults to the public SciNCL checkpoint.
SCINCL_MODEL_NAME = os.getenv("SCINCL_MODEL_NAME")
# MUST differ from the SPECTER2 collection (allenai/main.py uses WANDB_PROJECT_NAME as its collection
# name) so this ingest never overwrites the deployed proximity store.
SCINCL_COLLECTION_NAME = os.getenv("SCINCL_COLLECTION_NAME")

# ---------- Shared inputs (identical env keys to the AllenAI module) ----------
ANNOTATION_FILE = os.getenv("ANNOTATION_FILE")
CANONICAL_MAP_FILE = os.getenv("CANONICAL_MAP_FILE")
EMBEDDING_DATABASE_NAME = os.getenv("EMBEDDING_DATABASE_NAME")
ENRICHED_DATASET_FILE = os.getenv("ENRICHED_DATASET_FILE")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH"))
WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME")
