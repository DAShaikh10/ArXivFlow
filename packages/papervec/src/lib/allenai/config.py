"""
Load configuration for AllenAI module.

`@author`: DAShaikh10
"""

import os

from dotenv import load_dotenv

# Load environment variables and override any existing system/shell variables.
load_dotenv(override=True)

ADAPTER_NAME = os.getenv("ADAPTER_NAME")
ANNOTATION_FILE = os.getenv("ANNOTATION_FILE")
BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME")
CANONICAL_MAP_FILE = os.getenv("CANONICAL_MAP_FILE")
EMBEDDING_DATABASE_NAME = os.getenv("EMBEDDING_DATABASE_NAME")
ENRICHED_DATASET_FILE = os.getenv("ENRICHED_DATASET_FILE")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH"))
WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME")
