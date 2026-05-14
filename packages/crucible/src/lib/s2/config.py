"""
@Author: DAShaikh10
@Description: Load configuration for the Semantic Scholar scraper module.
"""

import os

from dotenv import load_dotenv

# Load environment variables and override any existing system/shell variables.
load_dotenv(override=True)

S2_API_KEY = os.getenv("S2_API_KEY")
API_BASE_URL = os.getenv("SEMANTIC_SCHOLAR_API_BASE_URL")
API_FIELDS = os.getenv("SEMANTIC_SCHOLAR_API_FIELDS")
BATCH_SIZE = int(os.getenv("SEMANTIC_SCHOLAR_BATCH_SIZE"))
CONCURRENCY = int(os.getenv("SEMANTIC_SCHOLAR_CONCURRENCY"))
DELAY_BETWEEN_BATCHES = int(os.getenv("SEMANTIC_SCHOLAR_DELAY_BETWEEN_BATCHES"))
ENRICHED_DATASET_FILE = os.getenv("ENRICHED_DATASET_FILE")
MAX_RETRIES = int(os.getenv("MAX_RETRIES"))
RAW_DATASET_FILE = os.getenv("RAW_DATASET_FILE")
RETRY_DELAY = int(os.getenv("SEMANTIC_SCHOLAR_RETRY_DELAY"))
