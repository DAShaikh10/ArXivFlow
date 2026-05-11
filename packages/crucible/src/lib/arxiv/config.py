"""
@Author: DAShaikh10
@Description: Load configuration for ArXiv scraper module.
"""

import os

from dotenv import load_dotenv

# Load environment variables and override any existing system/shell variables.
load_dotenv(override=True)

API_BASE_URL = os.getenv("ARXIV_API_BASE_URL")
BATCH_SIZE = int(os.getenv("ARXIV_API_MAX_RESULTS_PER_REQUEST"))
CATEGORY = os.getenv("ARXIV_CATEGORY")
END_DATE = os.getenv("ARXIV_SEARCH_END_DATE")
MAX_RETRIES = int(os.getenv("MAX_RETRIES"))
MONTHS_BACK = int(os.getenv("ARXIV_SEARCH_MONTHS_BACK"))
RAW_DATASET_FILE = os.getenv("RAW_DATASET_FILE")
RETRY_DELAY = int(os.getenv("ARXIV_RETRY_DELAY"))
START_DATE = os.getenv("ARXIV_SEARCH_START_DATE")
TOTAL_RESULTS = int(os.getenv("ARXIV_SEARCH_TOTAL_RESULTS"))
