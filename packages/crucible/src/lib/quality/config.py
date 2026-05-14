"""
@Author: DAShaikh10
@Description: Load configuration for data quality checks.
"""

import os

from dotenv import load_dotenv

# Load environment variables and override any existing system/shell variables.
load_dotenv(override=True)

ENRICHED_DATASET_FILE = os.getenv("ENRICHED_DATASET_FILE")
PRE_ANNOTATION_MIN_ABSTRACT_WORDS = int(os.getenv("PRE_ANNOTATION_MIN_ABSTRACT_WORDS"))
PRE_ANNOTATION_REPORT_PATH = os.getenv("PRE_ANNOTATION_REPORT_PATH")
