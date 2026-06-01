"""
Load configuration for the annotation module.

`@author`: DAShaikh10
"""

import os

from dotenv import load_dotenv

# Load environment variables and override any existing system/shell variables.
load_dotenv(override=True)

HUMAN_ANNOTATION_FILE = os.getenv("HUMAN_ANNOTATION_FILE")
PHI_NER_ANNOTATION_FILE = os.getenv("PHI_NER_ANNOTATION_FILE")
QWEN_NER_ANNOTATION_FILE = os.getenv("QWEN_NER_ANNOTATION_FILE")
WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME")
