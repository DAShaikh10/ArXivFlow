"""
Load configuration for the annotation module.

`@author`: DAShaikh10
"""

import os

from dotenv import load_dotenv

# Load environment variables and override any existing system/shell variables.
load_dotenv(override=True)

CLEANED_DATASET_FILE = os.getenv("CLEANED_DATASET_FILE")
SENSITIVITY_THRESHOLD = float(os.getenv("SENSITIVITY_THRESHOLD"))
GLINER_MODEL = os.getenv("GLINER_MODEL_NAME")
GLINER_NER_ANNOTATION_FILE = os.getenv("GLINER_NER_ANNOTATION_FILE")
LM_MODEL_NAME = os.getenv("LM_MODEL_NAME")
MAX_TOKENS = int(os.getenv("MAX_TOKENS"))
MODEL_MAX_LEN = int(os.getenv("MODEL_MAX_LEN"))
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE"))
VLLM_TENSOR_PARALLEL_SIZE = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE"))
WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME")
