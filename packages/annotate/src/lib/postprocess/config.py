"""
Load configuration for the post-process module.

`@author`: DAShaikh10
"""

import os

from dotenv import load_dotenv

# Load environment variables and override any existing system/shell variables.
load_dotenv(override=True)

HUMAN_ANNOTATION_FILE = os.getenv("HUMAN_ANNOTATION_FILE")
