"""
@Author: DAShaikh10
@Description: Central logging utility.
"""

import logging
import os
import re
import sys
from datetime import datetime

# Determine the module name running.
main_module = sys.modules.get("__main__")
module_name = "crucible"
if main_module:
    if hasattr(main_module, "__spec__") and main_module.__spec__:
        module_name = main_module.__spec__.name
    elif hasattr(main_module, "__file__") and main_module.__file__:
        module_name = os.path.basename(main_module.__file__).replace(".py", "")

# Strip 'src.lib.' from the module name for cleaner log files if it starts with it.
module_name = re.sub(r"^src\.lib\.", "", module_name)

# Create a timestamp YYYYMMDD_HHMMSS.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = "src/logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(LOG_DIR, f"{module_name}_{timestamp}.log")

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename, mode="w+"), logging.StreamHandler()],
    force=True,  # Ensure we override if basicConfig was already called.
)

# Singleton configured logger instance.
logger = logging.getLogger(module_name)
