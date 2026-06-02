"""
Central logging utility for papervec.

`@author`: DAShaikh10
"""

import logging
import os
import re
import sys
from datetime import datetime

# Determine the module name running.
main_module = sys.modules.get("__main__")
module_name = "papervec"
if main_module:
    if hasattr(main_module, "__spec__") and main_module.__spec__:
        module_name = main_module.__spec__.name
    elif hasattr(main_module, "__file__") and main_module.__file__:
        module_name = os.path.basename(main_module.__file__).replace(".py", "")

# Strip 'src.lib.' from the module name for cleaner log files if it starts with it.
module_name = re.sub(r"^src\.lib\.", "", module_name)

# Create a timestamp YYYYMMDD_HHMMSS.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Ensure it creates inside the package's src directory if run from standard paths
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(LOG_DIR, f"{module_name}_{timestamp}.log")

# Configure logging.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename, mode="w+"), logging.StreamHandler()],
    force=True,
)

# Singleton configured logger instance.
logger = logging.getLogger(module_name)
