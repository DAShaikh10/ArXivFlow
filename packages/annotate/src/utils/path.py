"""
Utility to resolve file paths for dataset storage.
Handles both local monorepo and standard Docker environments.

`@author`: DAShaikh10
"""

import os


def resolve_path(current_dir: str, file: str) -> str:
    """
    Resolves the absolute path for the project root based on the current directory.
    Handles both local monorepo and standard Docker environments.
    """

    # Check if we are running in the local monorepo or standard Docker environment.
    if "packages/annotate" in current_dir.replace("\\", "/"):
        # Local setup: E.g. ArXivFlow/packages/annotate/src/lib/ner/gliner.py
        project_root = os.path.abspath(os.path.join(current_dir, "../../../../.."))
    else:
        # Docker setup: E.g. /app/src/lib/ner/gliner.py
        project_root = os.path.abspath(os.path.join(current_dir, "../../.."))

    data_dir = os.environ.get("DATA_DIR", os.path.join(project_root, "data"))
    os.makedirs(data_dir, exist_ok=True)

    return os.path.join(data_dir, file)
