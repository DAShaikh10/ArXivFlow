"""
Loading helpers for Label Studio annotation files.

Label Studio exports a JSON array of task objects, but some artifacts in the data directory
carry a `.jsonl` extension while still holding a single array (and a few are true JSON Lines).
This loader normalizes all of those into a plain list of task records so callers don't have to
re-detect the shape each time.

`@author`: DAShaikh10
"""

import json
from typing import Dict, List


def load_label_studio_records(path: str, encoding: str = "utf-8") -> List[Dict]:
    """
    Load a Label Studio annotation file into a list of task records.

    Handles three on-disk shapes transparently: a JSON array (the standard export), a single
    JSON object (wrapped into a one-element list), and true JSON Lines (one object per line).

    Args:
        path (str): Path to the annotation / prediction file (`.json` or `.jsonl`).
        encoding (str): Encoding for the file.

    Returns:
        List[Dict]: The task records contained in the file.
    """

    with open(path, "r", encoding=encoding) as handle:
        head = handle.read(64).lstrip()
        handle.seek(0)
        if head.startswith("[") or head.startswith("{"):
            data = json.load(handle)
            return data if isinstance(data, list) else [data]

        # Fall back to true JSON Lines.
        return [json.loads(line) for line in handle if line.strip()]
