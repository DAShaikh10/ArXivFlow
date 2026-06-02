"""
Build per-paper canonical tag metadata for ChromaDB from annotations.

`@author`: DAShaikh10
"""

import json
import re
import unicodedata
from collections import defaultdict
from typing import Dict, List, Optional, Set

from src.utils import logger

# Inverse of annotate's `schema.FIELD_TO_LABEL`: on-disk Label Studio `labels` carry the
# human-readable name ("Machine Learning Architecture"); we bucket spans under the snake_case
# field used as the top-level key in `canonical_map.json`.
LABEL_TO_FIELD: Dict[str, str] = {
    "Target NLP Task": "target_nlp_task",
    "Machine Learning Architecture": "machine_learning_architecture",
    "Training or Fine-tuning Method": "training_method",
    "Dataset or Benchmark Name": "dataset_name",
    "Application Domain": "application_domain",
    "Evaluation Metric": "evaluation_metric",
    "Language or Dialect": "language_dialect",
}

# Characters stripped from the OUTSIDE of a span. Mirrors annotate's post-process: hyphens/slashes
# stay (part of real names like "bert-base"); only wrapping quotes, brackets and trailing
# punctuation go.
_WRAP_CHARS: str = " \t\n\r\"'`“”‘’()[]{}.,;:"


def load_canonical_map(path: str, encoding: str = "utf-8") -> Dict[str, Dict[str, str]]:
    """
    Load the hand-curated canonical map: per category, a flat {normalized variant -> canonical id}.

    Args:
        path (str): Path to `canonical_map.json`.
        encoding (str): Encoding for the file.

    Returns:
        Dict[str, Dict[str, str]]: category -> {variant -> canonical}.
    """

    with open(path, "r", encoding=encoding) as handle:
        return json.load(handle)


def load_records(path: str, encoding: str = "utf-8") -> List[Dict]:
    """
    Load a Label Studio annotation file into a list of task records.

    Mirrors annotate's `load_label_studio_records`: handles a JSON array (standard export), a single
    JSON object (wrapped into a one-element list), and true JSON Lines transparently.

    Args:
        path (str): Path to the annotation file (`.json` or `.jsonl`).
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


def build_paper_tags(
    records: List[Dict],
    canonical_map: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, Set[str]]]:
    """
    Walk Label Studio task records and collect the canonical tag set per paper, per category.

    Reads only `annotations` (human gold), not `predictions` (model output), so the index reflects
    the verified tags. Each raw span is normalized and then resolved through `canonical_map`; spans
    whose normalized form is not in the map fall back to that normalized form (so nothing is
    silently dropped).

    Args:
        records (List[Dict]): Label Studio task records (each with `data.arxiv_id`).
        canonical_map (Dict[str, Dict[str, str]]): Output of `load_canonical_map`.

    Returns:
        Dict[str, Dict[str, Set[str]]]: arxiv_id -> category -> set of canonical tag ids.
    """

    # arxiv_id -> category -> {canonical}
    tags: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

    for record in records:
        arxiv_id = record.get("data", {}).get("arxiv_id")
        if not arxiv_id:
            continue

        for block in record.get("annotations", []):
            # Skip anything that isn't a result block.
            if not isinstance(block, dict):
                continue

            for item in block.get("result", []):
                value = item.get("value", {})
                labels = value.get("labels") or []
                raw = value.get("text", "")
                if not labels or not raw:
                    continue

                field = LABEL_TO_FIELD.get(labels[0])
                if field is None:
                    # Label outside the known NER schema; ignore rather than invent a category.
                    continue

                norm = normalize(raw)
                if not norm:
                    continue

                canonical = canonical_map.get(field, {}).get(norm, norm)
                tags[arxiv_id][field].add(canonical)

    logger.info("Built canonical tags for %d papers", len(tags))
    return tags


def metadata_for(arxiv_id: str, paper_tags: Dict[str, Dict[str, Set[str]]]) -> Optional[Dict[str, bool]]:
    """
    Flatten one paper's canonical tags into a Chroma metadata dict of boolean keys.

    Args:
        arxiv_id (str): The paper id.
        paper_tags (Dict[str, Dict[str, Set[str]]]): Output of `build_paper_tags`.

    Returns:
        Optional[Dict[str, bool]]: `{"{category}:{canonical}": True, ...}`, or `None` when the paper
        has no tags. `None` (not `{}`) is required because Chroma rejects empty metadata dicts.
    """

    metadata: Dict[str, bool] = {
        f"{field}:{canonical}": True
        for field, canonicals in paper_tags.get(arxiv_id, {}).items()
        for canonical in canonicals
    }
    return metadata or None


def normalize(text: str) -> str:
    """
    Map a raw NER span to its deterministic normalized form (NFKC, lowercase, whitespace collapse,
    wrapping-punctuation strip). Kept identical to annotate's post-process `normalize` so the result
    matches the keys curated into `canonical_map.json`.

    Args:
        text (str): The raw entity surface form as extracted.

    Returns:
        str: The normalized form (may be empty if the span was punctuation-only).
    """

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(_WRAP_CHARS)
