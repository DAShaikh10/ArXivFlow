"""
Dump every distinct NER span per category, lowercased and normalized, ready to be hand-curated
into a canonical {variant -> canonical} map.

This is the step BETWEEN extraction and indexing. Raw spans fragment the same concept into many
surface forms ("BERT", "bert-base", "Bidirectional Encoder Representations from Transformers"), so
writing boolean metadata keys straight from them makes an exact-match Chroma `WHERE` miss most hits.
This script collapses the spans to a deterministic normalized form and groups the raw variants under
it, frequency-sorted (Zipfian — the head covers most papers), so the curation list is short.

It emits two artifacts in the data directory:

  tags_review.json   — per category, a frequency-sorted list of
                       {canonical, count, variants:[raw...]} for eyeballing. NOT consumed by code;
                       it is the worksheet you read to decide the merges.

  canonical_map.seed.json — per category, an IDENTITY {normalized -> normalized} skeleton, ordered
                       most-frequent first. This is the file you EDIT: point each variant's value at
                       a shared canonical id (see module docstring of `apply` / the README). Same
                       a flat per-category {variant -> canonical} map the indexing step consumes.

Passing several files POOLS their spans, so the resulting map covers every annotator at once.

`@author`: DAShaikh10
"""

import argparse
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List

from src.lib.ner import FIELD_TO_LABEL
from src.utils import load_label_studio_records, logger, resolve_path

from . import config

# Inverse of schema.FIELD_TO_LABEL: the on-disk Label Studio `labels` carry the human-readable name
# ("Machine Learning Architecture"); we bucket spans under the snake_case field used everywhere else.
LABEL_TO_FIELD: Dict[str, str] = {label: field for field, label in FIELD_TO_LABEL.items()}

REVIEW_FILE: str = "tags_review.json"
SEED_FILE: str = "canonical_map.seed.json"

# Characters we strip from the OUTSIDE of a span. Hyphens/slashes are kept because they are part of
# real names ("bert-base", "roberta/large"); only wrapping quotes, brackets and trailing punctuation go.
_WRAP_CHARS: str = " \t\n\r\"'`“”‘’()[]{}.,;:"


def normalize(text: str) -> str:
    """
    Map a raw span to its deterministic normalized form.

    Conservative on purpose: it only does what is safe without human judgement — Unicode NFKC (so
    fancy dashes/quotes fold to ASCII), lowercase, whitespace collapse, and stripping wrapping
    punctuation. It deliberately does NOT merge synonyms or drop stopwords like "the ... dataset" —
    that is a curation decision and belongs in the hand-edited canonical map, not here.

    Args:
        text (str): The raw entity surface form as extracted.

    Returns:
        str: The normalized form (may be empty if the span was punctuation-only).
    """

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(_WRAP_CHARS)


def collect_spans(records: List[Dict]) -> Dict[str, Dict[str, Counter]]:
    """
    Walk Label Studio task records and bucket every span by category and normalized form.

    Reads both `predictions` (model output) and `annotations` (human gold) so the same loader serves
    every file shape in the data directory.

    Args:
        records (List[Dict]): Label Studio task records.

    Returns:
        Dict[str, Dict[str, Counter]]: category -> normalized form -> Counter of the raw variants
        that normalized to it (the Counter value is how often each raw spelling appeared).
    """

    # category -> normalized -> Counter({raw_variant: occurrences})
    buckets: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))

    for record in records:
        for block in record.get("predictions", []) + record.get("annotations", []):
            # Some Label Studio exports put bare prediction IDs (ints) under `predictions`; the real
            # result blocks live under `annotations`. Skip anything that isn't a result block.
            if not isinstance(block, dict):
                continue

            for item in block.get("result", []):
                value = item.get("value", {})
                labels = value.get("labels") or []
                raw = value.get("text", "")
                if not labels or not raw:
                    continue

                # Bucket under the snake_case field; fall back to a slug for any unmapped label.
                field: str | None = LABEL_TO_FIELD.get(labels[0], labels[0].lower().replace(" ", "_"))
                norm: str = normalize(raw)
                if norm:
                    buckets[field][norm][raw.strip()] += 1

    return buckets


def build_outputs(buckets: Dict[str, Dict[str, Counter]]) -> tuple[Dict, Dict]:
    """
    Turn the raw buckets into the review worksheet and the editable canonical-map seed.

    Both are ordered most-frequent-first within each category so the high-impact entities (the ones
    queries actually filter on) sit at the top of the list you curate.

    Args:
        buckets (Dict[str, Dict[str, Counter]]): Output of `collect_spans`.

    Returns:
        tuple[Dict, Dict]: (review, seed).
    """

    review: Dict[str, List[Dict]] = {}
    seed: Dict[str, Dict[str, str]] = {}

    for field in sorted(buckets):
        # Sort normalized forms by total occurrences (desc), then alphabetically for stable ties.
        forms = sorted(
            buckets[field].items(),
            key=lambda kv: (-sum(kv[1].values()), kv[0]),
        )

        review[field] = [
            {
                "canonical": norm,  # pre-filled with the normalized form; edit to the shared id.
                "count": sum(variants.values()),
                "variants": [v for v, _ in variants.most_common()],
            }
            for norm, variants in forms
        ]
        # Identity skeleton: every normalized form maps to itself until you merge them by hand.
        seed[field] = {norm: norm for norm, _ in forms}

    return review, seed


def main() -> None:
    """
    Entry point to load all annotation file(s), dump the normalized tag inventory.
    """

    # Command-line arguments.
    parser = argparse.ArgumentParser(
        description="Extract and normalize NER canonical tags from Label Studio annotations."
    )
    parser.add_argument(
        "--inputs",
        "-i",
        type=str,
        nargs="*",
        default=[config.HUMAN_ANNOTATION_FILE],
        help="Path(s) to Label Studio annotation JSON file(s).",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Encoding for the output files.",
    )
    args = parser.parse_args()

    input_annotation_files = args.inputs
    current_dir = os.path.dirname(__file__)

    records: List[Dict] = []
    for name in input_annotation_files:
        path: str = resolve_path(current_dir, name)
        loaded: List[Dict] = load_label_studio_records(path)
        logger.info("Loaded %d records from %s", len(loaded), name)
        records.extend(loaded)

    buckets = collect_spans(records)
    review, seed = build_outputs(buckets)

    for field in sorted(review):
        logger.info("%-32s %5d distinct normalized tags", field, len(review[field]))

    review_path = resolve_path(current_dir, REVIEW_FILE)
    seed_path = resolve_path(current_dir, SEED_FILE)
    with open(review_path, "w", encoding=args.encoding or "utf-8") as handle:
        json.dump(review, handle, indent=2, ensure_ascii=False)

    with open(seed_path, "w", encoding=args.encoding or "utf-8") as handle:
        json.dump(seed, handle, indent=2, ensure_ascii=False)

    logger.info("Wrote review worksheet -> %s", review_path)
    logger.info("Wrote editable canonical-map seed -> %s", seed_path)


if __name__ == "__main__":
    main()
