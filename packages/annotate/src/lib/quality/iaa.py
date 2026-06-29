"""
Compute Inter-Annotator Agreement (IAA) for NER (Entity Matching).
Evaluates both exact (text + label) and relaxed (label + character overlap) matches across
annotators, Human vs Phi, Human vs Qwen, Qwen vs Phi.

`@author`: DAShaikh10
"""

import glob
import json
import os
from pathlib import Path

import krippendorff
import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.lib.ner.examples import FEW_SHOT_ARXIV_IDS
from src.utils import logger, resolve_path

from . import config


def _extract_spans(results: list) -> list:
    """
    Extracts span dicts from a Label Studio `result` list, preserving the character offsets
    so spans can be matched either exactly (text + label) or by overlap (label + intersecting
    character range). The format is shared between human annotations and LLM predictions, so
    both loaders reuse it.

    Args:
        results (list): A list of annotation/prediction results from Label Studio.

    Returns:
        list: A list of {"start", "end", "text", "label"} dicts. `text` is lowercased so exact
            matching stays case-insensitive (mirrors the previous behaviour).
    """

    spans = []

    for res in results:
        # Label Studio NER format stores data inside 'value'.
        val = res.get("value", {})
        text = val.get("text", "").strip().lower()
        labels = val.get("labels", [])

        if text and labels:
            # Assuming a single label per span.
            spans.append(
                {
                    "start": val.get("start"),
                    "end": val.get("end"),
                    "text": text,
                    "label": labels[0],
                }
            )

    return spans


def _spans_to_entity_set(spans: list) -> set:
    """
    Collapse spans to the unique (text, label) tuples used for exact matching. Deduping here
    reproduces the original set semantics, so the exact-match metrics are unchanged.

    Args:
        spans (list): A list of span dicts from `_extract_spans`.

    Returns:
        set: A set of (text, label) tuples.
    """

    return {(span["text"], span["label"]) for span in spans}


def _spans_overlap(span_a: dict, span_b: dict) -> bool:
    """
    Decides whether two spans match under relaxed criteria: identical label AND intersecting
    character ranges. This credits boundary disagreements such as "WikiText-2 dataset" vs
    "WikiText-2" or "fact-checking ecosystem" vs "fact-checking" as agreements.

    Args:
        span_a (dict): A span dict from `_extract_spans`.
        span_b (dict): A span dict from `_extract_spans`.

    Returns:
        bool: True if the spans share a label and overlap, else False.
    """

    if span_a["label"] != span_b["label"]:
        return False

    a_start, a_end = span_a.get("start"), span_a.get("end")
    b_start, b_end = span_b.get("start"), span_b.get("end")

    # Fall back to substring containment when offsets are missing: both annotators index the
    # same abstract, so a shared substring is the best available proxy for an overlap.
    if None in (a_start, a_end, b_start, b_end):
        return span_a["text"] in span_b["text"] or span_b["text"] in span_a["text"]

    # Half-open intervals [start, end) intersect iff each begins before the other ends.
    return a_start < b_end and b_start < a_end


def align_annotations(reference_data: dict, comparison_data: dict):
    """
    Creates binary alignment vectors for all pooled entities across papers shared by
    both annotators. Position i of each vector marks whether that annotator extracted
    the i-th unique (text, label) entity.

    Args:
        reference_data (dict): Annotations from the reference annotator.
        comparison_data (dict): Annotations from the comparison annotator.

    Returns:
        tuple: (reference_vec, comparison_vec, common_ids) where:
            - reference_vec (np.ndarray): Binary vector for the reference annotator.
            - comparison_vec (np.ndarray): Binary vector for the comparison annotator.
            - common_ids (list): List of paper IDs that were annotated by both.
    """

    common_ids = set(reference_data.keys()).intersection(set(comparison_data.keys()))

    reference_binary = []
    comparison_binary = []

    for paper_id in common_ids:
        # Exact matching ignores offsets: collapse spans to unique (text, label) tuples.
        reference_ents = _spans_to_entity_set(reference_data[paper_id])
        comparison_ents = _spans_to_entity_set(comparison_data[paper_id])

        # Union of all entities found by EITHER annotator in this paper.
        all_unique_ents = reference_ents.union(comparison_ents)

        for ent in all_unique_ents:
            # Did each annotator extract this exact entity + label? (1 = Yes, 0 = No)
            reference_binary.append(1 if ent in reference_ents else 0)
            comparison_binary.append(1 if ent in comparison_ents else 0)

    return np.array(reference_binary), np.array(comparison_binary), list(common_ids)


def load_lm_predictions(filepath: str) -> dict:
    """
    Parses LLM (Phi / Qwen) outputs stored in the Label Studio prediction format.

    The LMNER pipeline serialises its results as a single JSON array (despite the
    `.jsonl` extension), where each task carries its spans under `predictions[0].result`.

    Args:
        filepath (str): Path to the LLM annotation file (e.g. `phi-4-annotation.jsonl`).

    Returns:
        dict: A dictionary mapping 'arxiv_id' -> list of span dicts (see `_extract_spans`).
    """

    lm_data = {}

    with open(filepath, "r", encoding="utf-8") as f:
        file_data = json.load(f)

    # Ensure file_data is iterable. If it loaded a single root object, wrap it.
    if isinstance(file_data, dict):
        file_data = [file_data]

    for task in file_data:
        if not isinstance(task, dict):
            continue

        data_block = task.get("data", {})
        paper_id = str(data_block.get("arxiv_id", "")).strip()

        if not paper_id:
            continue

        predictions = task.get("predictions", [])
        if not predictions:
            continue

        lm_data[paper_id] = _extract_spans(predictions[0].get("result", []))

    return lm_data


def load_label_studio_annotations(file_path: str) -> dict:
    """
    Parses Label Studio JSON files for NER tasks (human gold annotations).

    Args:
        file_path (str): Path to the Label Studio annotation JSON file.

    Returns:
        dict: A dictionary mapping 'arxiv_id' -> list of span dicts (see `_extract_spans`).
    """

    ls_data = {}
    file_pattern = Path(file_path)

    for filepath in glob.glob(str(file_pattern)):
        with open(filepath, "r", encoding="utf-8") as f:
            file_data = json.load(f)

            # Ensure file_data is iterable. If it loaded a single root object, wrap it.
            if isinstance(file_data, dict):
                file_data = [file_data]

            for item in file_data:
                # If the item itself is a list (nested export), process its internal tasks.
                tasks_to_process = item if isinstance(item, list) else [item]

                for task in tasks_to_process:
                    if not isinstance(task, dict):
                        continue

                    data_block = task.get("data", {})
                    paper_id = str(data_block.get("arxiv_id", "")).strip()

                    if not paper_id:
                        continue

                    annotations = task.get("annotations", [])
                    if not annotations:
                        continue

                    ls_data[paper_id] = _extract_spans(annotations[0].get("result", []))

    return ls_data


def calculate_metrics(reference_vec: np.ndarray, comparison_vec: np.ndarray) -> dict:
    """
    Calculates Percent Agreement, Cohen's Kappa, Krippendorff's Alpha and standard NER
    metrics on two aligned binary vectors.

    The agreement metrics are symmetric, whereas Precision/Recall/F1 are asymmetric:
    `reference_vec` is treated as the reference (y_true) and `comparison_vec` as the
    prediction (y_pred).

    Args:
        reference_vec (np.ndarray): Binary vector for the reference annotator.
        comparison_vec (np.ndarray): Binary vector for the comparison annotator.

    Returns:
        dict: A dictionary containing all computed metrics.
    """

    if len(reference_vec) == 0:
        return {
            "percent_agreement": 0.0,
            "cohen_kappa": 0.0,
            "krippendorff_alpha": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    percent_agreement = float(np.sum(reference_vec == comparison_vec) / len(reference_vec))
    cohen_kappa = cohen_kappa_score(reference_vec, comparison_vec)

    # Stack into shape (num_annotators, num_samples).
    reliability_data = np.vstack((reference_vec, comparison_vec))
    try:
        # Use nominal measurement level for binary inclusion data.
        k_alpha = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement="nominal")
    except ValueError:
        # Failsafe for zero variance (e.g. annotators agreed on everything, highly unlikely in NER).
        k_alpha = 1.0 if np.array_equal(reference_vec, comparison_vec) else 0.0

    # Standard NER Metrics (treating reference_vec as y_true, comparison_vec as y_pred).
    return {
        "percent_agreement": percent_agreement,
        "cohen_kappa": cohen_kappa,
        "krippendorff_alpha": k_alpha,
        "precision": precision_score(reference_vec, comparison_vec, zero_division=0),
        "recall": recall_score(reference_vec, comparison_vec, zero_division=0),
        "f1": f1_score(reference_vec, comparison_vec, zero_division=0),
    }


# pylint: disable=too-many-locals


def calculate_overlap_metrics(reference_data: dict, comparison_data: dict) -> dict:
    """
    Computes relaxed (partial-overlap) Precision/Recall/F1 over the papers shared by both
    annotators. A comparison span is a true positive when it shares its label with a reference
    span and their character ranges overlap, so span-boundary disagreements no longer count as
    misses. Matching is greedy and 1-to-1 within each paper, so neither side double-counts.

    `reference_data` is treated as the gold (y_true) and `comparison_data` as the prediction
    (y_pred), mirroring the asymmetry of `calculate_metrics`.

    Args:
        reference_data (dict): Spans from the reference annotator (arxiv_id -> list of spans).
        comparison_data (dict): Spans from the comparison annotator (arxiv_id -> list of spans).

    Returns:
        dict: {"precision", "recall", "f1", "tp", "fp", "fn"} aggregated across shared papers.
    """

    common_ids = set(reference_data.keys()).intersection(set(comparison_data.keys()))

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for paper_id in common_ids:
        reference_spans = reference_data[paper_id]
        comparison_spans = comparison_data[paper_id]

        matched_refs: set = set()
        for comp_span in comparison_spans:
            # Greedily claim the first not-yet-matched reference span this comparison span overlaps.
            match_idx = next(
                (
                    idx
                    for idx, ref_span in enumerate(reference_spans)
                    if idx not in matched_refs and _spans_overlap(comp_span, ref_span)
                ),
                None,
            )
            if match_idx is None:
                false_positives += 1
            else:
                true_positives += 1
                matched_refs.add(match_idx)

        # Reference spans that no comparison span overlapped are misses.
        false_negatives += len(reference_spans) - len(matched_refs)

    predicted = true_positives + false_positives
    actual = true_positives + false_negatives
    precision = true_positives / predicted if predicted else 0.0
    recall = true_positives / actual if actual else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
    }


# pylint: enable=too-many-locals


def report_comparison(reference_label: str, comparison_label: str, reference_data: dict, comparison_data: dict) -> None:
    """
    Aligns two annotators, computes their agreement metrics and prints a formatted report.

    Args:
        reference_label (str): Name of the reference annotator (e.g. "Human").
        comparison_label (str): Name of the comparison annotator (e.g. "Phi").
        reference_data (dict): Annotations from the reference annotator.
        comparison_data (dict): Annotations from the comparison annotator.
    """

    reference_vec, comparison_vec, common_ids = align_annotations(reference_data, comparison_data)

    logger.info("=" * 60)
    logger.info("NER Inter-Annotator Agreement (%s vs %s)", reference_label, comparison_label)
    logger.info("=" * 60)

    if not common_ids:
        logger.info("No matching papers found between the two datasets.")
        logger.info("=" * 60)
        return

    if len(reference_vec) == 0:
        logger.info("Papers matched, but no entities were found in either dataset.")
        logger.info("=" * 60)
        return

    metrics = calculate_metrics(reference_vec, comparison_vec)

    logger.info("Papers aligned       : %d", len(common_ids))
    logger.info("Unique entities      : %d", len(reference_vec))
    logger.info("-" * 60)
    logger.info("Percent Agreement    : %.3f", metrics["percent_agreement"])
    logger.info("Cohen's Kappa        : %.3f", metrics["cohen_kappa"])
    logger.info("Krippendorff's Alpha : %.3f", metrics["krippendorff_alpha"])
    logger.info("-" * 60)
    logger.info("Exact matching (text + label):")
    logger.info(
        "Precision            : %.3f (%s predictions vs %s)",
        metrics["precision"],
        comparison_label,
        reference_label,
    )
    logger.info(
        "Recall               : %.3f (Did %s find %s entities?)",
        metrics["recall"],
        comparison_label,
        reference_label,
    )
    logger.info("F1-Score             : %.3f (Overall Balance)", metrics["f1"])

    # Relaxed matching credits span-boundary disagreements (e.g. "WikiText-2 dataset" vs
    # "WikiText-2"), which exact string equality penalises as a full miss.
    overlap = calculate_overlap_metrics(reference_data, comparison_data)
    logger.info("-" * 60)
    logger.info("Relaxed matching (label + character overlap):")
    logger.info(
        "Precision            : %.3f (%s predictions vs %s)",
        overlap["precision"],
        comparison_label,
        reference_label,
    )
    logger.info(
        "Recall               : %.3f (Did %s find %s entities?)",
        overlap["recall"],
        comparison_label,
        reference_label,
    )
    logger.info(
        "F1-Score             : %.3f (tp=%d fp=%d fn=%d)",
        overlap["f1"],
        overlap["tp"],
        overlap["fp"],
        overlap["fn"],
    )
    logger.info("=" * 60)


# NOTE: This script should rather support any annotation files as opposed to hardcoding the Phi / Qwen paths.
# Right now this is low priority.
def main() -> None:
    """
    Entry point: loads the human, Phi and Qwen annotations and reports pairwise agreement.
    """

    current_dir = os.path.dirname(__file__)
    human_annotations_file_path = resolve_path(current_dir, config.HUMAN_ANNOTATION_FILE)
    phi_annotations_file_path = resolve_path(current_dir, config.PHI_NER_ANNOTATION_FILE)
    qwen_annotations_file_path = resolve_path(current_dir, config.QWEN_NER_ANNOTATION_FILE)

    logger.debug("Loading Label Studio (human) NER annotations...")
    human_annotations = load_label_studio_annotations(human_annotations_file_path)

    logger.debug("Loading Phi NER predictions...")
    phi_annotations = load_lm_predictions(phi_annotations_file_path)

    logger.debug("Loading Qwen NER predictions...")
    qwen_annotations = load_lm_predictions(qwen_annotations_file_path)

    # The few-shot abstracts are shown to Phi/Qwen as worked examples, so the models would be graded
    # on inputs whose gold answers they saw — inflating agreement. Dropping them.
    if FEW_SHOT_ARXIV_IDS:
        logger.info("Excluding %d few-shot example(s) from the eval: %s", len(FEW_SHOT_ARXIV_IDS), FEW_SHOT_ARXIV_IDS)
        for dataset in (human_annotations, phi_annotations, qwen_annotations):
            for arxiv_id in FEW_SHOT_ARXIV_IDS:
                dataset.pop(arxiv_id, None)

    report_comparison("Human", "Phi", human_annotations, phi_annotations)
    report_comparison("Human", "Qwen", human_annotations, qwen_annotations)
    report_comparison("Qwen", "Phi", qwen_annotations, phi_annotations)


if __name__ == "__main__":
    main()
