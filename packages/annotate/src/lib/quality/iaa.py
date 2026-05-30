"""
Compute Inter-Annotator Agreement (IAA) for NER (Entity Matching).
Evaluates exact text and label matches across annotators, Human vs Phi, Human vs Qwen, Qwen vs Phi.

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

from src.utils import logger, resolve_path

from . import config


def load_label_studio_annotations(file_path: str) -> dict:
    """
    Parses Label Studio JSON files for NER tasks (human gold annotations).

    Args:
        file_path (str): Path to the Label Studio annotation JSON file.

    Returns:
        dict: A dictionary mapping 'arxiv_id' -> set of (text.lower(), label) tuples.
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

                    ls_data[paper_id] = _extract_entities(annotations[0].get("result", []))

    return ls_data


def load_lm_predictions(filepath: str) -> dict:
    """
    Parses LLM (Phi / Qwen) outputs stored in the Label Studio prediction format.

    The LMNER pipeline serialises its results as a single JSON array (despite the
    `.jsonl` extension), where each task carries its spans under `predictions[0].result`.

    Args:
        filepath (str): Path to the LLM annotation file (e.g. `phi-4-annotation.jsonl`).

    Returns:
        dict: A dictionary mapping 'arxiv_id' -> set of (text.lower(), label) tuples.
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

        lm_data[paper_id] = _extract_entities(predictions[0].get("result", []))

    return lm_data


def _extract_entities(results: list) -> set:
    """
    Extracts (text.lower(), label) tuples from a Label Studio `result` list. The format
    is shared between human annotations and LLM predictions, so both loaders reuse it.

    Args:
        results (list): A list of annotation/prediction results from Label Studio.

    Returns:
        set: A set of (text.lower(), label) tuples representing the extracted entities.
    """

    entities = set()

    for res in results:
        # Label Studio NER format stores data inside 'value'.
        val = res.get("value", {})
        text = val.get("text", "").strip().lower()
        labels = val.get("labels", [])

        if text and labels:
            # Assuming a single label per span.
            entities.add((text, labels[0]))

    return entities


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
        reference_ents = reference_data[paper_id]
        comparison_ents = comparison_data[paper_id]

        # Union of all entities found by EITHER annotator in this paper.
        all_unique_ents = reference_ents.union(comparison_ents)

        for ent in all_unique_ents:
            # Did each annotator extract this exact entity + label? (1 = Yes, 0 = No)
            reference_binary.append(1 if ent in reference_ents else 0)
            comparison_binary.append(1 if ent in comparison_ents else 0)

    return np.array(reference_binary), np.array(comparison_binary), list(common_ids)


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

    print("\n" + "=" * 60)
    print(f"NER Inter-Annotator Agreement ({reference_label} vs {comparison_label})")
    print("=" * 60)

    if not common_ids:
        print("No matching papers found between the two datasets.")
        print("=" * 60)
        return

    if len(reference_vec) == 0:
        print("Papers matched, but no entities were found in either dataset.")
        print("=" * 60)
        return

    metrics = calculate_metrics(reference_vec, comparison_vec)

    print(f"Papers aligned       : {len(common_ids)}")
    print(f"Unique entities      : {len(reference_vec)}")
    print("-" * 60)
    print(f"Percent Agreement    : {metrics['percent_agreement']:.3f}")
    print(f"Cohen's Kappa        : {metrics['cohen_kappa']:.3f}")
    print(f"Krippendorff's Alpha : {metrics['krippendorff_alpha']:.3f}")
    print("-" * 60)
    print(f"Precision            : {metrics['precision']:.3f} ({comparison_label} predictions vs {reference_label})")
    print(f"Recall               : {metrics['recall']:.3f} (Did {comparison_label} find {reference_label} entities?)")
    print(f"F1-Score             : {metrics['f1']:.3f} (Overall Balance)")
    print("=" * 60)


def main() -> None:
    """
    Entry point: loads the human, Phi and Qwen annotations and reports pairwise agreement.
    """

    current_dir = os.path.dirname(__file__)
    human_annotations_file_path = resolve_path(current_dir, "human-annotation.json")
    phi_annotations_file_path = resolve_path(current_dir, config.PHI_NER_ANNOTATION_FILE)
    qwen_annotations_file_path = resolve_path(current_dir, config.QWEN_NER_ANNOTATION_FILE)

    logger.debug("Loading Label Studio (human) NER annotations...")
    human_annotations = load_label_studio_annotations(human_annotations_file_path)

    logger.debug("Loading Phi NER predictions...")
    phi_annotations = load_lm_predictions(phi_annotations_file_path)

    logger.debug("Loading Qwen NER predictions...")
    qwen_annotations = load_lm_predictions(qwen_annotations_file_path)

    report_comparison("Human", "Phi", human_annotations, phi_annotations)
    report_comparison("Human", "Qwen", human_annotations, qwen_annotations)
    report_comparison("Qwen", "Phi", qwen_annotations, phi_annotations)


if __name__ == "__main__":
    main()
