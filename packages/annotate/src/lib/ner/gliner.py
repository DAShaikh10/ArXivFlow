"""
Perform Named Entity Recognition (NER) on paper abstracts using the GLiNER model.

`@author`: DAShaikh10
"""

import asyncio
import json
import os
from typing import List

import aiofiles
import torch
from gliner import GLiNER
from tqdm import tqdm

import wandb
from src.utils import logger, resolve_path

from . import config
from .schema import NER_LABELS, Annotation, Paper


async def read_abstracts(encoding: str = "utf-8") -> List[Paper]:
    """
    Reads abstracts from a JSONL file and returns a list of dictionaries containing the paper ID and abstract.

    Args:
        encoding: The encoding to use for reading the file.

    Returns:
        List of dictionaries with keys 'id' and 'abstract'.

    Raises:
        FileNotFoundError: If dataset file does not exist.
        ValueError: If JSON parsing fails or required fields are missing.
        asyncio.TimeoutError: If file read exceeds timeout.
    """

    logger.debug("read_abstracts - START")
    wandb.log({"status": "read_abstracts - START"})

    current_dir: str = os.path.dirname(__file__)
    dataset_path: str = resolve_path(current_dir, config.CLEANED_DATASET_FILE)

    if not os.path.exists(dataset_path):
        logger.error("Dataset file not found at %s", dataset_path)
        wandb.log({"error": f"Dataset file not found at {dataset_path}"})
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

    papers: List[Paper] = []
    async with aiofiles.open(dataset_path, "r", encoding=encoding) as in_file:
        async for line in in_file:
            if not line.strip():
                continue
            try:
                paper = json.loads(line)
                papers.append({"id": paper["arxiv_id"], "abstract": paper["abstract"]})
            except json.JSONDecodeError as exception:
                logger.error("Failed to parse JSON line: %s", str(exception))
                wandb.log({"error": f"Failed to parse JSON line: {str(exception)}"})
                continue

    logger.info("Successfully read %d abstracts from dataset", len(papers))
    wandb.log({"status": "papers_loaded", "count": len(papers)})

    logger.debug("read_abstracts - END")
    wandb.log({"status": "read_abstracts - END"})

    return papers


async def save_annotated_dataset(dataset: List[Annotation], encoding: str = "utf-8") -> None:
    """
    Saves the annotated dataset to a JSONL file.

    Args:
        dataset: List of dictionaries containing the annotated data.
        encoding: The encoding to use for the output file.

    Raises:
        IOError: If file write operation fails.
        ValueError: If dataset is empty or invalid.
    """

    if not dataset:
        error_msg = "Cannot save empty dataset"
        logger.warning(error_msg)
        wandb.log({"warning": error_msg})
        return

    logger.debug("save_annotated_dataset - START")
    wandb.log({"status": "save_annotated_dataset - START"})

    current_dir: str = os.path.dirname(__file__)
    dataset_path: str = resolve_path(current_dir, config.GLINER_NER_ANNOTATION_FILE)

    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

        async with aiofiles.open(dataset_path, "w", encoding=encoding) as f:
            for annotation in dataset:
                try:
                    await f.write(json.dumps(annotation) + "\n")
                except (TypeError, ValueError, json.JSONDecodeError) as exception:
                    logger.error(
                        "Failed to serialize annotation %s: %s", annotation.get("id", "UNKNOWN"), str(exception)
                    )
                    wandb.log(
                        {"error": f"Failed to serialize annotation {annotation.get('id', 'UNKNOWN')}: {str(exception)}"}
                    )
                    continue

        # Save artifacts to Weights & Biases for future reference and reproducibility
        artifact = wandb.Artifact("gliner-ner-annotations", type="dataset")
        artifact.add_file(dataset_path)
        wandb.log_artifact(artifact)

        logger.info("Processing complete! Labels saved to '%s'.", dataset_path)
        wandb.log(
            {"status": "annotated_dataset_saved", "dataset_path": dataset_path, "annotations_count": len(dataset)}
        )
    except IOError as exception:
        logger.error("Failed to write annotation file: %s", str(exception))
        wandb.log({"error": f"Failed to write annotation file: {str(exception)}"})
        raise
    except Exception as exception:
        error_msg = f"Unexpected error saving annotations: {str(exception)}"
        logger.error(error_msg)
        wandb.log({"error": error_msg})
        raise
    finally:
        logger.debug("save_annotated_dataset - END")
        wandb.log({"status": "save_annotated_dataset - END"})


def cleanup_gpu_memory(gliner: GLiNER, dev: str) -> None:
    """
    Clean up GPU memory after processing.

    Args:
        gliner: The GLiNER model instance to delete.
        dev: The device type (e.g., "cuda") to check before cleanup.
    """

    try:
        if dev == "cuda":
            del gliner
            torch.cuda.empty_cache()
            logger.debug("GPU memory cleaned up")

    # pylint: disable=broad-except
    except Exception as exception:
        logger.error("Failed to cleanup GPU memory: %s", str(exception))
        wandb.log({"error": f"Failed to cleanup GPU memory: {str(exception)}"})
    # pylint: enable=broad-except


if __name__ == "__main__":
    model = None
    device = None

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", device.upper())

        logger.debug("Loading GLiNER model: %s", config.GLINER_MODEL)
        model = GLiNER.from_pretrained(config.GLINER_MODEL).to(device)
        logger.debug("Model loaded successfully")

        # GLiNER works best with title-case, natural language descriptions.
        labels: List[str] = NER_LABELS

        # Setup Weights & Biases for experiment tracking.
        wandb.init(
            project=config.WANDB_PROJECT_NAME,
            name="gliner-ner-extraction",
            config={
                "device": device,
                "labels": labels,
                "model": config.GLINER_MODEL,
                "sensitivity_threshold": config.SENSITIVITY_THRESHOLD,
            },
            job_type="ner",
        )

        logger.info("Reading abstracts from dataset...")
        abstracts = asyncio.run(read_abstracts())

        if not abstracts:
            logger.warning("No abstracts found in dataset")
            wandb.log({"warning": "No abstracts found"})
        else:
            logger.info("Starting extraction for %d abstracts...", len(abstracts))
            wandb.log({"num_abstracts": len(abstracts)})

            annotated_dataset: List[Annotation] = []
            failed_count = 0

            for item in tqdm(abstracts, desc="Processing abstracts"):
                try:
                    # GLiNER extracts matching text spans, labels, and character index anchors.
                    entities = model.predict_entities(
                        item["abstract"],
                        labels,
                        threshold=config.SENSITIVITY_THRESHOLD,
                        return_class_probs=True,
                    )

                    formatted_entities = []
                    for entity in entities:
                        formatted_entities.append(
                            {
                                "text": entity["text"],
                                "label": entity["label"],
                                "start": entity["start"],
                                "end": entity["end"],
                            }
                        )

                    annotated_dataset.append({"id": item["id"], "entities": formatted_entities})

                # pylint: disable=broad-except
                except Exception as exception:
                    logger.warning("Failed to process paper %s: %s", item.get("id", "UNKNOWN"), str(exception))
                    failed_count += 1
                    continue
                # pylint: enable=broad-except

            logger.info("Processed %d abstracts successfully (%d failed)", len(annotated_dataset), failed_count)
            wandb.log({"processed_count": len(annotated_dataset), "failed_count": failed_count})

            if annotated_dataset:
                logger.info("Saving annotated dataset...")
                asyncio.run(save_annotated_dataset(annotated_dataset))
            else:
                logger.warning("No annotations to save")
                wandb.log({"warning": "No annotations generated"})
    except KeyboardInterrupt:
        logger.error("Processing interrupted by user")
        wandb.log({"error": "Processing interrupted"})
    except Exception as exception:
        logger.error("Fatal error during processing: %s", str(exception))
        wandb.log({"error": f"Fatal error: {str(exception)}"})
        raise
    finally:
        if model is not None and device is not None:
            cleanup_gpu_memory(model, device)

        wandb.finish()
