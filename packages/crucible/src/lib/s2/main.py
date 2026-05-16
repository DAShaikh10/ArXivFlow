"""
@Author: DAShaikh10
@Description: Main entry point for the Semantic Scholar API scraper. Invoke to fetch "more" metadata for research papers
              fetched from ArXiv API using Semanic Scholar API.
"""

import asyncio
import os
import wandb

from src.utils import logger, resolve_path

from . import config
from .semantic_scholar import SemanticScholar


async def main() -> None:
    """
    Entry point that streams the JSONL dataset and enriches it in batches.
    """

    # Initialize WandB and log configuration.
    wandb.init(
        project=config.WANDB_PROJECT_NAME,
        job_type="scrape_s2_api",
        config={
            "BASE_URL": config.API_BASE_URL,
            "BATCH_SIZE": config.BATCH_SIZE,
            "CONCURRENCY": config.CONCURRENCY,
            "MAX_RETRIES": config.MAX_RETRIES,
            "RETRY_DELAY": config.RETRY_DELAY,
        },
    )

    current_dir = os.path.dirname(__file__)
    dataset_path = resolve_path(current_dir, config.RAW_DATASET_FILE)

    if not os.path.exists(dataset_path):
        logger.error("Dataset file not found at %s", dataset_path)
        wandb.log({"error": f"Dataset file not found at {dataset_path}"})
        wandb.finish()
        return

    out_path = resolve_path(current_dir, config.ENRICHED_DATASET_FILE)

    client = SemanticScholar()
    await client.enrich_dataset(
        dataset_path,
        out_path,
    )

    wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())
