"""
@Author: DAShaikh10
@Description: Main entry point for the Semantic Scholar API scraper. Invoke to fetch "more" metadata for research papers
              fetched from ArXiv API using Semanic Scholar API.
"""

import asyncio
import os

from src.utils import logger, resolve_path

from . import config
from .semantic_scholar import SemanticScholar


async def main() -> None:
    """
    Entry point that streams the JSONL dataset and enriches it in batches.
    """

    current_dir = os.path.dirname(__file__)
    dataset_path = resolve_path(current_dir, config.RAW_DATASET_FILE)

    if not os.path.exists(dataset_path):
        logger.error("Dataset file not found at %s", dataset_path)
        return

    out_path = resolve_path(current_dir, config.ENRICHED_DATASET_FILE)

    client = SemanticScholar()
    await client.enrich_dataset(
        dataset_path,
        out_path,
    )


if __name__ == "__main__":
    asyncio.run(main())
