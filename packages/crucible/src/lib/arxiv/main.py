"""
Main entry point for the ArXiv API scraper.
Invoke to fetch metadata for research papers based on specified filters and save results to a JSONL file.
Log the process and artifacts to both locally and on Weights & Biases for tracking and visualization.

`@author`: DAShaikh10
"""

import asyncio
import wandb

from . import config
from .arxiv import ArXiv

if __name__ == "__main__":
    # Initialize WandB and log configuration.
    wandb.init(
        project=config.WANDB_PROJECT_NAME,
        name="scrape_arxiv_api",
        config={
            "BASE_URL": config.API_BASE_URL,
            "BATCH_SIZE": config.BATCH_SIZE,
            "CATEGORY": config.CATEGORY,
            "CONCURRENCY": config.CONCURRENCY,
            "END_DATE": config.END_DATE,
            "MAX_RETRIES": config.MAX_RETRIES,
            "RETRY_DELAY": config.RETRY_DELAY,
            "START_DATE": config.START_DATE,
            "TOTAL_RESULTS": config.TOTAL_RESULTS,
        },
        job_type="scrape",
    )

    client = ArXiv()

    asyncio.run(client.bulk_fetch_metadata())

    wandb.finish()
