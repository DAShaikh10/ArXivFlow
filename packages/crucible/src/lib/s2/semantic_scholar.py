"""
@Author: DAShaikh10
@Description: Semantic Scholar API client & utility class to enrich ArXiv research paper metadata
              with influential citation count and references.
              Ref.: https://www.semanticscholar.org/faq/influential-citations
"""

import asyncio
import os
from http import HTTPStatus
from typing import List, Optional, Tuple

import aiohttp
from aiolimiter import AsyncLimiter
from tenacity import RetryCallState, retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from tqdm.asyncio import tqdm

import wandb

from src.lib.arxiv.schema import ArXivMetadata, Reference
from src.utils import logger, stream_jsonl, write_jsonl_batch

from . import config


class SemanticScholar:
    """
    Semantic Scholar API client for fetching influential citation count and references from Semantic Scholar API.
    Enriches existing ArXiv data with citation and reference information in batches while respecting API rate limits.
    """

    def __init__(self) -> None:
        self.headers = {"x-api-key": config.S2_API_KEY} if config.S2_API_KEY else {}
        self.rate_limiter = AsyncLimiter(max_rate=config.CONCURRENCY, time_period=config.DELAY_BETWEEN_BATCHES)

    @staticmethod
    def _return_failed_batch(retry_state: RetryCallState) -> Tuple[Optional[list], bool]:
        """
        Fallback if all Tenacity retries fail.
        """

        logger.error("_fetch_batch_data - Exhausted all retries. Last error: %s", retry_state.outcome.exception())
        return None, True

    @retry(
        stop=stop_after_attempt(config.MAX_RETRIES),
        wait=wait_fixed(config.RETRY_DELAY),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        retry_error_callback=_return_failed_batch,
    )
    async def _fetch_batch_data(
        self, session: aiohttp.ClientSession, arxiv_urls: List[str]
    ) -> Tuple[Optional[list], bool]:
        """
        Fetch metadata for a batch of papers by ArXiv URLs from Semantic Scholar API.

        Args:
            session (aiohttp.ClientSession): The aiohttp session for making requests.
            arxiv_urls (List[str]): A list of ArXiv URLs of the papers to fetch.

        Returns:
            Tuple[Optional[list], bool]: A tuple containing the list of paper metadata if successful (else None),
                                         and a boolean indicating if it is a retryable failure.
        """

        url = f"{config.API_BASE_URL}/batch?fields={config.API_FIELDS}"
        payload = {"ids": [f"URL:{url}" for url in arxiv_urls]}

        try:
            async with self.rate_limiter:
                async with session.post(url, json=payload, headers=self.headers) as response:
                    if response.status == HTTPStatus.NOT_FOUND:
                        logger.warning("_fetch_batch_data - Some papers not found on Semantic Scholar.")
                        wandb.log({"warning": "Some papers not found on Semantic Scholar."})
                        return None, False

                    response.raise_for_status()
                    return await response.json(), False
        except (aiohttp.ClientError, asyncio.TimeoutError) as exception:
            logger.error("_fetch_batch_data - Request failed: %s. Retrying...", exception)
            wandb.log({"status": f"_fetch_batch_data - Request failed: {exception}. Retrying..."})
            raise

    def _parse_references(self, s2_references: list) -> List[Reference]:
        """
        Parse Semantic Scholar references into our schema.

        Args:
            s2_references (list): A list of raw reference dictionaries from Semantic Scholar.

        Returns:
            List[Reference]: A parsed list of references adhering to our Reference schema.
        """

        parsed_references: List[Reference] = []
        for reference in s2_references:
            arxiv_id = None
            if reference.get("externalIds") is not None:
                arxiv_id = reference.get("externalIds").get("ArXiv", None)

            title = reference.get("title")
            url = reference.get("url", None)

            parsed_references.append({"arxiv_id": arxiv_id, "title": title, "url": url})

        return parsed_references

    async def enrich_batch(
        self, session: aiohttp.ClientSession, batch: List[ArXivMetadata]
    ) -> Tuple[List[ArXivMetadata], List[ArXivMetadata]]:
        """
        Enrich a batch of ArXiv metadata entries using the provided aiohttp session.

        Args:
            session (aiohttp.ClientSession): The aiohttp session for making requests.
            batch (List[ArXivMetadata]): A list of ArXiv metadata entries to enrich.

        Returns:
            Tuple[List[ArXivMetadata], List[ArXivMetadata]]: A tuple containing a list of enriched metadata entries
                                                            and a list of entries that failed enrichment.
        """

        logger.debug("enrich_batch - START")
        wandb.log({"status": "enrich_batch - START"})

        enriched_batch = []
        failed_batch = []

        valid_papers = []
        arxiv_urls = []

        for paper in batch:
            url = paper.get("url")
            if url:
                valid_papers.append(paper)
                arxiv_urls.append(url)
            else:
                enriched_batch.append(paper)

        if arxiv_urls:
            s2_batch_data, is_retryable = await self._fetch_batch_data(session, arxiv_urls)

            if is_retryable:
                failed_batch.extend(valid_papers)
            elif s2_batch_data is not None:
                for idx, paper in enumerate(valid_papers):
                    s2_data = s2_batch_data[idx] if idx < len(s2_batch_data) else None
                    if s2_data:
                        paper["influential_citations"] = s2_data.get("influentialCitationCount")
                        references = s2_data.get("references")
                        if references:
                            paper["references"] = self._parse_references(references)
                    enriched_batch.append(paper)
            else:
                enriched_batch.extend(valid_papers)

        logger.debug("enrich_batch - END")
        wandb.log({"status": "enrich_batch - END"})

        return enriched_batch, failed_batch

    async def enrich_dataset(self, dataset_path: str, out_path: str) -> None:
        """
        Stream `dataset_path` (JSONL), enrich in batches and write results to `out_path`
        using a Producer-Consumer pipeline to decouple I/O and network requests.
        Failed papers will be logged as errors.

        Args:
            dataset_path (str): Path to input JSONL file (one JSON object per line).
            out_path (str): Path to write enriched JSONL results.
        """

        logger.debug("enrich_dataset - START")
        wandb.log({"status": "enrich_dataset - START"})

        # Clear out existing output files if they exist (to overwrite rather than append)
        if os.path.exists(out_path):
            os.remove(out_path)

        logger.debug("Using concurrency level: %s", config.CONCURRENCY)
        wandb.log({"status": f"enrich_batch - Using concurrency level: {config.CONCURRENCY}"})
        async with aiohttp.ClientSession() as session:
            # We use an asyncio queue to hold batches of papers to fetch.
            fetch_queue: asyncio.Queue = asyncio.Queue()

            with tqdm(desc="Enriching Papers") as pbar:

                async def worker():
                    while True:
                        batch = await fetch_queue.get()
                        if batch is None:
                            fetch_queue.task_done()
                            break

                        enriched, failed = await self.enrich_batch(session, batch)
                        await write_jsonl_batch(out_path, enriched, append=True)
                        for failed_paper in failed:
                            logger.error(
                                "enrich_dataset - Failed to enrich paper with URL: %s", failed_paper.get("url", "N/A")
                            )
                            wandb.log({"error": f"Failed to enrich paper with URL: {failed_paper.get('url', 'N/A')}"})

                        pbar.update(len(batch))
                        fetch_queue.task_done()

                # Start the workers. They will run until they receive a `None` batch, which signals them to exit.
                # We use config.CONCURRENCY to control how many concurrent workers we have making API requests.
                workers = [asyncio.create_task(worker()) for _ in range(config.CONCURRENCY)]

                # Producer: Read JSONL and put batches into the queue.
                batch: List[ArXivMetadata] = []
                async for paper in stream_jsonl(dataset_path):
                    batch.append(paper)

                    if len(batch) >= config.BATCH_SIZE:
                        await fetch_queue.put(batch)
                        batch = []

                # Final partial batch.
                if batch:
                    await fetch_queue.put(batch)

                # Signal the workers to exit.
                for _ in range(config.CONCURRENCY):
                    await fetch_queue.put(None)

                # Wait for all tasks to be processed.
                await fetch_queue.join()
                await asyncio.gather(*workers)

        logger.debug("enrich_dataset - END")
        wandb.log({"status": "enrich_dataset - END"})
