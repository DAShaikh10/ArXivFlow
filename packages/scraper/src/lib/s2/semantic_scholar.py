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
from tenacity import RetryCallState, retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from tqdm.asyncio import tqdm

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

    def _return_failed_paper(self, retry_state: RetryCallState) -> Tuple[Optional[dict], bool]:
        """
        Fallback if all Tenacity retries fail.
        """

        logger.error("_fetch_paper_data - Exhausted all retries. Last error: %s", retry_state.outcome.exception())
        return None, True

    @retry(
        stop=stop_after_attempt(config.MAX_RETRIES),
        wait=wait_fixed(config.RETRY_DELAY),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        retry_error_callback=_return_failed_paper,
    )
    async def _fetch_paper_data(self, session: aiohttp.ClientSession, arxiv_url: str) -> Tuple[Optional[dict], bool]:
        """
        Fetch metadata for a single paper by ArXiv URL from Semantic Scholar API.

        Args:
            session (aiohttp.ClientSession): The aiohttp session for making requests.
            arxiv_url (str): The ArXiv URL of the paper to fetch.

        Returns:
            Tuple[Optional[dict], bool]: A tuple containing the paper's metadata if found (else None),
                                         and a boolean indicating if it is a retryable failure.
        """

        url = f"{config.API_BASE_URL}/url:{arxiv_url}?fields={config.API_FIELDS}"

        async with session.get(url, headers=self.headers) as response:
            if response.status == HTTPStatus.NOT_FOUND:
                logger.warning("_fetch_paper_data - Paper %s not found on Semantic Scholar.", arxiv_url)
                return None, False

            response.raise_for_status()
            return await response.json(), False

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
        self, session: aiohttp.ClientSession, batch: List[ArXivMetadata], concurrency: int = 3
    ) -> Tuple[List[ArXivMetadata], List[ArXivMetadata]]:
        """
        Enrich a batch of ArXiv metadata entries using the provided aiohttp session.

        Returns a tuple of (enriched_batch, retryable_failed_batch).
        """

        logger.debug("enrich_batch - START")

        async def process_paper(paper: ArXivMetadata) -> Tuple[ArXivMetadata, bool]:
            arxiv_url = paper.get("url", "")
            if not arxiv_url:
                return paper, False

            s2_data, is_retryable = await self._fetch_paper_data(session, arxiv_url)
            if s2_data:
                paper["influential_citations"] = s2_data.get("influentialCitationCount")
                paper["references"] = self._parse_references(s2_data.get("references"))

            return paper, is_retryable

        semaphore = asyncio.Semaphore(concurrency)

        async def bound_process(paper: ArXivMetadata) -> Tuple[ArXivMetadata, bool]:
            async with semaphore:
                return await process_paper(paper)

        tasks = [bound_process(paper) for paper in batch]
        results = await asyncio.gather(*tasks)

        enriched_batch = [paper for paper, is_retryable in results if not is_retryable]
        failed_batch = [paper for paper, is_retryable in results if is_retryable]

        logger.debug("enrich_batch - END")

        return enriched_batch, failed_batch

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    async def enrich_dataset(
        self,
        dataset_path: str,
        out_path: str,
        batch_size: int = 100,
        concurrency: int = 3,
        delay_between_batches: float = 0.0,
    ) -> None:
        """
        Stream `dataset_path` (JSONL), enrich in batches and write results to `out_path`.
        Failed papers will be logged as errors.

        Args:
            dataset_path (str): Path to input JSONL file (one JSON object per line).
            out_path (str): Path to write enriched JSONL results.
            batch_size (int): Number of records to process per batch.
            concurrency (int): Max concurrent requests when enriching a batch.
            delay_between_batches (float): Optional delay in second(s) between batches to ease rate limits.
        """

        logger.debug("enrich_dataset - START")

        # Clear out existing output files if they exist (to overwrite rather than append)
        if os.path.exists(out_path):
            os.remove(out_path)

        async with aiohttp.ClientSession() as session:
            with tqdm(desc="Enriching Papers") as pbar:
                batch: List[ArXivMetadata] = []
                async for paper in stream_jsonl(dataset_path):
                    batch.append(paper)

                    if len(batch) >= batch_size:
                        enriched, failed = await self.enrich_batch(session, batch, concurrency)
                        await write_jsonl_batch(out_path, enriched, append=True)
                        for failed_paper in failed:
                            logger.error(
                                "enrich_dataset - Failed to enrich paper with URL: %s", failed_paper.get("url", "N/A")
                            )

                        pbar.update(len(batch))
                        batch = []
                        if delay_between_batches:
                            await asyncio.sleep(delay_between_batches)

                # Final partial batch.
                if batch:
                    enriched, failed = await self.enrich_batch(session, batch, concurrency)
                    await write_jsonl_batch(out_path, enriched, append=True)
                    for failed_paper in failed:
                        logger.error(
                            "enrich_dataset - Failed to enrich paper with URL: %s", failed_paper.get("url", "N/A")
                        )
                    pbar.update(len(batch))

        logger.debug("enrich_dataset - END")

    # pylint: enable=too-many-arguments, too-many-positional-arguments
