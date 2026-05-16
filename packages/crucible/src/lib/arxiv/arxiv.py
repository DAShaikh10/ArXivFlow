"""
@Author: DAShaikh10
@Description: ArXiv API client & utility class to download & save research paper metadata based on specified filters.
"""

import asyncio
import json
import os
import xml.etree.ElementTree as ET

import aiofiles
import aiohttp
from aiolimiter import AsyncLimiter
from tenacity import RetryCallState, retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from tqdm.asyncio import tqdm

import wandb

from src.utils import logger, resolve_path

from . import config
from .schema import ArXivMetadata


class ArXiv:
    """
    ArXiv API client for fetching research paper metadata based on specified `category` and `date range` filters.
    Includes internal methods for parsing XML responses and saving results to a JSONL file.
    """

    def __init__(self) -> None:
        # ArXiv API is strictly limited to 1 request per config.RETRY_DELAY seconds (typically 3s).
        self.rate_limiter = AsyncLimiter(max_rate=config.CONCURRENCY, time_period=config.RETRY_DELAY)

    @staticmethod
    def _return_empty_list(retry_state: RetryCallState) -> list[ArXivMetadata]:
        """
        Fallback if all Tenacity retries fail.
        """

        logger.error("fetch_metadata - Exhausted all retries. Last error: %s", retry_state.outcome.exception())
        return []

    def _add_category_filter(self, query: str) -> str:
        """
        Append the category filter to the search query if it's not already present.
        Currently uses the **'AND'** operator to combine the category filter with existing filters,
        which is the most common use case for query scope restriction.

        Args:
            query (str): The original search query.

        Returns:
            str: The modified search query with the category filter added if it was not already present.

        Raises:
            ValueError: If the search query does not match expected patterns
                        and cannot be modified to include the date range filter.
        """

        # If the search query is empty, add the category filter.
        if query.endswith("search_query="):
            query += f"cat:{config.CATEGORY}"

            logger.info("_add_category_filter - Added search query with category filter: '%s'", query)
            wandb.log({"status": f"bulk_fetch - Added category filter: {config.CATEGORY}"})

            return query

        # If the search query already has some filters, add the category filter with 'AND'.
        # Currently, we only support adding the category filter as an 'AND' condition to ensure query scope restriction.
        # ArXiV API uses '+' for spaces and 'AND' for combining filters.
        # Ref.: https://info.arxiv.org/help/api/user-manual.html#query_details
        if query.find("search_query=") != -1:
            query += f"+AND+cat:{config.CATEGORY}"

            logger.info("_add_category_filter - Updated search query with category filter: '%s'", query)
            wandb.log({"status": f"bulk_fetch - Updated search query with category filter: {config.CATEGORY}"})

            return query

        raise ValueError("Search query does not match expected patterns. Cannot add category filter.")

    def _add_date_range_filter(self, query: str) -> str:
        """
        Append the date range filter to the search query if it's not already present.
        Currently uses the **'AND'** operator to combine the date range filter with existing filters,
        which is the most common use case for query scope restriction.

        Args:
            query (str): The original search query.

        Returns:
            str: The modified search query with the date range filter added if it was not already present.

        Raises:
            ValueError: If the search query does not match expected patterns
                        and cannot be modified to include the date range filter.
        """

        # If the search query is empty, add the date range filter.
        if query.endswith("search_query="):
            query += f"submittedDate:[{config.START_DATE}+TO+{config.END_DATE}]"

            logger.info("_add_date_range_filter - Added search query with date range filter: '%s'", query)
            wandb.log({"status": f"bulk_fetch - Added date range filter: {config.START_DATE} to {config.END_DATE}"})

            return query

        # If the search query already has some filters, add the date range filter with 'AND'.
        # Currently, we only support adding the date range filter as 'AND' condition to ensure query scope restriction.
        # ArXiV API uses '+' for spaces and 'AND' for combining filters.
        # Ref.: https://info.arxiv.org/help/api/user-manual.html#query_details
        if query.find("search_query=") != -1:
            query += f"+AND+submittedDate:[{config.START_DATE}+TO+{config.END_DATE}]"

            logger.info("_add_date_range_filter - Updated search query with date range filter: '%s'", query)
            wandb.log(
                {
                    "status": "bulk_fetch - Updated search query with date range filter: "
                    + f"{config.START_DATE} to {config.END_DATE}"
                }
            )

            return query

        raise ValueError("Search query does not match expected patterns. Cannot add date range filter.")

    # pylint: disable=too-many-locals
    def _parse_metadata_response(self, text: str) -> list[ArXivMetadata]:
        """
        ArXiv returns Atom XML response. Parse Atom XML response and extract entries.

        Args:
            text (str): The raw XML response as a string.

        Returns:
            list[ArXivMetadata]: A list of the selected metadata dictionary for each paper found in the response.
        """

        try:
            root = ET.fromstring(text)
        except ET.ParseError as exception:
            logger.error("_parse_metadata_response - Failed to parse XML response: %s", exception)
            wandb.log({"status": f"_parse_metadata_response - Failed to parse XML response: {exception}"})
            return []

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries: list[ArXivMetadata] = []

        for entry in root.findall("atom:entry", ns):
            arxiv_id_elem = entry.find("atom:id", ns)
            arxiv_id_full = arxiv_id_elem.text if arxiv_id_elem is not None else ""
            arxiv_id = arxiv_id_full.split("/abs/")[-1]

            title_elem = entry.find("atom:title", ns)
            title = title_elem.text.strip() if title_elem is not None else ""

            summary_elem = entry.find("atom:summary", ns)
            abstract = summary_elem.text.strip() if summary_elem is not None else ""

            published_date_elem = entry.find("atom:published", ns)
            published_date = published_date_elem.text.strip() if published_date_elem is not None else ""

            entries.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "abstract": abstract,
                    "influential_citations": 0,
                    "published_date": published_date,
                    "references": [],
                    "url": arxiv_id_full,
                }
            )

        return entries

    # pylint: enable=too-many-locals

    @retry(
        stop=stop_after_attempt(config.MAX_RETRIES),
        wait=wait_fixed(config.RETRY_DELAY),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        retry_error_callback=_return_empty_list,
    )
    async def fetch_metadata(self, session: aiohttp.ClientSession, url: str) -> list[ArXivMetadata]:
        """
        Fetch and parse Atom XML response with exponential backoff on failure.

        Args:
            session (aiohttp.ClientSession): The HTTP session to use for making the request.
            url (str): The URL to fetch the metadata from.

        Returns:
            list[ArXivMetadata]: A list of dictionaries containing the metadata for each paper found in the response.

        Raises:
            aiohttp.ClientError: If the HTTP request fails.
        """

        logger.debug("fetch_metadata - START")
        wandb.log({"status": "fetch_metadata - START"})

        try:
            async with self.rate_limiter:
                async with session.get(url) as response:
                    response.raise_for_status()
                    text = await response.text()

                    return self._parse_metadata_response(text)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exception:
            logger.error("fetch_metadata - Request failed: %s. Retrying...", exception)
            wandb.log({"status": f"fetch_metadata - Request failed: {exception}. Retrying..."})
            raise
        finally:
            logger.debug("fetch_metadata - END")
            wandb.log({"status": "fetch_metadata - END"})

    async def bulk_fetch_metadata(self) -> None:
        """
        Bulk fetch and save research paper metadata from [ArXiv](https://arxiv.org/) asynchronously.

        Search is performed based on specified `category` **and** `date` range filters _(Read from the `crucible/.env`)_

        The results are saved to a JSONL file to support streaming large datasets. _(`ArXiv/data/raw_dataset.jsonl`)_

        Raises:
            ValueError: If the search query does not match expected patterns
                        and cannot be modified to include the category or date range filters.
        """

        wandb.log({"status": "bulk_fetch - START"})
        logger.debug("bulk_fetch_metadata - START")

        try:
            # Build the base URL with filters based on the provided configuration.
            url = f"{config.API_BASE_URL}?search_query="
            if config.CATEGORY:
                logger.debug("Adding category filter: %s", config.CATEGORY)
                wandb.log({"status": f"bulk_fetch - Adding category filter: {config.CATEGORY}"})

                url = self._add_category_filter(url)
            if config.START_DATE and config.END_DATE:
                logger.debug("Adding date range filter: %s to %s", config.START_DATE, config.END_DATE)
                wandb.log(
                    {"status": f"bulk_fetch - Adding date range filter: {config.START_DATE} to {config.END_DATE}"}
                )

                url = self._add_date_range_filter(url)

            current_dir = os.path.dirname(__file__)
            dataset_path = resolve_path(current_dir, config.RAW_DATASET_FILE)
            total_records = 0

            # https://export.arxiv.org/api/query?search_query=au:del_maestro+AND+submittedDate:[...
            async with aiohttp.ClientSession() as session:
                tasks = []
                for idx in range(0, config.TOTAL_RESULTS, config.BATCH_SIZE):  # ArXiv API is 0-indexed.
                    tasks.append(
                        asyncio.create_task(
                            self.fetch_metadata(session, url + f"&start={idx}&max_results={config.BATCH_SIZE}")
                        )
                    )

                async with aiofiles.open(dataset_path, "w", encoding="utf-8") as f:
                    with tqdm(total=len(tasks), desc="Fetching ArXiv Metadata") as pbar:
                        for coro in asyncio.as_completed(tasks):
                            data = await coro
                            for item in data:
                                await f.write(json.dumps(item) + "\n")
                            total_records += len(data)
                            pbar.update(1)

            # Save output dataset as a WandB Artifact.
            artifact = wandb.Artifact("arxiv_raw_dataset", type="dataset")
            artifact.add_file(dataset_path)
            wandb.log_artifact(artifact)

            logger.info("Successfully saved %s records to %s", total_records, dataset_path)
            wandb.log({"status": f"Successfully saved {total_records} records to {dataset_path}"})

        except (ValueError, OSError) as exception:
            logger.error("bulk_fetch_metadata - Failed with exception: %s", exception)
            wandb.log({"status": f"bulk_fetch - Failed with exception: {exception}"})
        finally:
            wandb.log({"status": "bulk_fetch - END"})
            logger.debug("bulk_fetch_metadata - END")
