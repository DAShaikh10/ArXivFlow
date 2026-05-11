"""
@Author: DAShaikh10
@Description: Main entry point for the ArXiv API scraper. Invoke to fetch metadata for research papers
              based on specified filters and save results to a JSON file.
"""

import asyncio

from .arxiv import ArXiv

if __name__ == "__main__":
    client = ArXiv()
    asyncio.run(client.bulk_fetch_metadata())
