"""
@Author: DAShaikh10
@Description: Utility functions for robust File I/O operations, specifically handling JSONL.
"""

import json
import os
from typing import AsyncGenerator, List

import aiofiles

from . import logger


async def stream_jsonl(file_path: str, mode: str = "r", encoding: str = "utf-8") -> AsyncGenerator[dict, None]:
    """
    Stream a JSONL file line by line.

    Args:
        file_path (str): Path to the input JSONL file.
        mode (str): Mode to open the file.
        encoding (str): Encoding to use when opening the file.

    Yields:
        dict: The parsed JSON object for each line.
    """

    async with aiofiles.open(file_path, mode, encoding=encoding) as in_f:
        async for line in in_f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exception:
                logger.error("stream_jsonl - Failed to decode line: %s. Error: %s", line, exception)


async def write_jsonl_batch(file_path: str, batch: List[dict], append: bool = True, encoding: str = "utf-8") -> None:
    """
    Write or append a batch of dictionaries to a JSONL file.

    Args:
        file_path (str): Path to the output JSONL file.
        batch (List[dict]): The batch of dictionaries to write.
        append (bool): If True, appends to the file. If False, overwrites.
        encoding (str): Encoding to use when opening the file.
    """

    mode = "a" if append else "w"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    async with aiofiles.open(file_path, mode, encoding=encoding) as out_f:
        for item in batch:
            await out_f.write(json.dumps(item) + "\n")
