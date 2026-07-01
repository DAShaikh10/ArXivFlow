"""
SciNCL implementation for research paper embedding generation.

`@author`: DAShaikh10
"""

import os
from typing import Optional, Tuple

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

import wandb
from src.utils import logger, resolve_path

from . import config


class SciNCL:
    """
    SciNCL scientific-document encoder for paper-to-paper recommendation and similarity tasks.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        """
        Initializes the model class and detects the appropriate compute device.
        """

        logger.info("Initializing SciNCL with model '%s'", model_name or config.SCINCL_MODEL_NAME)
        wandb.log({"scincl_model": model_name or config.SCINCL_MODEL_NAME})

        self.model_name = model_name or config.SCINCL_MODEL_NAME
        self.device = self._get_device()
        self.model = None
        self.tokenizer = None

    def _get_device(self) -> torch.device:
        """
        Find appropriate compute device (GPU, MPS, or CPU) for model inference.

        Returns:
            torch.device: The best available compute device for model inference.
        """

        available_device = torch.device("cpu")

        # Determine the best device gracefully.
        if torch.cuda.is_available():
            available_device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            available_device = torch.device("mps")

        logger.info("Using compute device: %s", available_device)
        wandb.log({"scincl_compute_device": str(available_device)})

        return available_device

    def load(self) -> None:
        """
        Load the tokenizer and model onto the appropriate device. No adapter step — SciNCL's citation
        training lives in the base weights.
        """

        logger.debug("load - START")
        wandb.log({"scincl_initialization": "START"})

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

        self.model.to(self.device)
        self.model.eval()

        logger.debug("load - END")
        wandb.log({"scincl_initialization": "END"})

    def format_input(self, title: str, abstract: str) -> str:
        """
        Formats the title and abstract exactly as SciNCL expects: Title and Abstract concatenated with
        the tokenizer's [SEP] token. Identical to the SPECTER2 formatting so the two stores stay
        comparable on the same corpus.

        Args:
            title (str): The title of the paper.
            abstract (str): The abstract of the paper.

        Returns:
            str: The formatted input string for the model.
        """

        return f"{title}{self.tokenizer.sep_token}{abstract}"

    def generate_embeddings(self, encoding: str = "utf-8") -> Tuple[torch.Tensor, list[str], pd.DataFrame]:
        """
        Read through the papers dataset, format the inputs, and generate embeddings using the model.

        Currently, the code assumes ample memory availability & hence does not employ any memory-efficient techniques.
        """

        logger.debug("generate_embeddings - START")
        wandb.log({"scincl_embedding_generation": "START"})

        current_dir: str = os.path.dirname(__file__)
        dataset_path: str = resolve_path(current_dir, config.ENRICHED_DATASET_FILE)

        # NOTE: This code assumes ample memory availability & hence does not employ any memory-efficient techniques.
        raw: pd.DataFrame = pd.read_json(dataset_path, encoding=encoding, lines=True)
        # `authors` is optional — the current enriched dataset dropped it, and it only feeds nice-to-have
        # author metadata (never the embeddings or any eval signal). Backfill None so downstream stays uniform.
        if "authors" not in raw.columns:
            raw["authors"] = None
        papers: pd.DataFrame = raw[["arxiv_id", "title", "abstract", "authors"]]
        records = papers.to_dict(orient="records")

        # Create formatter inputs.
        formatted_inputs: list[str] = [
            self.format_input(record.get("title"), record.get("abstract")) for record in records
        ]

        logger.debug("Tokenizing %d papers...", len(formatted_inputs))
        wandb.log({"scincl_num_papers": len(formatted_inputs)})

        inputs = self.tokenizer(
            formatted_inputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=config.MAX_SEQ_LENGTH,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logger.debug("Executing model generation...")
        wandb.log({"scincl_embedding_generation": "MODEL_EXECUTION"})

        with torch.no_grad():
            outputs = self.model(**inputs)

            # Extract the special [CLS] token representation used for paper matching.
            # The [CLS] token is at index 0 of the sequence dimension.
            embeddings = outputs.last_hidden_state[:, 0, :]

        logger.debug("generate_embeddings - END")
        wandb.log({"scincl_embedding_generation": "END"})

        return embeddings, formatted_inputs, papers
