"""
AllenAI Specter 2 implementation for research paper embedding generation.

`@author`: DAShaikh10
"""

import os
from typing import Optional, Tuple

import pandas as pd
import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

import wandb
from src.utils import logger, resolve_path

from . import config


class Specter2Proximity:
    """
    Specter 2 model with the proximity adapter for paper-to-paper recommendation
    and similarity tasks.
    """

    def __init__(
        self,
        base_model_name: Optional[str] = None,
        adapter_name: Optional[str] = None,
    ) -> None:
        """
        Initializes the model class and detects the appropriate compute device.
        """

        logger.info(
            "Initializing Specter2Proximity with base model '%s' and adapter '%s'", base_model_name, adapter_name
        )
        wandb.log({"specter2_base_model": base_model_name, "specter2_adapter": adapter_name})

        self.adapter_name = adapter_name or config.ADAPTER_NAME
        self.base_model_name = base_model_name or config.BASE_MODEL_NAME
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
        wandb.log({"specter2_compute_device": str(available_device)})

        return available_device

    def load(self) -> None:
        """
        Load the tokenizer and adapter-enabled model onto the appropriate device.
        """

        logger.debug("load - START")
        wandb.log({"specter2_initialization": "START"})

        # Load the model tokenizer and model with adapter support.
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.model = AutoAdapterModel.from_pretrained(self.base_model_name)

        # Load and activate the proximity adapter configuration.
        self.model.load_adapter(self.adapter_name, source="hf", load_as="specter2_proximity", set_active=True)
        self.model.to(self.device)
        self.model.eval()

        logger.debug("load - END")
        wandb.log({"specter2_initialization": "END"})

    def format_input(self, title: str, abstract: str) -> str:
        """
        Formats the title and abstract exactly as expected by Specter 2.
        Concatenates Title and Abstract separated explicitly by the [SEP] token.

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
        wandb.log({"specter2_embedding_generation": "START"})

        current_dir: str = os.path.dirname(__file__)
        dataset_path: str = resolve_path(current_dir, config.ENRICHED_DATASET_FILE)

        # NOTE: This code assumes ample memory availability & hence does not employ any memory-efficient techniques.
        papers: pd.DataFrame = pd.read_json(dataset_path, encoding=encoding, lines=True)[
            ["arxiv_id", "title", "abstract"]
        ]
        records = papers.to_dict(orient="records")

        # Create formatter inputs.
        formatted_inputs: list[str] = [
            self.format_input(record.get("title"), record.get("abstract")) for record in records
        ]

        logger.debug("Tokenizing %d papers...", len(formatted_inputs))
        wandb.log({"specter2_num_papers": len(formatted_inputs)})

        inputs = self.tokenizer(
            formatted_inputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=config.MAX_SEQ_LENGTH,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logger.debug("Executing model generation...")
        wandb.log({"specter2_embedding_generation": "MODEL_EXECUTION"})

        with torch.no_grad():
            outputs = self.model(**inputs)

            # Extract the special [CLS] token representation used for paper matching.
            # The [CLS] token is at index 0 of the sequence dimension.
            embeddings = outputs.last_hidden_state[:, 0, :]

        logger.debug("generate_embeddings - END")
        wandb.log({"specter2_embedding_generation": "END"})

        return embeddings, formatted_inputs, papers
