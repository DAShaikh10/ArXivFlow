"""
Utilities for a Label Studio model base class.

`@author`: DAShaikh10
"""

from abc import ABC
from typing import Optional

from label_studio_sdk.label_interface import LabelInterface
from label_studio_sdk._extensions.label_studio_tools.core.label_config import parse_config

import wandb

from src.utils import logger

# pylint: disable=too-few-public-methods


class LabelStudioMLBase(ABC):
    """
    Base class for Label Studio ML backends (compatibility surface).
    """

    def __init__(self, project_id: Optional[str], label_config=None, **kwargs) -> None:
        self._label_studio_client = None
        self.extra_params = kwargs
        self.project_id = project_id or ""

        if label_config is not None:
            self.label_config = label_config
            self.parsed_label_config = parse_config(label_config)
            self.label_interface = LabelInterface(config=label_config)
        else:
            logger.warning("Label config is not provided")
            wandb.log({"label_config_provided": False})


# pylint: enable=too-few-public-methods
