"""
Pydantic schemas used by the FastAPI backend.

`@author`: DAShaikh10
"""

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class ModelResponse(BaseModel):
    """
    Response schema returned by the model endpoints.

    - `model_version`: optional model version string
    - `predictions`: list of prediction objects
    """

    model_version: Optional[str] = None
    predictions: List[Any] = Field(default_factory=list)
