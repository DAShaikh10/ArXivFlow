"""
Models library to be used in Label Studio ML backend.
Each model should inherit from LabelStudioMLBase and implement the required methods for inference.

`@author`: DAShaikh10
"""

from .model import LabelStudioMLBase

__all__ = ["LabelStudioMLBase"]
