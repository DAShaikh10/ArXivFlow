"""
Schema definitions for NER module.

`@author`: DAShaikh10
"""

from typing import TypedDict, List


class Annotation(TypedDict):
    """
    Annotation schema representing the final output of the NER process for a single paper.
    """

    id: str
    entities: List[Entity]


class Entity(TypedDict):
    """
    Entity schema representing a named entity found in the paper abstract.
    """

    text: str
    label: str
    start: int
    end: int


class Paper(TypedDict):
    """
    Paper schema representing the essential information of a research paper for NER processing.
    """

    id: str
    abstract: str
