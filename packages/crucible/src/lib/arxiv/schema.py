"""
Schema definitions for ArXiv metadata.

`@author`: DAShaikh10
"""

from typing import List, TypedDict


class Reference(TypedDict):
    """
    Reference schema representing a cited paper in the ArXiv metadata.
    Populated from Semantic Scholar API.
    """

    arxiv_id: str | None
    title: str
    url: str | None


class ArXivMetadata(TypedDict):
    """
    ArXiv metadata schema representing a research paper.
    """

    abstract: str
    arxiv_id: str
    authors: List[str]
    influential_citations: int
    published_date: str
    references: List[Reference]
    title: str
    url: str
