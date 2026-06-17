"""
Pydantic response models for the ArXivFlow API.

These mirror the shapes the Next.js client consumes (see apps/client/src/lib/types.ts). Field names
stay snake_case on the wire; the client maps them at its boundary.

`@author`: DAShaikh10
"""

from typing import Optional

from pydantic import BaseModel


class Category(BaseModel):
    """
    An embedding cluster used to colour/group papers.
    """

    id: str
    name: str
    color: str
    count: int


class Reference(BaseModel):
    """
    A cited work, as scraped into the enriched dataset.
    """

    arxiv_id: Optional[str] = None
    title: str
    url: Optional[str] = None
    # True when the referenced paper is itself part of our indexed corpus (so the UI can deep-link).
    in_corpus: bool = False


class Topic(BaseModel):
    """
    A canonical NER tag attached to a paper via human annotation (Chroma metadata).
    """

    field: str
    value: str


class PaperSummary(BaseModel):
    """
    The card shape rendered in the Discover feed and atlas neighbour lists.
    """

    id: str
    title: str
    abstract: str
    authors: list[str] = []
    published_date: Optional[str] = None
    influential_citations: int = 0
    reference_count: int = 0
    url: Optional[str] = None
    cluster_id: Optional[str] = None
    # Citation-authority score (0..1): log1p(influential_citations) normalised by the corpus max.
    # A real standing-in-the-field signal, not relevance — see store._prominence. Shown per card;
    # the feed itself is ranked by raw citations ("cited" sort)
    prominence: float = 0.0


class PaperDetail(PaperSummary):
    """
    Full paper view: summary + references + canonical topics.
    """

    references: list[Reference] = []
    topics: list[Topic] = []


class Neighbor(BaseModel):
    """
    A nearest neighbour returned by a cosine query against the SPECTER2 embeddings.
    """

    id: str
    title: str
    authors: list[str] = []
    similarity: float
    published_date: Optional[str] = None
    influential_citations: int = 0
    cluster_id: Optional[str] = None


class NeighborsResponse(BaseModel):
    """
    Embedding-similarity result set for a paper, with the measured query latency.
    """

    source_id: str
    neighbors: list[Neighbor]
    took_ms: float


class PaperListResponse(BaseModel):
    """
    A page of the corpus listing.
    """

    items: list[PaperSummary]
    total: int
    limit: int
    offset: int
    took_ms: float


class AtlasPoint(BaseModel):
    """
    One projected paper in the 2D embedding atlas.
    """

    id: str
    x: float
    y: float
    cluster_id: str
    title: str
    published_date: Optional[str] = None
    influential_citations: int = 0


class AtlasResponse(BaseModel):
    """
    The full atlas payload: every projected point plus the cluster legend.
    """

    points: list[AtlasPoint]
    categories: list[Category]
    count: int


class HealthResponse(BaseModel):
    """
    Liveness + data-readiness probe.
    """

    status: str
    papers: int
    embeddings: int
    atlas_ready: bool
