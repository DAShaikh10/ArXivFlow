"""
Paper listing & detail endpoints — the "listing part" of the corpus.

`@author`: DAShaikh10
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ..schemas import Category, PaperDetail, PaperListResponse
from ..store import store

router = APIRouter(prefix="/api", tags=["papers"])


@router.get("/categories", response_model=list[Category])
def list_categories() -> list[Category]:
    """
    Embedding clusters used to group/colour papers (empty until the atlas projection is built).
    """

    return [Category(**category) for category in store.categories()]


@router.get("/years", response_model=list[int])
def list_years() -> list[int]:
    """
    Publication years present in the corpus, ascending — the options for the 'Since' filter.
    """

    return store.years()


@router.get("/papers", response_model=PaperListResponse)
def list_papers(
    sort: str = Query("cited", pattern="^(cited|newest|title)$"),
    year_from: Optional[int] = Query(None, ge=1900, le=2100),
    year_to: Optional[int] = Query(None, ge=1900, le=2100),
    cluster_id: Optional[str] = Query(None),
    q: Optional[str] = Query(None, description="Title substring filter."),
    limit: int = Query(25, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> PaperListResponse:
    """
    A filtered, sorted, paginated page of the indexed corpus.
    """

    return PaperListResponse(
        **store.list_papers(
            sort=sort,
            year_from=year_from,
            year_to=year_to,
            cluster_id=cluster_id,
            query=q,
            limit=limit,
            offset=offset,
        )
    )


@router.get("/papers/{arxiv_id}", response_model=PaperDetail)
def get_paper(arxiv_id: str) -> PaperDetail:
    """
    Full paper detail: metadata, references (with corpus deep-link flags), and canonical topics.
    """

    paper = store.get_paper(arxiv_id)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"Paper '{arxiv_id}' not found in the corpus.")
    return PaperDetail(**paper)
