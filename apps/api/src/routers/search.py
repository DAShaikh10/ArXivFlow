"""
Free-text search endpoint — "Semantic Search v2".

`@author`: DAShaikh10
"""

import logging

from fastapi import APIRouter, HTTPException, Query

from ..schemas import SearchResponse
from ..store import store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["search"])


@router.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Free-text query, ranked over the corpus."),
    k: int = Query(20, ge=1, le=50),
    recommender: str = Query(
        "rrf",
        pattern="^(rrf|dense|bm25)$",
        description="rrf=dense+BM25 fusion; dense=SPECTER2 ad-hoc query; bm25=lexical over Title+Abstract.",
    ),
) -> SearchResponse:
    """
    Top-k papers ranked by the chosen recommender (empty result set if nothing matches).
    """

    try:
        return SearchResponse(**store.search(q, k=k, recommender=recommender))
    except Exception as exc:  # pragma: no cover - surfaces a model-load/inference failure clearly.
        # Log the real traceback so a genuine bug (e.g. in result assembly) isn't silently reported as
        # an encoder failure.
        logger.exception("Search failed (recommender=%s, q=%r)", recommender, q)
        if recommender in ("rrf", "dense"):
            raise HTTPException(
                status_code=503,
                detail=f"Dense search is unavailable (SPECTER2 query encoder failed to load or run): {exc}",
            ) from exc
        raise
