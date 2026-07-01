"""
Related-paper & atlas endpoints.

"More Like This" is served by a selectable recommender: `dense` (SPECTER2 cosine nearest neighbours
from ChromaDB) or `bm25` (lexical similarity over Title+Abstract). BM25 is the stronger signal on the
honest tag-overlap evaluation; dense stays the default and the semantic reference. The 2D atlas is
served from the precomputed projection.

`@author`: DAShaikh10
"""

from fastapi import APIRouter, HTTPException, Query

from ..schemas import AtlasResponse, NeighborsResponse
from ..store import store

router = APIRouter(prefix="/api", tags=["embeddings"])


@router.get("/papers/{arxiv_id}/neighbors", response_model=NeighborsResponse)
def paper_neighbors(
    arxiv_id: str,
    k: int = Query(8, ge=1, le=50),
    recommender: str = Query("dense", pattern="^(dense|bm25)$", description="dense=SPECTER2 cosine; bm25=lexical."),
) -> NeighborsResponse:
    """
    Top-k related papers for a paper (excluding itself), via the chosen recommender.
    """

    result = store.neighbors(arxiv_id, k=k, recommender=recommender)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Paper '{arxiv_id}' not found in the corpus.")
    return NeighborsResponse(**result)


@router.get("/atlas", response_model=AtlasResponse)
def atlas() -> AtlasResponse:
    """
    Every paper projected to 2D, plus the cluster legend, for the Embedding Atlas view.
    """

    if not store.atlas_ready:
        raise HTTPException(
            status_code=503,
            detail="Atlas projection not built. Run `moon run api:atlas` (src/build_atlas.py), then restart the API.",
        )
    return AtlasResponse(**store.atlas())
