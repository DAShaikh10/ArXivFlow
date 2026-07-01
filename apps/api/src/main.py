"""
ArXivFlow API entry point.

Boots the in-memory DataStore once (dataset + ChromaDB + atlas projection) inside the FastAPI
lifespan, then serves paper listings and SPECTER2 embedding neighbours to the Next.js client.

Run locally:  moon run api:dev

`@author`: DAShaikh10
"""

import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from . import config
from .query_encoder import query_encoder
from .routers import embeddings, papers, search
from .schemas import HealthResponse
from .store import store


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Load all data artifacts before the server starts accepting traffic.
    """

    store.load()
    # Warm the SPECTER2 query encoder off the request path so the first dense search (the default
    # recommender) doesn't stall on the one-time model download + torch init. Everything else serves
    # immediately; a dense request arriving mid-warmup waits on the same lock-guarded load.
    threading.Thread(target=query_encoder.warmup, name="specter2-warmup", daemon=True).start()
    yield


app = FastAPI(
    title="ArXivFlow API",
    version="0.1.0",
    description="Paper listings & SPECTER2 embedding neighbours over the indexed arXiv NLP corpus.",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(papers.router)
app.include_router(embeddings.router)
app.include_router(search.router)


@app.get("/api/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """
    Liveness + data-readiness probe.
    """

    paper_count, embedding_count = store.counts()
    return HealthResponse(
        status="ok",
        papers=paper_count,
        embeddings=embedding_count,
        atlas_ready=store.atlas_ready,
        search_ready=query_encoder.ready,
    )


def main() -> None:
    """
    Console entry point.
    """

    uvicorn.run("src.main:app", host=config.API_HOST, port=config.API_PORT, reload=False)


if __name__ == "__main__":
    main()
