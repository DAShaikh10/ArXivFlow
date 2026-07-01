"""
Runtime configuration for the ArXivFlow API.

`@author`: DAShaikh10
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# Directory holding this file: `<root>/apps/api/src` locally, `/app/src` in the Docker image
# (the Dockerfile flattens the sources), so we can't assume a fixed monorepo depth.
_CURRENT_DIR: str = os.path.dirname(os.path.abspath(__file__))


def _resolve(relative: str) -> Path:
    """
    Resolve a data-artifact path relative to the data directory, honouring the `DATA_DIR` override.

    Mirrors `packages/papervec/src/utils/path.py`: locally the data dir is `<root>/data`; in the
    Docker image the sources live at `/app/src` and the PVC is mounted at `/app/data`. `DATA_DIR`
    overrides both. An absolute `relative` value is returned unchanged (pathlib join semantics).
    """

    normalized = _CURRENT_DIR.replace("\\", "/")
    if "apps/api" in normalized:
        # Local monorepo: <root>/apps/api/src -> <root>
        project_root = os.path.abspath(os.path.join(_CURRENT_DIR, "../../.."))
    else:
        # Docker image: /app/src -> /app
        project_root = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))

    data_dir = os.environ.get("DATA_DIR", os.path.join(project_root, "data"))
    return Path(data_dir) / relative


ENRICHED_DATASET_FILE: Path = _resolve(os.getenv("ENRICHED_DATASET_FILE"))
EMBEDDING_DATABASE_PATH: Path = _resolve(os.getenv("EMBEDDING_DATABASE_PATH"))
ATLAS_PROJECTION_FILE: Path = _resolve(os.getenv("ATLAS_PROJECTION_FILE"))

EMBEDDING_COLLECTION_NAME: str = os.getenv("EMBEDDING_COLLECTION_NAME", "ArXivFlow")

SPECTER2_BASE_MODEL: str = os.getenv("SPECTER2_BASE_MODEL")
SPECTER2_QUERY_ADAPTER: str = os.getenv("SPECTER2_QUERY_ADAPTER")
SPECTER2_MAX_SEQ_LENGTH: int = int(os.getenv("SPECTER2_MAX_SEQ_LENGTH"))

# Search v2 RRF fusion (default recommender): fuse the top-`POOL` of the dense and BM25 rankings by
# 1/(RRF_K + rank). RRF_K=60 is the common default; the pool bounds how deep each list contributes.
SEARCH_FUSION_POOL: int = int(os.getenv("SEARCH_FUSION_POOL"))
SEARCH_RRF_K: int = int(os.getenv("SEARCH_RRF_K"))

API_HOST: str = os.getenv("API_HOST")
API_PORT: int = int(os.getenv("API_PORT"))

CORS_ORIGINS: list[str] = [origin.strip() for origin in os.getenv("CORS_ORIGINS").split(",") if origin.strip()]

# Atlas projection knobs (consumed by src/build_atlas.py only).
ATLAS_CLUSTERS: int = int(os.getenv("ATLAS_CLUSTERS"))
ATLAS_UMAP_NEIGHBORS: int = int(os.getenv("ATLAS_UMAP_NEIGHBORS"))
ATLAS_UMAP_MIN_DIST: float = float(os.getenv("ATLAS_UMAP_MIN_DIST"))

RANDOM_SEED: int = int(os.getenv("RANDOM_SEED"))
