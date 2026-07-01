"""
Main entry point for SciNCL embedding generation and storage into ChromaDB with canonical tags.

`@author`: DAShaikh10
"""

import os
from typing import Dict

import chromadb

import wandb
from src.utils import logger, resolve_path

from ..allenai import tags
from . import config
from .scincl import SciNCL


def _metadata_with_authors(
    arxiv_id: str, authors_by_id: Dict[str, str], tags_by_id: list[str]
) -> Dict[str, bool | str] | None:
    """
    Merge canonical tags with the author string; returns None when the paper has neither.

    Args:
        arxiv_id (str): The arXiv ID of the paper for which to build metadata.
        authors_by_id (Dict[str, str]): A mapping from arXiv ID to a semicolon-separated string of authors.
        tags_by_id (list[str]): A list of canonical tags associated with the paper.

    Returns:
        Dict[str, bool | str] | None: A dictionary containing canonical tags and author string for the given paper,
        or None if the paper has neither tags nor authors.
    """

    metadata: Dict[str, bool | str] = dict(tags.metadata_for(arxiv_id, tags_by_id) or {})
    author_str: str | None = authors_by_id.get(arxiv_id)
    if author_str:
        metadata["authors"] = author_str
    return metadata or None


if __name__ == "__main__":
    # Initialize WandB and log configuration.
    wandb.init(
        project=config.WANDB_PROJECT_NAME,
        name="scincl_embedding_generation",
        config={
            "SCINCL_MODEL_NAME": config.SCINCL_MODEL_NAME,
            "EMBEDDING_DATABASE_NAME": config.EMBEDDING_DATABASE_NAME,
            "SCINCL_COLLECTION_NAME": config.SCINCL_COLLECTION_NAME,
        },
        job_type="embeddings",
    )

    # Resolve the path for the ChromaDB database file (shared with the SPECTER2 store).
    current_dir: str = os.path.dirname(__file__)
    database_path: str = resolve_path(current_dir, config.EMBEDDING_DATABASE_NAME)

    # Instantiate ChromaDB client and create/get the SciNCL collection (distinct from SPECTER2's).
    chroma_client: chromadb.ClientAPI = chromadb.PersistentClient(path=database_path)
    collection: chromadb.Collection = chroma_client.get_or_create_collection(
        name=config.SCINCL_COLLECTION_NAME,
        # Hierarchical Navigable Small World (HNSW) index with cosine similarity.
        metadata={"hnsw:space": "cosine"},
    )

    # Initialize the SciNCL embedder and generate embeddings.
    embedder = SciNCL()

    embedder.load()
    embeddings, formatted_inputs, papers = embedder.generate_embeddings()
    embeddings_list = embeddings.cpu().numpy().tolist()

    logger.debug("Embeddings shape: %s", embeddings.shape)
    wandb.log({"embedding_dimension": embeddings.shape})

    # Build per-paper canonical tag metadata so papers can be filtered by tag at query time
    # (e.g. collection.query(..., where={"machine_learning_architecture:bert": True})).
    annotation_path: str = resolve_path(current_dir, config.ANNOTATION_FILE)
    canonical_map_path: str = resolve_path(current_dir, config.CANONICAL_MAP_FILE)

    canonical_map = tags.load_canonical_map(canonical_map_path)
    paper_tags = tags.build_paper_tags(tags.load_records(annotation_path), canonical_map)

    arxiv_ids: list[str] = papers.get("arxiv_id").tolist()
    authors_list: list[list[str] | None] = papers.get("authors").tolist()

    author_map: Dict[str, str] = {}
    for paper_id, authors in zip(arxiv_ids, authors_list):
        if isinstance(authors, list) and authors:
            author_map[paper_id] = "; ".join(author for author in authors if author)

    # Aligned with `arxiv_ids`; papers with neither tags nor authors get `None` (Chroma rejects empty dicts).
    metadatas: list[Dict[str, bool | str] | None] = [
        _metadata_with_authors(id, author_map, paper_tags) for id in arxiv_ids
    ]
    num_tagged = sum(1 for metadata in metadatas if metadata)
    logger.info("%d of %d papers carry canonical tags", num_tagged, len(arxiv_ids))
    wandb.log({"num_papers_tagged": num_tagged})

    # Ingest into ChromaDB.
    logger.info("Adding %s papers to ChromaDB...", len(embeddings_list))
    wandb.log({"num_papers_embedded": len(embeddings_list)})

    collection.add(
        ids=arxiv_ids,
        embeddings=embeddings_list,
        documents=formatted_inputs,
        metadatas=metadatas,
    )

    # Save output dataset as a WandB Artifact.
    artifact = wandb.Artifact("scincl-embeddings", type="database")
    artifact.add_dir(database_path)
    wandb.log_artifact(artifact)

    wandb.finish()
