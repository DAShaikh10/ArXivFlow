"""
Main entry point for AllenAI Specter2Proximity embedding generation and storage into ChromaDB with canonical tags.

`@author`: DAShaikh10
"""

import os
from typing import Dict

import chromadb

import wandb
from src.utils import logger, resolve_path

from . import config, tags
from .specter import Specter2Proximity

if __name__ == "__main__":
    # Initialize WandB and log configuration.
    wandb.init(
        project=config.WANDB_PROJECT_NAME,
        name="specter2proximity_embedding_generation",
        config={
            "HF_ADAPTER_NAME": config.ADAPTER_NAME,
            "EMBEDDING_DATABASE_NAME": config.EMBEDDING_DATABASE_NAME,
            "HF_MODEL_NAME": config.BASE_MODEL_NAME,
        },
        job_type="embeddings",
    )

    # Resolve the path for the ChromaDB database file.
    current_dir: str = os.path.dirname(__file__)
    database_path: str = resolve_path(current_dir, config.EMBEDDING_DATABASE_NAME)

    # Instantiate ChromaDB client and create/get collection for storing embeddings.
    chroma_client: chromadb.ClientAPI = chromadb.PersistentClient(path=database_path)
    collection: chromadb.Collection = chroma_client.get_or_create_collection(
        name=config.WANDB_PROJECT_NAME,
        # Hierarchical Navigable Small World (HNSW) index with cosine similarity.
        metadata={"hnsw:space": "cosine"},
    )

    # Initialize the Specter2Proximity embedder and generate embeddings.
    embedder = Specter2Proximity()

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
    # Aligned with `arxiv_ids`; untagged papers get `None` (Chroma rejects empty metadata dicts).
    metadatas: list[Dict[str, bool] | None] = [tags.metadata_for(arxiv_id, paper_tags) for arxiv_id in arxiv_ids]
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
    artifact = wandb.Artifact("specter-2-proximity-embeddings", type="database")
    artifact.add_dir(database_path)
    wandb.log_artifact(artifact)

    wandb.finish()
