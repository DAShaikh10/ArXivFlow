"""
Precompute the 2D embedding atlas.

The 768-dim SPECTER2 vectors are projected to 2D with UMAP for layout, and KMeans assigns each paper
to a cluster. Each cluster is named after the canonical NER tag most common among its papers, read from the same Chroma
metadata papervec wrote.

The result is written to `data/embeddings/atlas.json` and served verbatim by `GET /api/atlas`.

`@author`: DAShaikh10
"""

import json
import re
from collections import Counter, defaultdict

import chromadb
import numpy as np
from sklearn.cluster import KMeans
from umap import UMAP

from src.utils import logger

from . import config

# A categorical palette wide enough for the default cluster count; reused cyclically if exceeded.
PALETTE = [
    "#5b5bf0",
    "#0ea5e9",
    "#ec4899",
    "#22c55e",
    "#f59e0b",
    "#ef4444",
    "#14b8a6",
    "#a855f7",
    "#84cc16",
    "#f97316",
    "#06b6d4",
    "#e11d48",
]

# Field preference for cluster naming: a task-oriented label reads best on the legend.
NAME_FIELD_PRIORITY: list[str] = [
    "target_nlp_task",
    "machine_learning_architecture",
    "application_domain",
    "training_method",
]

# Generic / structural words filtered out when naming a cluster from paper titles (the fallback used
# when this ChromaDB build carries no canonical-tag metadata).
TITLE_STOPWORDS: frozenset[str] = frozenset("""
    a an and are as at be by for from in into is it its of on or our the their this to via with we
    using use based toward towards using approach approaches method methods model models learning
    neural deep network networks via new novel improving improved efficient effective robust
    framework system systems study analysis evaluation paper task tasks data via large language
    """.split())
_WORD_RE = re.compile(r"[a-z][a-z0-9\-]+")


def _prettify(value: str) -> str:
    """
    Turn a canonical tag id ('nmt', 'question answering') into a legend label.

    Args:
        value (str): the raw tag value, typically a lowercase string with words separated by underscores

    Returns:
        str: a prettified label, e.g. 'NMT' or 'Question Answering'
    """

    cleaned = value.replace("_", " ").strip()
    return cleaned.upper() if len(cleaned) <= 4 else cleaned.title()


def _cluster_name(topic_counts: Counter) -> str:
    """
    Pick the most representative canonical tag for a cluster, honouring the field priority.

    Args:
        topic_counts (Counter): counts of (field, value) pairs across the cluster's papers,
        e.g. {("target_nlp_task", "question answering"): 12}

    Returns:
        str: a candidate cluster name, or empty if none found.
    """

    if not topic_counts:
        return ""

    for field in NAME_FIELD_PRIORITY:
        scoped = Counter({(f, v): n for (f, v), n in topic_counts.items() if f == field})
        if scoped:
            (_, value), _ = scoped.most_common(1)[0]
            return _prettify(value)

    (_, value), _ = topic_counts.most_common(1)[0]
    return _prettify(value)


def _name_from_titles(titles: list[str], used: set[str]) -> str:
    """
    Fallback cluster name: the most distinctive content word across the cluster's titles.

    Args:
        titles (list[str]): the paper titles in the cluster (metadata may be missing, so we use these as a last resort).
        used (set[str]): names already taken by other clusters, to avoid duplicates.

    Returns:
        str: a candidate cluster name, or empty if none found.
    """

    counts = Counter(
        word
        for title in titles
        for word in _WORD_RE.findall(title.lower())
        if word not in TITLE_STOPWORDS and len(word) > 2
    )
    for word, _ in counts.most_common():
        label = _prettify(word)
        if label not in used:
            return label
    return ""


def _dimentionality_reduction(ids: list[str], vectors: np.ndarray):
    n_clusters: int = min(config.ATLAS_CLUSTERS, len(ids))
    logger.info("Clustering into %s groups (KMeans) ...", n_clusters)
    labels = KMeans(n_clusters=n_clusters, random_state=config.RANDOM_SEED).fit_predict(vectors)

    logger.debug("Projecting to 2D (UMAP, cosine metric) ...")
    coords = UMAP(
        n_components=2,
        n_neighbors=config.ATLAS_UMAP_NEIGHBORS,
        min_dist=config.ATLAS_UMAP_MIN_DIST,
        metric="cosine",
        random_state=config.RANDOM_SEED,
    ).fit_transform(vectors)

    return n_clusters, labels, coords

# pylint: disable=too-many-locals

def main() -> None:
    """
    Entry point that reads the SPECTER2 embeddings and metadata from ChromaDB, cluster and project them,
    then write the result as a JSON file for the frontend to consume.
    """

    logger.debug("main - START")

    client: chromadb.ClientAPI = chromadb.PersistentClient(path=str(config.EMBEDDING_DATABASE_PATH))
    collection: chromadb.Collection = client.get_collection(name=config.EMBEDDING_COLLECTION_NAME)

    record: chromadb.GetResult = collection.get(include=["embeddings", "metadatas", "documents"])
    ids: list[str] = record["ids"]

    vectors = np.asarray(record["embeddings"], dtype=np.float32)
    metadatas = record.get("metadatas") or [None] * len(ids)
    documents = record.get("documents") or [""] * len(ids)

    # Documents are "Title[SEP]Abstract" (papervec format); keep just the title for cluster naming.
    _titles = {paper_id: (doc or "").split("[SEP]", 1)[0].strip() for paper_id, doc in zip(ids, documents)}
    logger.debug("Loaded %s embeddings of dim %s.", len(ids), vectors.shape[1])

    n_clusters, labels, coords = _dimentionality_reduction(ids, vectors)

    # Scale coordinates into a friendly ~1000x680 SVG-style viewport, padded.
    coords = np.asarray(coords, dtype=np.float64)
    mins, maxs = coords.min(axis=0), coords.max(axis=0)
    span = np.where(maxs - mins == 0, 1.0, maxs - mins)
    norm = (coords - mins) / span
    norm[:, 0] = norm[:, 0] * 920 + 40
    norm[:, 1] = norm[:, 1] * 600 + 40

    # Gather per-cluster topic frequencies (and titles, for the no-metadata fallback) for naming.
    cluster_topics: dict[int, Counter] = defaultdict(Counter)
    cluster_titles: dict[int, list[str]] = defaultdict(list)
    for paper_id, label, metadata in zip(ids, labels, metadatas):
        cluster_titles[int(label)].append(_titles.get(paper_id, ""))
        if not metadata:
            continue
        for key, present in metadata.items():
            if present and ":" in key:
                field, value = key.split(":", 1)
                cluster_topics[int(label)][(field, value)] += 1

    points = [
        {
            "id": paper_id,
            "x": round(float(norm[i, 0]), 2),
            "y": round(float(norm[i, 1]), 2),
            "cluster_id": f"c{int(labels[i])}",
        }
        for i, paper_id in enumerate(ids)
    ]

    counts = Counter(int(label) for label in labels)
    used_names: set[str] = set()
    categories = []
    for cluster in range(n_clusters):
        name = (
            _cluster_name(cluster_topics.get(cluster, Counter()))
            or _name_from_titles(cluster_titles.get(cluster, []), used_names)
            or f"Cluster {cluster + 1}"
        )
        # Disambiguate clusters that resolve to the same dominant tag.
        if name in used_names:
            name = f"{name} ({cluster + 1})"
        used_names.add(name)
        categories.append(
            {
                "id": f"c{cluster}",
                "name": name,
                "color": PALETTE[cluster % len(PALETTE)],
                "count": counts.get(cluster, 0),
            }
        )

    payload = {"points": points, "categories": categories, "count": len(points)}
    config.ATLAS_PROJECTION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.ATLAS_PROJECTION_FILE, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    logger.debug("Wrote %s points / %s clusters to %s", len(points), len(categories), config.ATLAS_PROJECTION_FILE)
    for category in categories:
        logger.debug("  %s  %s  %s", category["id"], category["count"], category["name"])

    logger.debug("main - END")

# pylint: enable=too-many-locals

if __name__ == "__main__":
    main()
