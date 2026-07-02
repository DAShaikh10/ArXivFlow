"""
Corpus-relatedness analysis: how related the corpus papers actually are, measured several
*independent* ways, and how strongly those independent signals agree.

This module only **counts and compares** the positive-pair graphs the Tier 0 ground-truth builders
(`src.lib.eval.groundtruth`) already produce — it adds no new relation logic, so every relation here
is exactly the one the evaluation harness scores against. The two questions it answers:

  * Density   — for each relation definition, how many papers have at least one related neighbour,
                how many pairs, and the degree distribution.
  * Agreement — do two signals that share *no inputs* (NER tag overlap, built from abstract spans, vs
                bibliographic coupling, built from reference lists) flag the same pairs as related,
                and by how much above chance (lift)? Agreement above chance is the actual validation
                that the relatedness is real and not an artefact of one proxy.

`@author`: DAShaikh10
"""

import json
from statistics import median
from typing import Dict, List, Sequence, Set, Tuple

from src.lib.eval import groundtruth

# An undirected edge, endpoints order-normalized so each pair is counted once.
Pair = Tuple[str, str]


def corpus_ids_from_dataset(dataset_path: str) -> List[str]:
    """
    Ordered corpus ids read straight from the enriched dataset — every non-blank line is one paper.

    This is the corpus the presented numbers were computed over; it needs no embeddings DB. Use
    `chroma_corpus_ids` for parity with the eval harness (which keys off the Chroma collection).
    """

    ids: List[str] = []
    with open(dataset_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                ids.append(json.loads(line)["arxiv_id"])
    return ids


def to_pairs(positives: Dict[str, Set[str]]) -> Set[Pair]:
    """
    Undirected adjacency map -> set of order-normalized (a, b) pairs, each edge once.
    """

    pairs: Set[Pair] = set()
    for source, neighbours in positives.items():
        for neighbour in neighbours:
            pairs.add((source, neighbour) if source < neighbour else (neighbour, source))
    return pairs


def graph_stats(positives: Dict[str, Set[str]], ids: Sequence[str]) -> dict:
    """
    Coverage, edge count, and degree distribution for one relatedness graph over `ids`.
    """

    n = len(ids)
    degrees = [len(positives.get(pid, ())) for pid in ids]
    nonzero = [d for d in degrees if d]
    covered = len(nonzero)
    return {
        "corpus": n,
        "covered": covered,
        "coverage_pct": round(100 * covered / n, 1) if n else 0.0,
        "pairs": sum(degrees) // 2,
        "median_degree_nonzero": int(median(nonzero)) if nonzero else 0,
        "max_degree": max(degrees) if degrees else 0,
    }


def coupling_pairs(  # pylint: disable=too-many-arguments
    dataset_path: str,
    ids: Sequence[str],
    *,
    flavor: str = "count",
    min_shared: int = 2,
    min_similarity: float = 0.1,
    weights: Dict[Tuple[str, str], float] = None,
) -> Set[Pair]:
    """
    Coupling edges as a normalized pair set, in either flavor.

    `flavor="count"` — shared-reference count >= `min_shared` (the threshold knob).
    `flavor="idf"`   — IDF-weighted cosine >= `min_similarity` (the clean ranked graph). Pass a
                       precomputed `weights` (from `groundtruth.build_coupling_weights`) to avoid
                       recomputing it across calls.
    """

    if flavor == "idf":
        graph = weights if weights is not None else groundtruth.build_coupling_weights(dataset_path, ids)
        return {pair for pair, similarity in graph.items() if similarity >= min_similarity}
    return to_pairs(groundtruth.build_coupling_positives(dataset_path, ids, min_shared))


def density_rows(  # pylint: disable=too-many-arguments
    dataset_path: str,
    ids: Sequence[str],
    tag_sets: Dict[str, Set[str]],
    *,
    cocitation_thresholds: Sequence[int] = (1, 2),
    coupling_thresholds: Sequence[int] = (2, 3, 5),
    idf_coupling_similarities: Sequence[float] = (0.05, 0.1, 0.2),
    tag_thresholds: Sequence[int] = (1, 2),
    coupling_weights: Dict[Tuple[str, str], float] = None,
) -> List[dict]:
    """
    One row per relation definition: its label plus `graph_stats`.

    Defaults reproduce the presented density table (direct citation; co-citation >=1,>=2; coupling
    >=2,>=3,>=5; IDF coupling cos>=0.05,0.10,0.20; tag overlap >=1,>=2). Pass `coupling_weights` to
    reuse a precomputed IDF graph.
    """

    rows: List[Tuple[str, Dict[str, Set[str]]]] = [
        ("Direct citation", groundtruth.build_citation_positives(dataset_path, ids)),
    ]
    for threshold in cocitation_thresholds:
        rows.append(
            (f"Co-citation (≥{threshold})", groundtruth.build_cocitation_positives(dataset_path, ids, threshold))
        )
    for threshold in coupling_thresholds:
        rows.append(
            (
                f"Coupling (≥{threshold} shared refs)",
                groundtruth.build_coupling_positives(dataset_path, ids, threshold),
            )
        )
    if idf_coupling_similarities:
        graph = (
            coupling_weights if coupling_weights is not None else groundtruth.build_coupling_weights(dataset_path, ids)
        )
        for cutoff in idf_coupling_similarities:
            rows.append((f"IDF coupling (cos ≥{cutoff})", groundtruth.positives_from_weights(graph, cutoff)))
    for threshold in tag_thresholds:
        rows.append((f"Tag overlap (≥{threshold} shared tags)", groundtruth.build_tag_positives(tag_sets, threshold)))

    return [{"relation": label, **graph_stats(positives, ids)} for label, positives in rows]


# pylint: disable=redefined-outer-name


def agreement(tag_pairs: Set[Pair], coupling_pairs: Set[Pair], n_papers: int) -> dict:
    """
    Cross-signal agreement and lift over chance between two independent pair sets.

    `chance_rate` is the share of *all* possible pairs that coupling flags (the rate at which a random
    tag pair would land in the coupling set by chance). `observed_rate` is the share of tag pairs that
    are *also* coupled. `lift = observed / chance` — how many times more likely a tag-related pair is
    to be citation-coupled than chance. Lift well above 1 is the validation.
    """

    total_possible = n_papers * (n_papers - 1) // 2
    intersection = tag_pairs & coupling_pairs
    chance_rate = len(coupling_pairs) / total_possible if total_possible else 0.0
    observed_rate = len(intersection) / len(tag_pairs) if tag_pairs else 0.0
    return {
        "tag_pairs": len(tag_pairs),
        "coupling_pairs": len(coupling_pairs),
        "intersection": len(intersection),
        "total_possible_pairs": total_possible,
        "chance_rate": round(chance_rate, 4),
        "observed_rate": round(observed_rate, 4),
        "lift": round(observed_rate / chance_rate, 2) if chance_rate else 0.0,
    }


# pylint: enable=redefined-outer-name


def agreement_sweep(
    dataset_path: str,
    ids: Sequence[str],
    tag_sets: Dict[str, Set[str]],
    *,
    tag_min_shared: int = 2,
    coupling_thresholds: Sequence[int] = (2, 3, 5),
) -> List[dict]:
    """
    Agreement + lift of the tag-overlap graph against count-coupling at each shared-count threshold.

    The independent yardstick is tag overlap (the embeddings never saw the tags, and it shares no
    inputs with the reference lists coupling is built from). Defaults reproduce the presented lift
    table (tag >=2 vs coupling >=2,>=3,>=5).
    """

    tag_pairs = to_pairs(groundtruth.build_tag_positives(tag_sets, tag_min_shared))
    rows: List[dict] = []
    for threshold in coupling_thresholds:
        pairs = to_pairs(groundtruth.build_coupling_positives(dataset_path, ids, threshold))
        rows.append(
            {
                "coupling_min_shared": threshold,
                "tag_min_shared": tag_min_shared,
                **agreement(tag_pairs, pairs, len(ids)),
            }
        )
    return rows


def idf_agreement_sweep(  # pylint: disable=too-many-arguments
    dataset_path: str,
    ids: Sequence[str],
    tag_sets: Dict[str, Set[str]],
    *,
    tag_min_shared: int = 2,
    similarities: Sequence[float] = (0.05, 0.1, 0.2),
    weights: Dict[Tuple[str, str], float] = None,
) -> List[dict]:
    """
    Agreement + lift of tag overlap against IDF-weighted coupling at each cosine cutoff.

    The IDF graph down-weights ubiquitous references, so at a comparable pair count it concentrates
    far harder among tag-related pairs than count-coupling — i.e. much higher lift for the same graph
    size. Pass a precomputed `weights` to avoid recomputing the cosine graph per cutoff.
    """

    tag_pairs = to_pairs(groundtruth.build_tag_positives(tag_sets, tag_min_shared))
    graph = weights if weights is not None else groundtruth.build_coupling_weights(dataset_path, ids)
    rows: List[dict] = []
    for cutoff in similarities:
        pairs = {pair for pair, similarity in graph.items() if similarity >= cutoff}
        rows.append(
            {
                "coupling_cosine": cutoff,
                "tag_min_shared": tag_min_shared,
                **agreement(tag_pairs, pairs, len(ids)),
            }
        )
    return rows


# pylint: disable=too-many-arguments


def high_confidence_pairs(
    dataset_path: str,
    ids: Sequence[str],
    tag_sets: Dict[str, Set[str]],
    *,
    tag_min_shared: int = 2,
    coupling_flavor: str = "count",
    coupling_min_shared: int = 2,
    coupling_min_similarity: float = 0.1,
    coupling_weights: Dict[Tuple[str, str], float] = None,
) -> List[Pair]:
    """
    The high-confidence "genuinely related" core: pairs flagged by *both* tag overlap and coupling.

    Two signals with no shared inputs agreeing on a pair is far stronger evidence than either alone.
    `coupling_flavor` selects which coupling graph to intersect with — `"count"` (>= `coupling_min_shared`
    shared refs; defaults reproduce the ~1,000-pair core) or `"idf"` (cosine >= `coupling_min_similarity`).
    """

    tag_pairs = to_pairs(groundtruth.build_tag_positives(tag_sets, tag_min_shared))
    coupled = coupling_pairs(
        dataset_path,
        ids,
        flavor=coupling_flavor,
        min_shared=coupling_min_shared,
        min_similarity=coupling_min_similarity,
        weights=coupling_weights,
    )
    return sorted(tag_pairs & coupled)


# pylint: enable=too-many-arguments
