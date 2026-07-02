"""
Build proxy ground-truth positive sets for Tier 0 evaluation.

Two independent signals:

  * Tag overlap   — canonical NER-tag Jaccard >= threshold. Independent of the embeddings (the model
                    never saw the tags), so this is the honest headline test.
  * Citation overlap — undirected direct citation between corpus papers, matched on version-stripped
                    arXiv ids. SPECTER2 is trained on citation proximity, so this is partly
                    in-distribution; reported as a secondary signal.

`@author`: DAShaikh10
"""

import json
import math
import re
import unicodedata
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Trailing arXiv version suffix ("2007.10310v3" -> "2007.10310"). Paper ids carry a version; scraped
# reference ids do not, so the two only match once it is stripped. Mirrors api/store._normalize_id.
_VERSION_SUFFIX = re.compile(r"v\d+$")

# Label Studio label name -> canonical_map.json category key. Mirrors papervec's tags.LABEL_TO_FIELD,
# so the tokens we build here match the canonical ids papervec would have written.
_LABEL_TO_FIELD: Dict[str, str] = {
    "Target NLP Task": "target_nlp_task",
    "Machine Learning Architecture": "machine_learning_architecture",
    "Training or Fine-tuning Method": "training_method",
    "Dataset or Benchmark Name": "dataset_name",
    "Application Domain": "application_domain",
    "Evaluation Metric": "evaluation_metric",
    "Language or Dialect": "language_dialect",
}

# Wrapping characters stripped from a raw span; kept identical to papervec's normalize.
_WRAP_CHARS = " \t\n\r\"'`“”‘’()[]{}.,;:"


def normalize_id(arxiv_id: Optional[str]) -> Optional[str]:
    """
    Strip the version suffix so a versioned paper id and an unversioned reference id compare equal.
    """

    if not arxiv_id:
        return None
    return _VERSION_SUFFIX.sub("", arxiv_id.strip())


def _normalize_span(text: str) -> str:
    """
    NFKC + lowercase + whitespace-collapse + wrapping-punctuation strip (papervec's normalize).
    """

    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(_WRAP_CHARS)


def _load_records(path: str) -> List[dict]:
    """
    Load a Label Studio export: a JSON array, a single JSON object, or true JSON Lines.
    """

    with open(path, "r", encoding="utf-8") as handle:
        head = handle.read(64).lstrip()
        handle.seek(0)
        if head.startswith("[") or head.startswith("{"):
            data = json.load(handle)
            return data if isinstance(data, list) else [data]
        return [json.loads(line) for line in handle if line.strip()]


def tag_sets_from_annotations(  # pylint: disable=too-many-locals
    annotation_path: str,
    canonical_map_path: str,
    corpus_ids: Sequence[str],
    blocks: Iterable[str] = ("annotations",),
) -> Dict[str, Set[str]]:
    """
    Per-paper canonical tag set, built from a Label Studio export (not the Chroma metadata, which the
    deployed embeddings store never had baked in).

    Each "{field}:{canonical}" token is the same identity papervec would have written, so two papers
    match only on the same canonical value in the same field. `blocks` selects which result blocks to
    read: the gold ground truth uses `("annotations",)` (human only, the default); an LLM serving set
    stores its spans under `("predictions",)`. Only papers present in the corpus are kept.
    """

    with open(canonical_map_path, "r", encoding="utf-8") as handle:
        canonical_map: Dict[str, Dict[str, str]] = json.load(handle)

    norm_to_id = {normalize_id(pid): pid for pid in corpus_ids if normalize_id(pid)}

    tag_sets: Dict[str, Set[str]] = defaultdict(set)
    for record in _load_records(annotation_path):
        corpus_id = norm_to_id.get(normalize_id(record.get("data", {}).get("arxiv_id")))
        if corpus_id is None:
            continue
        result_blocks = [block for key in blocks for block in record.get(key, [])]
        for block in result_blocks:
            if not isinstance(block, dict):
                continue
            for item in block.get("result", []):
                value = item.get("value", {})
                labels = value.get("labels") or []
                raw = value.get("text", "")
                if not labels or not raw:
                    continue
                field = _LABEL_TO_FIELD.get(labels[0])
                if field is None:
                    continue
                norm = _normalize_span(raw)
                if not norm:
                    continue
                canonical = canonical_map.get(field, {}).get(norm, norm)
                tag_sets[corpus_id].add(f"{field}:{canonical}")

    # Every corpus paper should have an entry so coverage/denominators are over the full corpus.
    return {pid: tag_sets.get(pid, set()) for pid in corpus_ids}


def build_tag_positives(
    tag_sets: Dict[str, Set[str]],
    min_shared: int,
    min_jaccard: float = 0.0,
) -> Dict[str, Set[str]]:
    """
    Undirected tag-overlap positives.

    A pair is mutually relevant when it shares at least `min_shared` distinct canonical tags AND its
    tag-set Jaccard is >= `min_jaccard`. The shared-count floor is the primary guard: with sparse
    annotations, a single shared *generic* tag ("transformer", "english") gives a high Jaccard but no
    real relatedness, so requiring two distinct shared tags filters those coincidences out. `min_jaccard`
    defaults to 0 (off) and is available only as an extra tightener. Only tagged papers participate.
    """

    tagged = [paper_id for paper_id, tags in tag_sets.items() if tags]
    positives: Dict[str, Set[str]] = defaultdict(set)

    for i, a in enumerate(tagged):
        tags_a = tag_sets[a]
        for b in tagged[i + 1 :]:
            tags_b = tag_sets[b]
            intersection = len(tags_a & tags_b)
            if intersection < min_shared:
                continue
            if min_jaccard > 0.0 and intersection / len(tags_a | tags_b) < min_jaccard:
                continue
            positives[a].add(b)
            positives[b].add(a)
    return positives


def build_citation_positives(dataset_path: str, corpus_ids: Sequence[str]) -> Dict[str, Set[str]]:
    """
    Undirected direct-citation positives between corpus papers.

    Reference ids are unversioned, so both endpoints are resolved through a version-stripped index to
    the canonical (versioned) corpus id. References that point outside the corpus are dropped.
    """

    norm_to_id: Dict[str, str] = {}
    for paper_id in corpus_ids:
        norm = normalize_id(paper_id)
        if norm:
            norm_to_id[norm] = paper_id

    positives: Dict[str, Set[str]] = defaultdict(set)
    with open(dataset_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            paper = json.loads(line)
            source = norm_to_id.get(normalize_id(paper.get("arxiv_id")) or "")
            if source is None:
                continue
            for ref in paper.get("references") or []:
                target = norm_to_id.get(normalize_id(ref.get("arxiv_id")) or "")
                if target and target != source:
                    positives[source].add(target)
                    positives[target].add(source)
    return positives


def build_cocitation_positives(  # pylint: disable=too-many-locals
    dataset_path: str,
    corpus_ids: Sequence[str],
    min_shared: int = 1,
) -> Dict[str, Set[str]]:
    """
    Undirected co-citation positives: two corpus papers are related when at least `min_shared` corpus
    papers cite *both* of them.

    Co-citation is the incoming-citation dual of bibliographic coupling (shared outgoing references) and
    is distinct from direct citation. No retrieval signal uses it — the citation signal is built from
    coupling + direct citation — so it is the cleanest available *non-circular* citation-family ground
    truth for honestly scoring fusion and reranking. (It is still a citation-derived proxy, not human
    relevance; SPECTER2 is citation-trained, so dense is expected to do well on it.)

    Each citing paper contributes a co-citation to every pair within the set of corpus papers it cites.
    """

    norm_to_id: Dict[str, str] = {}
    for paper_id in corpus_ids:
        norm = normalize_id(paper_id)
        if norm:
            norm_to_id[norm] = paper_id

    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with open(dataset_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            paper = json.loads(line)
            cited = sorted(
                {
                    target
                    for ref in paper.get("references") or []
                    if (target := norm_to_id.get(normalize_id(ref.get("arxiv_id")) or "")) is not None
                }
            )
            for i, first in enumerate(cited):
                for second in cited[i + 1 :]:
                    counts[first][second] += 1
                    counts[second][first] += 1

    positives: Dict[str, Set[str]] = defaultdict(set)
    for first, neighbours in counts.items():
        for second, shared in neighbours.items():
            if shared >= min_shared:
                positives[first].add(second)
    return positives


def _reference_keys(paper: dict) -> Set[str]:
    """
    Reference identities for one paper: version-stripped arXiv id, else normalized title. External
    references are kept — coupling does not require the cited paper to be in the corpus.
    """

    keys: Set[str] = set()
    for ref in paper.get("references") or []:
        norm = normalize_id(ref.get("arxiv_id"))
        if norm:
            keys.add(f"id:{norm}")
        elif ref.get("title"):
            title = _normalize_span(ref["title"])
            if title:
                keys.add(f"t:{title}")
    return keys


# pylint: disable=too-many-locals


def build_coupling_positives(
    dataset_path: str,
    corpus_ids: Sequence[str],
    min_shared: int = 2,
) -> Dict[str, Set[str]]:
    """
    Undirected bibliographic-coupling positives: two corpus papers are related when they share at least
    `min_shared` references — *and the shared reference need not be in the corpus*. This is the original
    "A and B both cite C" idea; C is usually an external seminal paper.

    High coverage (most papers share references), so a far stronger yardstick than co-citation. The
    `min_shared >= 2` floor guards against a single ubiquitously-cited paper (e.g. BERT) linking everything.
    NOTE: this is the exact relation the *citation retrieval signal* ranks by, so it is **circular** for
    that signal (and any fusion containing it) — it honestly scores only the content signals (dense, BM25,
    entity) and content-only fusion.
    """

    norm_to_id = {normalize_id(pid): pid for pid in corpus_ids if normalize_id(pid)}

    inverted: Dict[str, List[str]] = defaultdict(list)
    with open(dataset_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            paper = json.loads(line)
            corpus_id = norm_to_id.get(normalize_id(paper.get("arxiv_id")) or "")
            if corpus_id is None:
                continue
            for key in _reference_keys(paper):
                inverted[key].append(corpus_id)

    shared_counts: Dict[tuple, int] = defaultdict(int)
    for papers in inverted.values():
        if len(papers) < 2:
            continue
        for i, first in enumerate(papers):
            for second in papers[i + 1 :]:
                shared_counts[(first, second) if first < second else (second, first)] += 1

    positives: Dict[str, Set[str]] = defaultdict(set)
    for (first, second), shared in shared_counts.items():
        if shared >= min_shared:
            positives[first].add(second)
            positives[second].add(first)
    return positives


def _corpus_reference_keys(dataset_path: str, corpus_ids: Sequence[str]) -> Dict[str, Set[str]]:
    """
    Per corpus paper, its set of reference identities (see `_reference_keys`). External references
    kept — coupling does not require the cited paper to be in the corpus.
    """

    norm_to_id = {normalize_id(pid): pid for pid in corpus_ids if normalize_id(pid)}
    paper_refs: Dict[str, Set[str]] = {}
    with open(dataset_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            paper = json.loads(line)
            corpus_id = norm_to_id.get(normalize_id(paper.get("arxiv_id")) or "")
            if corpus_id is not None:
                paper_refs[corpus_id] = _reference_keys(paper)
    return paper_refs


def build_coupling_weights(dataset_path: str, corpus_ids: Sequence[str]) -> Dict[Tuple[str, str], float]:
    """
    IDF-weighted bibliographic-coupling similarity for every corpus pair that shares >=1 reference:
    cosine similarity over IDF-weighted reference vectors (binary term frequency — a paper either
    cites a reference or not).

    Each reference key `r` carries weight ``idf(r) = log(N / df(r))`` where ``df`` is the number of
    corpus papers citing it: a reference cited by *many* corpus papers (e.g. "Attention Is All You
    Need") gets a near-zero weight, a niche shared reference a large one. A reference cited by every
    corpus paper has ``idf = 0`` and drops out entirely. The pair score is

        sum_{r in A ∩ B} idf(r)^2  /  (||A|| * ||B||)        ||A|| = sqrt(sum_{r in A} idf(r)^2)

    i.e. a single continuous, bounded [0, 1] ranked graph that replaces the shared-count threshold
    knob of `build_coupling_positives`: two papers sharing one obscure reference outrank two sharing
    one ubiquitous reference, with no arbitrary integer cutoff. (Same circularity caveat as plain
    coupling: this still derives from reference lists, so it is in-distribution for the citation
    retrieval signal and honest only for the content signals.)
    """

    paper_refs = _corpus_reference_keys(dataset_path, corpus_ids)
    n_papers = len(corpus_ids)

    document_frequency: Dict[str, int] = defaultdict(int)
    for refs in paper_refs.values():
        for key in refs:
            document_frequency[key] += 1
    idf = {key: math.log(n_papers / count) for key, count in document_frequency.items()}

    # L2 norm of each paper's IDF-weighted reference vector (over all its references, shared or not).
    norms = {pid: math.sqrt(sum(idf[key] ** 2 for key in refs)) for pid, refs in paper_refs.items()}

    # Inverted index, then accumulate the shared-reference dot product per pair. References cited by
    # fewer than two corpus papers (or with idf 0) can never contribute and are skipped.
    inverted: Dict[str, List[str]] = defaultdict(list)
    for pid, refs in paper_refs.items():
        for key in refs:
            inverted[key].append(pid)

    dot: Dict[Tuple[str, str], float] = defaultdict(float)
    for key, papers in inverted.items():
        weight = idf[key] ** 2
        if len(papers) < 2 or weight == 0.0:
            continue
        for i, first in enumerate(papers):
            for second in papers[i + 1 :]:
                dot[(first, second) if first < second else (second, first)] += weight

    weights: Dict[Tuple[str, str], float] = {}
    for (first, second), product in dot.items():
        denominator = norms[first] * norms[second]
        if denominator > 0.0:
            weights[(first, second)] = product / denominator
    return weights


# pylint: enable=too-many-locals


def positives_from_weights(
    weights: Dict[Tuple[str, str], float],
    min_similarity: float,
) -> Dict[str, Set[str]]:
    """
    Threshold a weighted pair graph into an undirected positive set (one cosine cutoff, no recompute).
    """

    positives: Dict[str, Set[str]] = defaultdict(set)
    for (first, second), similarity in weights.items():
        if similarity >= min_similarity:
            positives[first].add(second)
            positives[second].add(first)
    return positives


def build_idf_coupling_positives(
    dataset_path: str,
    corpus_ids: Sequence[str],
    min_similarity: float = 0.1,
) -> Dict[str, Set[str]]:
    """
    Convenience wrapper: IDF-weighted coupling positives at a single cosine-similarity cutoff.
    """

    return positives_from_weights(build_coupling_weights(dataset_path, corpus_ids), min_similarity)
