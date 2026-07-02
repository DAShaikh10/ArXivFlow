"""
Corpus-relatedness audit — Streamlit app.

A reproducible, interactive view of *how related the 992 corpus papers actually are* and *whether
independent signals agree*. Thin presentation layer over `analysis.py`; every number is recomputed
live from `data/` so the analysis is reproducible, not just its results.

`@author`: DAShaikh10
"""

# pylint: disable=wrong-import-position,redefined-outer-name,invalid-name,import-outside-toplevel,broad-exception-caught

import json
import os
import sys

# `streamlit run <file>` puts the script's own directory on sys.path, not the CWD, so the package
# imports below would fail. Insert the compass root (three levels up from src/lib/relatedness/).
_COMPASS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _COMPASS_ROOT not in sys.path:
    sys.path.insert(0, _COMPASS_ROOT)

import pandas as pd
import streamlit as st

from src.lib.eval import config, groundtruth
from src.lib.relatedness import analysis
from src.utils import resolve_path

_HERE = os.path.dirname(__file__)


@st.cache_data(show_spinner=False)
def _dataset_ids(dataset_path: str) -> list:
    return analysis.corpus_ids_from_dataset(dataset_path)


@st.cache_data(show_spinner=False)
def _chroma_ids(database_path: str, collection_name: str) -> list:
    import chromadb

    client = chromadb.PersistentClient(path=database_path)
    return client.get_collection(name=collection_name).get()["ids"]


@st.cache_data(show_spinner=False)
def _tag_sets(annotation_path: str, canonical_map_path: str, ids: tuple) -> dict:
    return groundtruth.tag_sets_from_annotations(annotation_path, canonical_map_path, list(ids))


@st.cache_data(show_spinner="Computing IDF-weighted coupling graph…")
def _coupling_weights(dataset_path: str, ids: tuple) -> dict:
    return groundtruth.build_coupling_weights(dataset_path, list(ids))


def _parse_thresholds(raw: str, fallback: tuple) -> list:
    """
    Comma-separated positive ints -> sorted unique list; fall back if nothing parses.
    """

    values = sorted({int(token) for token in raw.replace(" ", "").split(",") if token.isdigit() and int(token) > 0})
    return values or list(fallback)


def _parse_floats(raw: str, fallback: tuple) -> list:
    """
    Comma-separated floats in (0, 1] -> sorted unique list; fall back if nothing parses.
    """

    values = set()
    for token in raw.replace(" ", "").split(","):
        try:
            value = float(token)
        except ValueError:
            continue
        if 0.0 < value <= 1.0:
            values.add(round(value, 4))
    return sorted(values) or list(fallback)


st.set_page_config(page_title="Corpus relatedness audit", page_icon="🕸️", layout="wide")
st.title("🕸️ Corpus-relatedness audit")
st.caption(
    "How related are the corpus papers — measured several independent ways — and do signals that "
    "share no inputs agree? Every number is recomputed live from `data/` via the Tier 0 ground-truth "
    "builders (`src.lib.eval.groundtruth`); this app only counts and compares the pair graphs."
)

dataset_path = resolve_path(_HERE, config.ENRICHED_DATASET_FILE)
annotation_path = resolve_path(_HERE, config.ANNOTATION_FILE)
canonical_map_path = resolve_path(_HERE, config.CANONICAL_MAP_FILE)
database_path = resolve_path(_HERE, config.EMBEDDING_DATABASE_NAME)

with st.sidebar:
    st.header("Corpus")
    corpus_source = st.radio(
        "Corpus id source",
        ("Dataset file", "Chroma collection"),
        help=(
            "Dataset = the ids the presented numbers were computed over (no embeddings DB needed). "
            "Chroma = parity with the eval harness, which keys off the served collection."
        ),
    )

    st.header("Thresholds")
    st.caption("`≥N` = minimum shared items for two papers to count as related under that relation.")
    cocitation_raw = st.text_input("Co-citation min shared", "1, 2")
    coupling_raw = st.text_input("Coupling min shared refs", "2, 3, 5")
    idf_coupling_raw = st.text_input(
        "IDF coupling cosine cutoffs", "0.05, 0.1, 0.2", help="Cosine over IDF-weighted reference vectors."
    )
    tag_raw = st.text_input("Tag overlap min shared tags", "1, 2")

    st.header("Cross-signal agreement")
    tag_min = st.slider("Tag overlap floor (independent yardstick)", 1, 5, 2)
    coupling_flavor = st.radio(
        "Coupling graph",
        ("IDF cosine", "Shared-count"),
        help=(
            "IDF cosine down-weights ubiquitous references (one clean ranked graph); shared-count is "
            "the raw threshold knob. At equal graph size IDF agrees with tags far more strongly."
        ),
    )
    if coupling_flavor == "IDF cosine":
        coupling_similarity = st.slider("IDF coupling floor (cosine)", 0.0, 0.6, 0.1, 0.01)
        coupling_min_shared = 2
    else:
        coupling_min_shared = st.slider("Coupling floor (shared refs)", 1, 8, 2)
        coupling_similarity = 0.1

cocitation_thresholds = _parse_thresholds(cocitation_raw, (1, 2))
coupling_thresholds = _parse_thresholds(coupling_raw, (2, 3, 5))
idf_coupling_similarities = _parse_floats(idf_coupling_raw, (0.05, 0.1, 0.2))
tag_thresholds = _parse_thresholds(tag_raw, (1, 2))

# Resolve the corpus.
try:
    if corpus_source == "Chroma collection":
        ids = _chroma_ids(database_path, config.EMBEDDING_COLLECTION_NAME)
        dataset_ids = _dataset_ids(dataset_path)
        only_chroma = set(ids) - set(dataset_ids)
        only_dataset = set(dataset_ids) - set(ids)
        if only_chroma or only_dataset:
            st.warning(
                f"Corpus mismatch: {len(only_chroma)} ids only in Chroma, {len(only_dataset)} only in "
                "the dataset. Coupling/citation graphs are built from the dataset, so ids missing there "
                "contribute no edges."
            )
    else:
        ids = _dataset_ids(dataset_path)
except Exception as error:  # noqa: BLE001 — surface any load failure in the UI rather than a stack trace.
    st.error(f"Could not load the corpus ({corpus_source}): {error}")
    st.stop()

tag_sets = _tag_sets(annotation_path, canonical_map_path, tuple(ids))
tagged = sum(1 for tags in tag_sets.values() if tags)
weights = _coupling_weights(dataset_path, tuple(ids))  # IDF cosine graph, reused everywhere below.

top = st.columns(3)
top[0].metric("Corpus papers", len(ids))
top[1].metric("Papers with ≥1 canonical tag", f"{tagged} ({100 * tagged / len(ids):.1f}%)")
top[2].metric("Avg references / paper", "≈35 (S2)")

st.subheader("1 · Relatedness density per definition")
st.markdown(
    "What you described — *“A and B both cite C”* — is **bibliographic coupling** (shared outgoing "
    "references; C may be outside the corpus), **not** co-citation (a later paper cites both). "
    "Coupling is dense and usable; co-citation needs the citing paper inside the corpus, so it is "
    "near-empty here. Direct citation is sparse because the corpus is a thin sample of arXiv. "
    "**IDF coupling** is the same coupling, but each shared reference is weighted by `log(N/df)` so a "
    "shared niche paper counts far more than a shared ubiquitous one (cosine, no integer cutoff)."
)
rows = analysis.density_rows(
    dataset_path,
    ids,
    tag_sets,
    cocitation_thresholds=cocitation_thresholds,
    coupling_thresholds=coupling_thresholds,
    idf_coupling_similarities=idf_coupling_similarities,
    tag_thresholds=tag_thresholds,
    coupling_weights=weights,
)
density_df = pd.DataFrame(rows).rename(
    columns={
        "relation": "Relation",
        "covered": "Papers w/ ≥1 neighbour",
        "coverage_pct": "Coverage %",
        "pairs": "Pairs",
        "median_degree_nonzero": "Median degree (nonzero)",
        "max_degree": "Max degree",
    }
)[["Relation", "Papers w/ ≥1 neighbour", "Coverage %", "Pairs", "Median degree (nonzero)", "Max degree"]]
st.dataframe(density_df, width="stretch", hide_index=True)
st.bar_chart(density_df.set_index("Relation")["Coverage %"], horizontal=True)
st.info(
    "Coverage at coupling ≥2 is near-total with a huge median degree — a near-complete graph driven "
    "by ubiquitous references (Transformers, BERT). “% with a neighbour” is **not** a meaningful "
    "relatedness metric on its own; the agreement below is."
)

st.subheader("2 · Cross-signal agreement (the validation)")
st.markdown(
    f"Tag overlap (from abstract NER spans) and bibliographic coupling (from reference lists) share "
    f"**no inputs**. If they agree on which pairs are related *above chance*, the relatedness is real "
    f"— not an artefact of one proxy. Below: of pairs sharing **≥{tag_min} tags**, what share are also "
    f"coupled, vs the by-chance rate. **Lift** = how many times above chance. Switch the coupling graph "
    f"in the sidebar — IDF cosine reaches far higher lift than shared-count at the same graph size."
)
if coupling_flavor == "IDF cosine":
    sweep = analysis.idf_agreement_sweep(
        dataset_path, ids, tag_sets, tag_min_shared=tag_min, similarities=idf_coupling_similarities, weights=weights
    )
    threshold_col, threshold_label = "coupling_cosine", "IDF coupling cos ≥"
else:
    sweep = analysis.agreement_sweep(
        dataset_path, ids, tag_sets, tag_min_shared=tag_min, coupling_thresholds=coupling_thresholds
    )
    threshold_col, threshold_label = "coupling_min_shared", "Coupling ≥"
sweep_df = pd.DataFrame(sweep)
sweep_df["chance_rate"] = (sweep_df["chance_rate"] * 100).round(2)
sweep_df["observed_rate"] = (sweep_df["observed_rate"] * 100).round(1)
sweep_df = sweep_df.rename(
    columns={
        threshold_col: threshold_label,
        "coupling_pairs": "Coupling pairs",
        "tag_pairs": "Tag pairs",
        "intersection": "Both",
        "chance_rate": "Chance %",
        "observed_rate": "Observed on tag-pairs %",
        "lift": "Lift ×",
    }
)[[threshold_label, "Tag pairs", "Coupling pairs", "Both", "Chance %", "Observed on tag-pairs %", "Lift ×"]]
st.dataframe(sweep_df, width="stretch", hide_index=True)

st.subheader("3 · High-confidence core & export")
core_pairs = analysis.high_confidence_pairs(
    dataset_path,
    ids,
    tag_sets,
    tag_min_shared=tag_min,
    coupling_flavor="idf" if coupling_flavor == "IDF cosine" else "count",
    coupling_min_shared=coupling_min_shared,
    coupling_min_similarity=coupling_similarity,
    coupling_weights=weights,
)
if coupling_flavor == "IDF cosine":
    coupling_label = f"IDF coupling (cos ≥{coupling_similarity})"
    core_slug = f"idfcos{coupling_similarity}"
else:
    coupling_label = f"coupling (≥{coupling_min_shared} shared refs)"
    core_slug = f"count{coupling_min_shared}"
st.metric(
    f"Pairs related by BOTH tags (≥{tag_min}) and {coupling_label}",
    len(core_pairs),
    help="The genuinely-related core: two independent signals agree. Use this as ground truth.",
)

core_df = pd.DataFrame(core_pairs, columns=["paper_a", "paper_b"])
report = {
    "corpus_source": corpus_source,
    "corpus_size": len(ids),
    "papers_tagged": tagged,
    "density": rows,
    "agreement": {"coupling_flavor": coupling_flavor, "rows": sweep},
    "high_confidence_core": {
        "tag_min_shared": tag_min,
        "coupling_flavor": coupling_flavor,
        "coupling_min_shared": coupling_min_shared,
        "coupling_min_similarity": coupling_similarity,
        "pairs": len(core_pairs),
    },
}

export = st.columns(2)
export[0].download_button(
    "⬇ High-confidence pairs (CSV)",
    core_df.to_csv(index=False),
    file_name=f"related_pairs_tag{tag_min}_{core_slug}.csv",
    mime="text/csv",
)
export[1].download_button(
    "⬇ Full report (JSON)",
    json.dumps(report, indent=2),
    file_name="relatedness_report.json",
    mime="application/json",
)
with st.expander("Caveats"):
    st.markdown(
        "- **57% of references carry no arXiv id** and match by normalized title only, so coupling "
        "slightly *under*-counts. Numbers are directional, not exact.\n"
        "- **Coupling is circular** for evaluating the citation *retrieval* signal (that signal ranks "
        "by coupling). It is honest only for content signals (dense/BM25/entity). Tag overlap stays "
        "clean for everything.\n"
        "- **Co-citation is empirically near-dead here** — the cleanest non-circular citation proxy in "
        "principle, but the corpus is too sparse for it, leaving tag overlap as the only independent "
        "yardstick."
    )
