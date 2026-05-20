"""
Main entry point for data quality checks for pre-annotation data quality checks stage.
This module orchestrates the execution of quality checks and report generation.

`@author`: DAShaikh10
"""

import asyncio
import json
import os
from typing import Any, Tuple

import streamlit as st

from src.utils import logger, resolve_path, write_json

from src.lib.quality import config
from src.lib.quality.pre import RawDataValidator


@st.cache_data(show_spinner=False)
def load_report(dataset_path: str) -> dict[str, Any]:
    """
    Load and cache the raw data Quality Check report.

    Args:
        dataset_path (str): The path to the dataset file to analyze.

    Returns:
        dict[str, Any]: The quality check report containing statistics and flagged records.
    """

    validator = RawDataValidator(dataset_path)
    return asyncio.run(validator.analyze())


def _render_counter_section(title: str, counter: dict[str, int]) -> None:
    """
    Render a counter section with progress bars.

    Args:
        title (str): The title of the section.
        counter (dict[str, int]): A mapping of labels to their counts to display in the section.
    """

    st.subheader(title)

    if not counter:
        st.caption("No entries.")
        return

    ordered_items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    max_count = max(count for _, count in ordered_items)

    for label, count in ordered_items[:10]:
        left, middle, right = st.columns([4, 1, 6])
        left.write(label.replace("_", " ").title())
        middle.metric("Count", count)
        right.progress(count / max_count if max_count else 0)

    if len(ordered_items) > 10:
        st.caption(f"Showing top 10 of {len(ordered_items)} entries.")


def _render_histogram_section(title: str, distribution: dict[str, int], x_label: str, y_label: str) -> None:
    """
    Render a histogram section.

    Args:
        title (str): The title of the section.
        distribution (dict[str, int]): A mapping of x values to their counts for the histogram.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
    """

    st.subheader(title)
    if distribution:
        chart_data = {int(k): v for k, v in distribution.items()}
        st.bar_chart(chart_data, x_label=x_label, y_label=y_label)
    else:
        st.caption("No entries.")


def _display_quality_report(dataset_path: str, dataset_label: str) -> None:
    """
    Display the quality check report for a given dataset.

    Args:
        dataset_path (str): Path to the dataset to analyze.
        dataset_label (str): Label for the dataset being displayed (e.g., "Original", "Cleaned").
    """

    if not os.path.exists(dataset_path):
        st.warning(f"{dataset_label} dataset not found at: {dataset_path}")
        return

    with st.spinner(f"Running quality checks on {dataset_label} dataset..."):
        report = load_report(dataset_path)

    st.markdown(f"**Viewing:** {dataset_label} dataset")

    summary = report.get("summary", {})
    summary_cols = st.columns(4)
    summary_cols[0].metric("Valid records", summary.get("valid_records", 0))
    summary_cols[1].metric("Missing paper fields", summary.get("records_with_missing_paper_fields", 0))
    summary_cols[2].metric("No references", summary.get("papers_with_no_references", 0))
    summary_cols[3].metric("Invalid JSON lines", summary.get("invalid_json_lines", 0))

    st.subheader("Coverage")
    coverage_cols = st.columns(3)
    percentages = report.get("percentages", {})
    coverage_cols[0].metric(
        "Missing paper fields %",
        f"{percentages.get('records_with_missing_paper_fields', 0)}%",
    )
    coverage_cols[1].metric(
        "No references %",
        f"{percentages.get('papers_with_no_references', 0)}%",
    )
    coverage_cols[2].metric(
        "References missing arXiv ID %",
        f"{percentages.get('papers_with_references_missing_arxiv_id', 0)}%",
    )

    _render_histogram_section(
        "Reference count distribution",
        report.get("reference_count_distribution", {}),
        "Number of references",
        "Record count",
    )
    _render_histogram_section(
        "Shared reference distribution",
        report.get("shared_reference_distribution", {}),
        "Number of papers citing the same reference",
        "Reference count",
    )

    _render_counter_section("Suspicious record counts", report.get("suspicious_record_counts", {}))
    _render_counter_section("Paper field missing counts", report.get("paper_field_missing_counts", {}))
    _render_counter_section("Reference missing counts", report.get("reference_missing_counts", {}))

    st.subheader("Flagged records")
    flagged = report.get("flagged_records", [])
    if flagged:
        st.dataframe(flagged[:100], width="stretch", hide_index=True)
    else:
        st.caption("No flagged records found.")

    with st.expander("Full report JSON"):
        st.json(report)


def _remove_records_with_no_references(input_path: str, output_path: str) -> Tuple[int, int]:
    """
    Remove records that have no references from a JSONL dataset file.

    Args:
        input_path: Path to the input JSONL dataset.
        output_path: Path to write the cleaned JSONL dataset.

    Returns:
        Tuple of (kept_count, removed_count).
    """

    kept = 0
    removed = 0

    write_path = output_path
    with open(input_path, "r", encoding="utf-8") as infile, open(write_path, "w", encoding="utf-8") as outfile:
        for raw in infile:
            line = raw.strip()
            if not line:
                continue

            rec = json.loads(line)
            refs = rec.get("references", [])
            if not isinstance(refs, list) or len(refs) == 0:
                removed += 1
                continue

            outfile.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    return kept, removed


def app() -> None:
    """
    Streamlit SPA to display data quality check results.
    """

    st.set_page_config(layout="wide")
    st.title("Pre-annotation Quality Check Dashboard")
    st.caption("Runs the dataset quality checks in memory and renders the results directly in Streamlit.")

    # Initialize session state for dataset tracking.
    if "original_dataset_path" not in st.session_state:
        current_dir = os.path.dirname(__file__)
        st.session_state.original_dataset_path = resolve_path(current_dir, config.ENRICHED_DATASET_FILE)
        st.session_state.cleaned_dataset_path = None
        st.session_state.active_dataset_for_display = "original"

    original_path = st.session_state.original_dataset_path
    cleaned_path = st.session_state.cleaned_dataset_path

    st.markdown(f"**Original dataset path:** {original_path}")

    if not os.path.exists(original_path):
        st.error("Original dataset file not found at the configured path.")
        return

    # Dataset selector widget (show only if cleaned dataset exists)
    if cleaned_path and os.path.exists(cleaned_path):
        dataset_options = ["Original dataset", "Cleaned dataset"]
        selected_dataset = st.radio(
            "Select dataset to analyze:",
            dataset_options,
            index=0 if st.session_state.active_dataset_for_display == "original" else 1,
            horizontal=True,
        )
        st.session_state.active_dataset_for_display = "cleaned" if "Cleaned" in selected_dataset else "original"
    else:
        st.session_state.active_dataset_for_display = "original"

    # Refresh analysis button.
    refresh_requested = st.button("Refresh analysis")
    if refresh_requested:
        load_report.clear()
        st.rerun()

    # Option to remove records with no references.
    st.subheader("Cleaning options")
    remove_no_refs_opt = st.checkbox(
        "Remove records with no references",
        value=False,
        help="Filter out dataset records that do not contain any references.",
    )

    if remove_no_refs_opt:
        st.caption("This will create a cleaned dataset without records that have no references.")
        if st.button("Run cleaning: remove no-reference records"):
            directory = os.path.dirname(original_path)
            filename = os.path.basename(original_path)
            base_name, ext = os.path.splitext(filename)
            output_path = os.path.join(directory, f"{base_name}_cleaned{ext}")

            kept_count, removed_count = _remove_records_with_no_references(original_path, output_path)

            st.success(f"Cleaning complete — kept {kept_count} records, removed {removed_count} records.")
            st.info(f"Cleaned dataset written to: {output_path}")

            # Update session state with cleaned dataset path and switch to it.
            st.session_state.cleaned_dataset_path = output_path
            st.session_state.active_dataset_for_display = "cleaned"

            load_report.clear()

            st.subheader("Cleaned Dataset Analysis")
            _display_quality_report(output_path, "Cleaned")
            st.rerun()
    else:
        st.divider()
        if st.session_state.active_dataset_for_display == "cleaned" and cleaned_path:
            _display_quality_report(cleaned_path, "Cleaned")
        else:
            _display_quality_report(original_path, "Original")


async def main() -> None:
    """
    Entrypoint for pre-annotation quality checks.
    """

    current_dir = os.path.dirname(__file__)
    dataset_path = resolve_path(current_dir, config.ENRICHED_DATASET_FILE)
    report_path = resolve_path(current_dir, config.PRE_ANNOTATION_REPORT_PATH)

    if not os.path.exists(dataset_path):
        logger.error("Dataset file not found at %s", dataset_path)
        return

    validator = RawDataValidator(dataset_path)
    report = await validator.analyze()

    if report_path:
        await write_json(report_path, report)

    logger.info(
        "Summary: %s records, %s with missing paper fields, %s with no references, %s invalid JSON lines",
        report["summary"]["valid_records"],
        report["summary"]["records_with_missing_paper_fields"],
        report["summary"]["papers_with_no_references"],
        report["summary"]["invalid_json_lines"],
    )

    if report_path:
        logger.info("Quality check report written to %s", report_path)


if __name__ == "__main__":
    asyncio.run(main())
    app()
