"""
@Author: DAShaikh10
@Description: Raw dataset quality checks for the enriched dataset.
"""

import json
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import aiofiles

from src.lib.quality import config


# pylint: disable=too-many-instance-attributes
class RawDataValidator:
    """
    Validates and generates quality check report for raw datasets.
    """

    ARXIV_ID_PATTERN = re.compile(r"^\d{4}\.\d{5}(v\d+)?$")
    HTTP_URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)
    PAPER_REQUIRED_FIELDS = ("arxiv_id", "title", "abstract", "published_date", "url", "influential_citations")
    WHITESPACE_PATTERN = re.compile(r"\s+")

    def __init__(self, dataset_path: str) -> None:
        self.dataset_path = dataset_path
        self.flagged_records: list[dict[str, Any]] = []
        self.global_reference_frequency: Counter[str] = Counter()
        self.paper_missing_fields = Counter()
        self.reference_count_distribution = Counter()
        self.reference_missing_fields = Counter()
        self.seen_arxiv_ids: set[str] = set()
        self.seen_titles: set[str] = set()
        self.stats = Counter()
        self.suspicious_reasons = Counter()

    @staticmethod
    def _is_missing(value: Any) -> bool:
        """
        Returns true when a value should be treated as missing.
        """

        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, (list, tuple, set, dict)):
            return not value

        return False

    @staticmethod
    def _percentage(count: int, total: int) -> float:
        """
        Compute a percentage while avoiding division by zero.
        """

        return 0.0 if total == 0 else round((count / total) * 100, 2)

    def _normalize_text(self, value: str) -> str:
        """
        Normalize text for comparisons by removing extra whitespace and converting to lowercase.

        Args:
            value (str): The text value to normalize.

        Returns:
            str: The normalized text.
        """

        return self.WHITESPACE_PATTERN.sub(" ", str(value).strip()).lower()

    def _parse_jsonl(self, raw_line: str) -> dict:
        """
        Parse a single line of JSONL and return the record dictionary.
        Updates stats. for blank lines and invalid JSON.

        Args:
            raw_line (str): The raw line string from the dataset file.

        Returns:
            dict: The parsed record dictionary, or an empty dict if parsing failed.
        """

        line = raw_line.strip()
        if not line:
            self.stats["blank_lines"] += 1
            return {}

        try:
            record = dict(json.loads(line))
        except (json.JSONDecodeError, ValueError, TypeError) as error:
            self.stats["invalid_json_lines"] += 1
            self.flagged_records.append(
                {
                    "line_number": self.stats["lines_read"],
                    "issues": ["invalid_json"],
                    "error": str(error),
                }
            )
            return {}

        return record

    def _process_references(self, references: list[Any]) -> tuple[int, bool, set[str]]:
        """
        Process a list of references for a single paper. Check the following:
        - Each reference is a dict with expected fields.
        - Count duplicate references within the same paper (based on strict matching).

        Args:
            references (list): The list of reference entries from the paper record.

        Returns:
            (duplicate_reference_count, paper_has_reference_arxiv_id, seen_global_keys)
        """

        duplicate_reference_count = 0
        seen_strict_keys: set[tuple[str, str, str]] = set()
        seen_global_keys: set[str] = set()
        paper_has_reference_arxiv_id = False

        for reference in references:
            if not isinstance(reference, dict):
                self.stats["malformed_references"] += 1
                continue

            self.stats["reference_total"] += 1

            reference_arxiv_id = str(reference.get("arxiv_id", "")).strip()
            reference_title = str(reference.get("title", "")).strip()
            reference_url = str(reference.get("url", "")).strip()

            if self._is_missing(reference_title):
                self.reference_missing_fields["title"] += 1
            if self._is_missing(reference_arxiv_id):
                self.reference_missing_fields["arxiv_id"] += 1
            else:
                paper_has_reference_arxiv_id = True
            if self._is_missing(reference_url):
                self.reference_missing_fields["url"] += 1

            normalized_arxiv_id = self._normalize_text(reference_arxiv_id)
            normalized_title = self._normalize_text(reference_title)
            normalized_url = self._normalize_text(reference_url)

            # Strict key for detecting intra-paper duplicates (matches previous exact behavior)
            strict_key = (normalized_title, normalized_arxiv_id, normalized_url)
            if strict_key in seen_strict_keys:
                duplicate_reference_count += 1
            else:
                seen_strict_keys.add(strict_key)

            # Relaxed key for global shared citations distribution
            if normalized_arxiv_id:
                global_key = f"arxiv:{normalized_arxiv_id}"
            elif normalized_title:
                global_key = f"title:{normalized_title}"
            elif normalized_url:
                global_key = f"url:{normalized_url}"
            else:
                global_key = f"empty:{id(reference)}"

            seen_global_keys.add(global_key)

        return duplicate_reference_count, paper_has_reference_arxiv_id, seen_global_keys

    def _arxiv_id_check(self, record: dict[str, Any], issues: list[str]) -> tuple[str, list[str]]:
        """
        Check the presence of `arxiv_id` field and validate the following:
        - Uniqueness of arxiv_id across records. There should never be duplicate `arxiv_ids` records in the dataset.
        - Format of arxiv_id matches expected pattern.

        Updates stats and issues list accordingly.

        Args:
            record (dict): The paper record being processed.
            issues (list): The list of issues identified for the current record, to be updated.

        Returns:
            tuple: (arxiv_id, updated_issues)
        """

        arxiv_id = str(record.get("arxiv_id", "")).strip()
        if arxiv_id:
            if arxiv_id in self.seen_arxiv_ids:
                self.stats["duplicate_arxiv_ids"] += 1
                issues.append("duplicate_arxiv_id")
            else:
                self.seen_arxiv_ids.add(arxiv_id)

            if not self.ARXIV_ID_PATTERN.match(arxiv_id):
                self.suspicious_reasons["invalid_arxiv_id_format"] += 1
                issues.append("invalid_arxiv_id_format")
        else:
            self.suspicious_reasons["missing_arxiv_id"] += 1

        return arxiv_id, issues

    def _reference_list_check(self, record: dict[str, Any], issues: list[str]) -> tuple[int, int, set[str], list[str]]:
        """
        Validate the `references` field of a single paper record.

        This method performs several checks:
        1. Verifies that the `references` field exists and is a list.
        2. Checks if the paper contains zero references.
        3. Delegates individual reference validation to `_process_references` to check
           for missing fields and intra-paper duplicates based on strict matching.
        4. Flags if the paper has references but absolutely none contain an `arxiv_id`.
        5. Updates the provided `issues` list with any discovered structural or duplicate issues.

        Args:
            record (dict): The paper record dictionary being processed.
            issues (list): The current list of identified string issues for this record.

        Returns:
            tuple: A 4-tuple containing:
                - references_len (int): Total number of references in the paper.
                - duplicate_reference_count (int): Number of intra-paper duplicate references found.
                - seen_reference_keys (set[str]): Set of relaxed reference keys for calculating global shared citations.
                - updated_issues (list[str]): The updated list of issues, potentially containing
                  'references_not_list', 'no_references', or 'duplicate_reference'.
        """

        references = record.get("references", [])
        if not isinstance(references, list):
            self.stats["references_not_list"] += 1
            issues.append("references_not_list")
            references = []

        references_len = len(references)

        if not references:
            self.stats["papers_with_no_references"] += 1
            issues.append("no_references")

        duplicate_reference_count, paper_has_reference_arxiv_id, seen_reference_keys = self._process_references(
            references
        )

        if references and not paper_has_reference_arxiv_id:
            self.stats["papers_with_references_missing_arxiv_id"] += 1

        if duplicate_reference_count > 0:
            issues.extend(["duplicate_reference"] * duplicate_reference_count)

        return references_len, duplicate_reference_count, seen_reference_keys, issues

    def _title_check(self, record: dict[str, Any], issues: list[str]) -> tuple[str, list[str]]:
        """
        Check the presence and length of the `title` field.
        Updates stats and issues list accordingly.

        Args:
            record (dict): The paper record being processed.
            issues (list): The list of issues identified for the current record, to be updated.

        Returns:
            tuple: (title, updated_issues)
        """

        title = str(record.get("title", "")).strip()
        if title:
            normalized_title = self._normalize_text(title)
            if normalized_title in self.seen_titles:
                self.stats["duplicate_titles"] += 1
                issues.append("duplicate_title")
            else:
                self.seen_titles.add(normalized_title)
        else:
            self.suspicious_reasons["empty_title"] += 1

        return title, issues

    def _process_record(self, record: dict[str, Any]) -> tuple[str, int, list[str], str, int, bool, set[str]]:
        """
        Process a single record checks and update counters.

        Returns: (issues, arxiv_id, title, references_len, duplicate_reference_count,
                  paper_field_missing_this_record, seen_reference_keys)
        """

        issues: list[str] = []
        paper_field_missing_this_record = False

        # Field presence checks.
        for field_name in self.PAPER_REQUIRED_FIELDS:
            if self._is_missing(record.get(field_name)):
                self.paper_missing_fields[field_name] += 1
                issues.append(f"missing_{field_name}")
                paper_field_missing_this_record = True

        # Individual field checks.
        abstract = str(record.get("abstract", "")).strip()
        if abstract and len(abstract.split()) < config.PRE_ANNOTATION_MIN_ABSTRACT_WORDS:
            self.suspicious_reasons["short_abstract"] += 1
            issues.append("short_abstract")

        arxiv_id, issues = self._arxiv_id_check(record, issues)

        references_len, duplicate_reference_count, seen_reference_keys, issues = self._reference_list_check(
            record, issues
        )

        title, issues = self._title_check(record, issues)

        url = str(record.get("url", "")).strip()
        if url and not self.HTTP_URL_PATTERN.match(url):
            self.suspicious_reasons["broken_url"] += 1
            issues.append("broken_url")

        return (
            arxiv_id,
            duplicate_reference_count,
            issues,
            title,
            references_len,
            paper_field_missing_this_record,
            seen_reference_keys,
        )

    async def analyze(self, encoding: str = "utf-8") -> dict[str, Any]:
        """
        Run the quality checks analysis on the raw dataset and return a report dictionary.

        Args:
            encoding (str): The file encoding to use when reading the dataset file.

        Returns:
            dict: The quality check report containing summary statistics and flagged records.
        """

        async with aiofiles.open(self.dataset_path, "r", encoding=encoding) as dataset_file:
            async for raw_line in dataset_file:
                # Record reading each line.
                self.stats["lines_read"] += 1

                record = self._parse_jsonl(raw_line)
                if not record:
                    continue

                # Record reading each valid record (even if it has issues) for percentage calculations.
                self.stats["valid_records"] += 1

                (
                    arxiv_id,
                    rec_dup_count,
                    issues,
                    title,
                    references_len,
                    paper_field_missing_this_record,
                    seen_reference_keys,
                ) = self._process_record(record)

                self.stats["duplicate_references"] += rec_dup_count
                self.reference_count_distribution[str(references_len)] += 1

                for ref_key in seen_reference_keys:
                    self.global_reference_frequency[ref_key] += 1

                if paper_field_missing_this_record:
                    self.stats["records_with_missing_paper_fields"] += 1

                if issues:
                    self.flagged_records.append(
                        {
                            "line_number": self.stats["lines_read"],
                            "arxiv_id": arxiv_id or None,
                            "title": title or None,
                            "issues": issues,
                        }
                    )

        return self.generate_report()

    def generate_report(self) -> dict[str, Any]:
        """
        Generate the raw data Quality Check report dictionary from collected statistics.
        """

        citation_counts = Counter(self.global_reference_frequency.values())
        shared_reference_distribution = {str(k): v for k, v in citation_counts.items()}

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_path": self.dataset_path,
            "thresholds": {
                "minimum_abstract_words": config.PRE_ANNOTATION_MIN_ABSTRACT_WORDS,
            },
            "summary": {
                "lines_read": self.stats["lines_read"],
                "blank_lines": self.stats["blank_lines"],
                "invalid_json_lines": self.stats["invalid_json_lines"],
                "valid_records": self.stats["valid_records"],
                "records_with_missing_paper_fields": self.stats["records_with_missing_paper_fields"],
                "papers_with_no_references": self.stats["papers_with_no_references"],
                "papers_with_references_missing_arxiv_id": self.stats["papers_with_references_missing_arxiv_id"],
                "duplicate_arxiv_ids": self.stats["duplicate_arxiv_ids"],
                "duplicate_titles": self.stats["duplicate_titles"],
                "references_not_list": self.stats["references_not_list"],
                "malformed_references": self.stats["malformed_references"],
                "reference_total": self.stats["reference_total"],
                "duplicate_references": self.stats["duplicate_references"],
            },
            "percentages": {
                "records_with_missing_paper_fields": self._percentage(
                    self.stats["records_with_missing_paper_fields"], self.stats["valid_records"]
                ),
                "papers_with_no_references": self._percentage(
                    self.stats["papers_with_no_references"], self.stats["valid_records"]
                ),
                "papers_with_references_missing_arxiv_id": self._percentage(
                    self.stats["papers_with_references_missing_arxiv_id"], self.stats["valid_records"]
                ),
            },
            "paper_field_missing_counts": dict(self.paper_missing_fields),
            "reference_missing_counts": dict(self.reference_missing_fields),
            "suspicious_record_counts": dict(self.suspicious_reasons),
            "reference_count_distribution": dict(self.reference_count_distribution),
            "shared_reference_distribution": shared_reference_distribution,
            "flagged_records": self.flagged_records,
        }


# pylint: enable=too-many-instance-attributes
