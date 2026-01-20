from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from frozendict import frozendict

from semhash.datamodels import DeduplicationResult, DuplicateRecord
from semhash.utils import Record, coerce_value, to_frozendict


def group_records_by_key(
    records: Sequence[dict[str, Any]],
    columns: Sequence[str],
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """
    Group records by exact match on columns, preserving first-occurrence order.

    :param records: Records to group.
    :param columns: Columns to use as grouping key.
    :return: Tuple of (deduplicated_records, items) where:
        - deduplicated_records: first record from each unique group
        - items: list of groups, each group is a list of exact duplicates
    """
    # Track buckets by key and preserve first-occurrence order
    buckets: dict[frozendict[str, Any], list[dict[str, Any]]] = {}
    order: list[frozendict[str, Any]] = []

    for record in records:
        key = to_frozendict(record, columns)
        bucket = buckets.get(key)
        if bucket is None:
            # First occurrence: create new bucket and track order
            buckets[key] = [record]
            order.append(key)
        else:
            # Duplicate: add to existing bucket
            bucket.append(record)

    # Reconstruct in first-occurrence order
    items = [buckets[k] for k in order]
    deduplicated_records = [bucket[0] for bucket in items]
    return deduplicated_records, items


def remove_exact_duplicates(
    records: Sequence[dict[str, Any]],
    columns: Sequence[str],
    reference_records: list[list[dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], list[dict[str, Any]]]]]:
    """
    Remove exact duplicates based on the hashable representation of each record.

    If reference_records is None, the function will only check for duplicates within the records list.

    :param records: A list of records to check for exact duplicates.
    :param columns: Columns to unpack.
    :param reference_records: A list of records to compare against. These are already unpacked
    :return: A list of deduplicated records and a list of duplicates.
    """
    deduplicated: list[dict[str, Any]] = []
    duplicates: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []

    column_set = set(columns)

    # Build seen set from reference_records (cross-dataset mode) or empty (single-dataset mode)
    seen: defaultdict[frozendict[str, Any], list[dict[str, Any]]] = defaultdict(list)
    if reference_records is not None:
        for record_set in reference_records:
            key = to_frozendict(record_set[0], column_set)
            seen[key] = list(record_set)

    for record in records:
        frozen_record = to_frozendict(record, column_set)
        if duplicated_records := seen.get(frozen_record):
            duplicates.append((record, duplicated_records))
        else:
            deduplicated.append(record)
            # Single-dataset mode: track this record for future comparisons
            if reference_records is None:
                seen[frozen_record].append(record)

    return deduplicated, duplicates


def prepare_records(
    records: Sequence[Record], columns: Sequence[str] | None
) -> tuple[list[dict[str, Any]], Sequence[str], bool]:
    """
    Validate and prepare records for processing.

    :param records: A list of records (strings or dictionaries).
    :param columns: Columns to use if records are dictionaries.
    :return: Tuple of (dict_records, columns, was_string).
    :raises ValueError: If records are empty.
    :raises ValueError: If columns are not provided for dictionary records.
    :raises ValueError: If dict record contains None values.
    :raises ValueError: If records are not homogeneous (mixed strings and dicts).
    """
    if len(records) == 0:
        raise ValueError("records must not be empty")

    if columns is None and isinstance(records[0], dict):
        raise ValueError("Columns must be specified when passing dictionaries.")

    # String path: convert to dicts with "text" column
    if isinstance(records[0], str):
        if not all(isinstance(r, str) for r in records):
            raise ValueError("All records must be strings when the first record is a string.")
        columns = ["text"]
        dict_records: list[dict[str, Any]] = [{"text": record} for record in records]
        was_string = True
    # Dict path: validate and coerce values
    else:
        if not all(isinstance(r, dict) for r in records):
            raise ValueError("All records must be dicts when the first record is a dict.")
        assert columns is not None

        # Coerce values: stringify primitives, keep complex types raw (for images, etc.)
        dict_records_typed: list[dict[str, Any]] = list(records)
        dict_records = []
        for record in dict_records_typed:
            # Start with a copy of the full record to preserve non-embedding fields
            coerced: dict[str, Any] = dict(record)
            # Then coerce only the embedding columns
            for column in columns:
                val = record.get(column)
                if val is None:
                    raise ValueError(f"Column '{column}' has None value in record {record}")
                coerced[column] = coerce_value(val)
            dict_records.append(coerced)
        was_string = False

    return dict_records, columns, was_string


def dict_to_string(record: dict[str, str], columns: Sequence[str]) -> str:
    r"""
    Turn a record into a single string.

    Uses self.columns to determine the order of the text segments.
    Each text is cleaned by replacing '\t' with ' '. The texts are then joined by '\t'.

    :param record: A record to unpack.
    :param columns: Columns to unpack.
    :return: A single string representation of the record.
    """
    return "\t".join(record.get(c, "").replace("\t", " ") for c in columns)


def map_deduplication_result_to_strings(result: DeduplicationResult, columns: Sequence[str]) -> DeduplicationResult:
    """Convert the record and duplicates in each DuplicateRecord back to strings if self.was_string is True."""
    deduplicated_str = [dict_to_string(r, columns) for r in result.selected]
    mapped = []
    for dup_rec in result.filtered:
        record_as_str = dict_to_string(dup_rec.record, columns)
        duplicates_as_str = [(dict_to_string(r, columns), score) for r, score in dup_rec.duplicates]
        mapped.append(
            DuplicateRecord(
                record=record_as_str,
                duplicates=duplicates_as_str,
                exact=dup_rec.exact,
            )
        )
    return DeduplicationResult(selected=deduplicated_str, filtered=mapped, threshold=result.threshold, columns=columns)


def add_scores_to_records(records: list[dict[str, str]]) -> list[tuple[dict[str, str], float]]:
    """Add scores to records and return a DeduplicationResult."""
    return [(record, 1.0) for record in records]
