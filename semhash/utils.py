from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Protocol, TypeAlias, TypeVar

import numpy as np
from frozendict import frozendict

# Type definitions
Record = TypeVar("Record", str, dict[str, Any])
DuplicateList: TypeAlias = list[tuple[Record, float]]


class Encoder(Protocol):
    """An encoder protocol for SemHash."""

    def encode(
        self,
        sentences: list[str] | str | Sequence[str],
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Encode a list of sentences into embeddings.

        :param sentences: A list of sentences to encode.
        :param **kwargs: Additional keyword arguments.
        :return: The embeddings of the sentences.
        """
        ...  # pragma: no cover


def to_frozendict(record: dict[str, str], columns: set[str]) -> frozendict[str, str]:
    """Convert a record to a frozendict."""
    return frozendict({k: record.get(k, "") for k in columns})


def compute_candidate_limit(
    total: int,
    selection_size: int,
    fraction: float = 0.1,
    min_candidates: int = 100,
    max_candidates: int = 1000,
) -> int:
    """
    Compute the 'auto' candidate limit based on the total number of records.

    :param total: Total number of records.
    :param selection_size: Number of representatives to select.
    :param fraction: Fraction of total records to consider as candidates.
    :param min_candidates: Minimum number of candidates.
    :param max_candidates: Maximum number of candidates.
    :return: Computed candidate limit.
    """
    # 1) fraction of total
    limit = int(total * fraction)
    # 2) ensure enough to pick selection_size
    limit = max(limit, selection_size)
    # 3) enforce lower bound
    limit = max(limit, min_candidates)
    # 4) enforce upper bound (and never exceed the dataset)
    limit = min(limit, max_candidates, total)
    return limit


def featurize(
    records: Sequence[dict[str, str]],
    columns: Sequence[str],
    model: Encoder,
) -> np.ndarray:
    """
    Featurize a list of records using the model.

    :param records: A list of records.
    :param columns: Columns to featurize.
    :param model: An Encoder model.
    :return: The embeddings of the records.
    """
    # Extract the embeddings for each column across all records
    embeddings_per_col = []
    for col in columns:
        col_texts = [r[col] for r in records]
        col_emb = model.encode(col_texts)
        embeddings_per_col.append(np.asarray(col_emb))

    return np.concatenate(embeddings_per_col, axis=1)


def remove_exact_duplicates(
    records: Sequence[dict[str, str]],
    columns: Sequence[str],
    reference_records: list[list[dict[str, str]]] | None = None,
) -> tuple[list[dict[str, str]], list[tuple[dict[str, str], list[dict[str, str]]]]]:
    """
    Remove exact duplicates based on the unpacked string representation of each record.

    If reference_records is None, the function will only check for duplicates within the records list.

    :param records: A list of records to check for exact duplicates.
    :param columns: Columns to unpack.
    :param reference_records: A list of records to compare against. These are already unpacked
    :return: A list of deduplicated records and a list of duplicates.
    """
    deduplicated = []
    duplicates = []

    column_set = set(columns)
    # Build a seen set from reference_records if provided
    seen: defaultdict[frozendict[str, str], list[dict[str, str]]] = defaultdict(list)
    if reference_records is not None:
        for record_set in reference_records:
            key = to_frozendict(record_set[0], column_set)
            seen[key] = list(record_set)
    in_one_set = reference_records is None

    for record in records:
        frozen_record = to_frozendict(record, column_set)
        if duplicated_records := seen.get(frozen_record):
            duplicates.append((record, duplicated_records))
        else:
            deduplicated.append(record)
            # Only add current documents to seen if no reference set is used
            if in_one_set:
                seen[frozen_record].append(record)

    return deduplicated, duplicates


def prepare_records(
    records: Sequence[Record], columns: Sequence[str] | None
) -> tuple[list[dict[str, str]], Sequence[str], bool]:
    """
    Validate and prepare records for processing.

    :param records: A list of records (strings or dictionaries).
    :param columns: Columns to use if records are dictionaries.
    :return: Tuple of (dict_records, columns, was_string).
    :raises ValueError: If records are empty.
    :raises ValueError: If columns are not provided for dictionary records.
    """
    if len(records) == 0:
        raise ValueError("records must not be empty")

    if columns is None and isinstance(records[0], dict):
        raise ValueError("Columns must be specified when passing dictionaries.")

    if isinstance(records[0], str):
        columns = ["text"]
        dict_records: list[dict[str, str]] = [{"text": str(record)} for record in records]
        was_string = True
    else:
        dict_records = list(records)
        was_string = False

    return dict_records, columns, was_string


def prepare_dataset_records(  # noqa: C901
    dataset: Any,
    columns: Sequence[str],
) -> tuple[list[dict[str, str]], list[list[dict[str, str]]], bool]:
    """
    Extract, validate, and exact-deduplicate dataset rows using columnar access.

    Expects HuggingFace Dataset-style columnar access (dataset[column_name] returns a sequence).

    :param dataset: A dataset with column_names attribute and columnar access.
    :param columns: Columns to use for deduplication.
    :return: Tuple of (deduplicated_records, items, was_string) where:
        - deduplicated_records: representative record per exact-duplicate bucket
        - items: buckets of exact duplicates (each bucket is list[record])
        - was_string: True iff columns == ["text"] and ALL raw values were strings
    :raises TypeError: If dataset doesn't have required attributes.
    :raises ValueError: If columns are not found in dataset.
    :raises ValueError: If dataset is empty.
    :raises ValueError: If column lengths don't match dataset length.
    :raises ValueError: If any column contains None values.
    """
    if not hasattr(dataset, "column_names") or not hasattr(dataset, "__len__"):
        raise TypeError("dataset must have 'column_names' and '__len__' attributes")

    missing = set(columns) - set(dataset.column_names)
    if missing:
        raise ValueError(f"Columns {missing} not found in dataset")

    n = len(dataset)
    if n == 0:
        raise ValueError("dataset must not be empty")

    cols = {c: dataset[c] for c in columns}
    for c in columns:
        if len(cols[c]) != n:
            raise ValueError(f"Column '{c}' length ({len(cols[c])}) does not match dataset length ({n})")

    col_set = set(columns)

    def coerce_at(i: int, c: str) -> str:
        raw = cols[c][i]
        if raw is None:
            raise ValueError(f"Column '{c}' has None at index {i}")
        return raw if isinstance(raw, str) else str(raw)

    was_string = len(columns) == 1 and columns[0] == "text"

    key_to_indices: dict[frozendict[str, str], list[int]] = defaultdict(list)
    key_first_idx: dict[frozendict[str, str], int] = {}

    for i in range(n):
        # Track "was_string" using RAW values (not coerced)
        if was_string and not isinstance(cols["text"][i], str):
            was_string = False

        row = {c: coerce_at(i, c) for c in columns}
        key = to_frozendict(row, col_set)

        key_to_indices[key].append(i)
        key_first_idx.setdefault(key, i)

    ordered_keys = sorted(key_to_indices.keys(), key=lambda k: key_first_idx[k])

    items: list[list[dict[str, str]]] = []
    deduplicated_records: list[dict[str, str]] = []

    for key in ordered_keys:
        indices = key_to_indices[key]
        bucket = [{c: coerce_at(i, c) for c in columns} for i in indices]
        deduplicated_records.append(bucket[0])
        items.append(bucket)

    return deduplicated_records, items, was_string
