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


class DatasetLike(Protocol):
    """
    Protocol for dataset-like objects compatible with SemHash.from_dataset().

    Any object that provides columnar access (dataset[column_name] -> sequence)
    satisfies this protocol. HuggingFace datasets.Dataset is the primary example,
    but custom dataset implementations are supported.
    """

    column_names: Sequence[str]

    def __len__(self) -> int:
        """Return the number of rows in the dataset."""
        ...  # pragma: no cover

    def __getitem__(self, key: str) -> Sequence[Any]:
        """Return all values for the given column name."""
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


def _validate_dataset(dataset: DatasetLike, columns: Sequence[str]) -> tuple[dict[str, Sequence[Any]], int]:
    """Validate dataset structure and extract columns. Returns (cols, n)."""
    try:
        column_names = dataset.column_names
    except AttributeError as e:
        raise TypeError("dataset must satisfy DatasetLike (column_names, __len__, __getitem__)") from e

    missing = set(columns) - set(column_names)
    if missing:
        raise ValueError(f"Columns {missing} not found in dataset")

    n = len(dataset)
    if n == 0:
        raise ValueError("dataset must not be empty")

    cols = {c: dataset[c] for c in columns}
    for c in columns:
        if len(cols[c]) != n:
            raise ValueError(f"Column '{c}' length ({len(cols[c])}) does not match dataset length ({n})")

    return cols, n


def prepare_dataset_records(
    dataset: DatasetLike,
    columns: Sequence[str],
) -> tuple[list[dict[str, str]], list[list[dict[str, str]]], bool]:
    """
    Extract, validate, and exact-deduplicate dataset rows using columnar access.

    Supports HuggingFace datasets.Dataset and any dataset-like object that provides:
    - column_names: Sequence[str]
    - __len__() -> int
    - __getitem__(column_name: str) -> Sequence[Any] (columnar access)

    :param dataset: A dataset-like object with columnar access.
    :param columns: Columns to use for deduplication.
    :return: Tuple of (deduplicated_records, items, was_string) where:
        - deduplicated_records: representative record per exact-duplicate bucket
        - items: buckets of exact duplicates (each bucket is list[record])
        - was_string: True iff columns == ["text"] and ALL raw values were strings
    """
    cols, n = _validate_dataset(dataset, columns)
    col_set = set(columns)
    was_string = len(columns) == 1 and columns[0] == "text"

    def coerce(raw: Any, *, col: str, idx: int) -> str:
        if raw is None:
            raise ValueError(f"Column '{col}' has None at index {idx}")
        return raw if isinstance(raw, str) else str(raw)

    # Single-pass grouping: key -> bucket of exact duplicates
    buckets: dict[frozendict[str, str], list[dict[str, str]]] = {}
    order: list[frozendict[str, str]] = []

    for i in range(n):
        # Track "was_string" using RAW values (not coerced)
        if was_string and not isinstance(cols["text"][i], str):
            was_string = False

        row = {c: coerce(cols[c][i], col=c, idx=i) for c in columns}
        key = to_frozendict(row, col_set)

        bucket = buckets.get(key)
        if bucket is None:
            buckets[key] = [row]
            order.append(key)
        else:
            bucket.append(row)

    # Preserve first-occurrence order via the order list
    items = [buckets[k] for k in order]
    deduplicated_records = [bucket[0] for bucket in items]

    return deduplicated_records, items, was_string
