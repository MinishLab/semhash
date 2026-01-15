import hashlib
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Protocol, TypeAlias, TypeVar

import numpy as np
from frozendict import frozendict

# Type definitions
Record = TypeVar("Record", str, dict[str, Any])
DuplicateList: TypeAlias = list[tuple[Record, float]]


class Encoder(Protocol):
    """An encoder protocol for SemHash. Supports text, images, or any encodable data."""

    def encode(
        self,
        inputs: Sequence[Any] | Any,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Encode a list of inputs into embeddings.

        :param inputs: A list of inputs to encode (strings, images, etc.).
        :param **kwargs: Additional keyword arguments.
        :return: The embeddings of the inputs.
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


def make_hashable(value: Any) -> Any:
    """
    Convert a value to a hashable representation for use as dict keys.

    Strings and other hashable types are returned as-is.
    Non-hashable types (like PIL images, numpy arrays) are hashed to a string.

    :param value: The value to make hashable.
    :return: A hashable representation of the value.
    """
    # Fast path: most values are strings or already hashable
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    # Handle objects with tobytes() (PIL Image, numpy array, etc.)
    if hasattr(value, "tobytes"):
        return hashlib.md5(value.tobytes()).hexdigest()
    # Fallback: try to hash, otherwise stringify
    try:
        hash(value)
        return value
    except TypeError:
        return str(value)


def coerce_value(value: Any) -> Any:
    """
    Coerce a value for encoding: stringify primitives, keep complex types raw.

    This ensures primitives (int, float, bool) work with text encoders,
    while complex types (PIL images, tensors, etc.) are passed through for multimodal encoders.

    :param value: The value to coerce.
    :return: The coerced value.
    """
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return value  # Complex types (images, tensors, etc.)


def to_frozendict(record: dict[str, Any], columns: Sequence[str] | set[str]) -> frozendict[str, Any]:
    """
    Convert a record to a frozendict with hashable values.

    :param record: The record to convert.
    :param columns: The columns to include.
    :return: A frozendict with only the specified columns (values made hashable).
    :raises ValueError: If a column is missing from the record.
    """
    try:
        return frozendict({k: make_hashable(record[k]) for k in columns})
    except KeyError as e:
        missing = e.args[0]
        raise ValueError(f"Missing column '{missing}' in record {record}") from e


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
    buckets: dict[frozendict[str, Any], list[dict[str, Any]]] = {}
    order: list[frozendict[str, Any]] = []

    for r in records:
        key = to_frozendict(r, columns)
        bucket = buckets.get(key)
        if bucket is None:
            buckets[key] = [r]
            order.append(key)
        else:
            bucket.append(r)

    items = [buckets[k] for k in order]
    deduplicated_records = [bucket[0] for bucket in items]
    return deduplicated_records, items


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
    records: Sequence[dict[str, Any]],
    columns: Sequence[str],
    model: Encoder,
) -> np.ndarray:
    """
    Featurize a list of records using the model.

    :param records: A list of records.
    :param columns: Columns to featurize.
    :param model: An Encoder model.
    :return: The embeddings of the records.
    :raises ValueError: If a column is missing from one or more records.
    """
    # Extract the embeddings for each column across all records
    embeddings_per_col = []
    for col in columns:
        try:
            col_texts = [r[col] for r in records]
        except KeyError as e:
            raise ValueError(f"Missing column '{col}' in one or more records") from e
        col_emb = model.encode(col_texts)
        embeddings_per_col.append(np.asarray(col_emb))

    return np.concatenate(embeddings_per_col, axis=1)


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
    # Build a seen set from reference_records if provided
    seen: defaultdict[frozendict[str, Any], list[dict[str, Any]]] = defaultdict(list)
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

    if isinstance(records[0], str):
        # Validate all records are strings
        if not all(isinstance(r, str) for r in records):
            raise ValueError("All records must be strings when the first record is a string.")
        columns = ["text"]
        dict_records: list[dict[str, Any]] = [{"text": record} for record in records]
        was_string = True
    else:
        # Validate all records are dicts
        if not all(isinstance(r, dict) for r in records):
            raise ValueError("All records must be dicts when the first record is a dict.")
        assert columns is not None
        # Coerce values: stringify primitives, keep complex types raw (for images, etc.)
        dict_records_typed: list[dict[str, Any]] = list(records)  # type: ignore[arg-type]
        dict_records = []
        for r in dict_records_typed:
            coerced: dict[str, Any] = {}
            for c in columns:
                val = r.get(c)
                if val is None:
                    raise ValueError(f"Column '{c}' has None value in record {r}")
                coerced[c] = coerce_value(val)
            dict_records.append(coerced)
        was_string = False

    return dict_records, columns, was_string


def _validate_dataset(dataset: DatasetLike, columns: Sequence[str]) -> tuple[dict[str, Sequence[Any]], int]:
    """Validate dataset structure and extract columns.."""
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
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]], bool]:
    """
    Extract, validate, and exact-deduplicate dataset rows using columnar access.

    :param dataset: A dataset-like object with columnar access.
    :param columns: Columns to use for deduplication.
    :return: Tuple of (deduplicated_records, items, was_string) where:
        - deduplicated_records: representative record per exact-duplicate bucket
        - items: buckets of exact duplicates (each bucket is list[record])
        - was_string: True iff columns == ["text"] and ALL raw values were strings
    """
    cols, n = _validate_dataset(dataset, columns)

    # was_string controls whether deduplicate() returns strings or dicts.
    # We only return strings if: (1) single column named "text", AND (2) all raw
    # values in the dataset are actual strings (not integers/floats coerced to str).
    was_string = len(columns) == 1 and columns[0] == "text"

    def validate_and_coerce(raw: Any, *, col: str, idx: int) -> Any:
        """Validate value is not None, then coerce for encoding."""
        if raw is None:
            raise ValueError(f"Column '{col}' has None at index {idx}")
        return coerce_value(raw)

    # Build all records while tracking was_string
    records: list[dict[str, Any]] = []
    for i in range(n):
        if was_string and not isinstance(cols["text"][i], str):
            was_string = False
        records.append({c: validate_and_coerce(cols[c][i], col=c, idx=i) for c in columns})

    # Group by exact match, preserving first-occurrence order
    deduplicated_records, items = group_records_by_key(records, columns)

    return deduplicated_records, items, was_string
