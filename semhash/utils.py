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
        frozen_record = frozendict({k: v for k, v in record.items() if k in column_set})
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
    :raises ValueError: If columns are not provided for dictionary records.
    """
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
