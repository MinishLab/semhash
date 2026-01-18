import hashlib
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
    :raises TypeError: If encoding fails due to incompatible data types.
    """
    # Extract the embeddings for each column across all records
    embeddings_per_col = []
    for col in columns:
        try:
            col_texts = [r[col] for r in records]
        except KeyError as e:
            raise ValueError(f"Missing column '{col}' in one or more records") from e
        try:
            col_emb = model.encode(col_texts)
        except TypeError as e:
            sample_type = type(col_texts[0]).__name__ if col_texts else "unknown"
            raise TypeError(
                f"Failed to encode column '{col}' (data type: {sample_type}). "
                f"If encoding non-text data, provide a compatible encoder via the `model` parameter. "
                f"See the SemHash documentation for more info."
            ) from e
        embeddings_per_col.append(np.asarray(col_emb))

    return np.concatenate(embeddings_per_col, axis=1)
