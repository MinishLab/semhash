import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Generic, Hashable, TypeVar

from semhash.utils import to_frozendict

Record = TypeVar("Record", str, dict[str, Any])


@dataclass
class DuplicateRecord(Generic[Record]):
    """
    A single record with its duplicates.

    Attributes
    ----------
        record: The original record being deduplicated.
        exact: Whether the record was identified as an exact match.
        duplicates: List of tuples consisting of duplicate records and their associated scores.

    """

    record: Record
    exact: bool
    duplicates: list[tuple[Record, float]] = field(default_factory=list)

    def _rethreshold(self, threshold: float) -> None:
        """Rethreshold the duplicates."""
        self.duplicates = [(d, score) for d, score in self.duplicates if score >= threshold]


@dataclass
class DeduplicationResult(Generic[Record]):
    """
    Deduplication result.

    Attributes
    ----------
        selected: List of deduplicated records after removing duplicates.
        filtered: List of DuplicateRecord objects containing details about duplicates of an original record.
        threshold: The similarity threshold used for deduplication.
        deduplicated: Deprecated, use selected instead.
        duplicates: Deprecated, use filtered instead.

    """

    selected: list[Record] = field(default_factory=list)
    filtered: list[DuplicateRecord] = field(default_factory=list)
    threshold: float = field(default=0.9)
    deduplicated: list[Record] = field(default_factory=list)  # Deprecated
    duplicates: list[DuplicateRecord] = field(default_factory=list)  # Deprecated

    def __post_init__(self) -> None:
        """Initialize deprecated fields and warn about deprecation."""
        if self.deduplicated or self.duplicates:
            warnings.warn(
                "'deduplicated' and 'duplicates' fields are deprecated and will be removed in a future release. Use 'selected' and 'filtered' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        if not self.selected and self.deduplicated:
            self.selected = self.deduplicated
        if not self.filtered and self.duplicates:
            self.filtered = self.duplicates
        if not self.deduplicated:
            self.deduplicated = self.selected
        if not self.duplicates:
            self.duplicates = self.filtered

    @property
    def duplicate_ratio(self) -> float:
        """Return the percentage of records dropped."""
        if denom := len(self.selected) + len(self.filtered):
            return 1.0 - len(self.selected) / denom
        return 0.0

    @property
    def exact_duplicate_ratio(self) -> float:
        """Return the percentage of records dropped due to an exact match."""
        if denom := len(self.selected) + len(self.filtered):
            return len([dup for dup in self.filtered if dup.exact]) / denom
        return 0.0

    def get_least_similar_from_duplicates(self, n: int = 1) -> list[tuple[Record, Record, float]]:
        """
        Return the N least similar duplicate pairs.

        :param n: The number of least similar pairs to return.
        :return: A list of tuples consisting of (original_record, duplicate_record, score).
        """
        all_pairs = [(dup.record, d, score) for dup in self.filtered for d, score in dup.duplicates]
        sorted_pairs = sorted(all_pairs, key=lambda x: x[2])  # Sort by score
        return sorted_pairs[:n]

    def rethreshold(self, threshold: float) -> None:
        """Rethreshold the duplicates."""
        if self.threshold > threshold:
            raise ValueError("Threshold is smaller than the given value.")
        for dup in self.filtered:
            dup._rethreshold(threshold)
            if not dup.duplicates:
                self.filtered.remove(dup)
                self.selected.append(dup.record)
        self.threshold = threshold

    @property
    def selected_with_duplicates(self) -> list[tuple[Record, list[tuple[Record, float]]]]:
        """
        For every kept record, return the duplicates that were removed along with their similarity scores.

        :return: A list of tuples where each tuple contains a kept record
                and a list of its duplicates with their similarity scores.
        """

        def _to_hashable(record: Record) -> Hashable:
            """Convert a record to a hashable representation."""
            if isinstance(record, dict):
                # Convert non-hashable values to str so the frozendict is hashable
                converted = {k: (v if isinstance(v, str) else str(v)) for k, v in record.items()}
                # Return a frozendict for immutability and hashability
                return to_frozendict(converted, set(record.keys()))

            # Strings are already hashable
            return record

        # Build a mapping from original-record  to  [(duplicate, score), …]
        buckets: defaultdict[Hashable, list[tuple[Record, float]]] = defaultdict(list)
        for duplicate_record in self.filtered:
            for original_record, score in duplicate_record.duplicates:
                buckets[_to_hashable(original_record)].append((duplicate_record.record, float(score)))

        # Assemble the final list in the same order as self.selected
        return [(rec, buckets.get(_to_hashable(rec), [])) for rec in self.selected]


@dataclass
class FilterResult(Generic[Record]):
    """
    Result of filtering operations.

    Attributes
    ----------
        selected: List of records that passed the filter criteria.
        filtered: List of records that were filtered out.
        scores_selected: List of scores for the selected records.
        scores_filtered: List of scores for the filtered records.

    """

    selected: list[Record]
    filtered: list[Record]
    scores_selected: list[float] = field(default_factory=list)
    scores_filtered: list[float] = field(default_factory=list)

    @property
    def filter_ratio(self) -> float:
        """Return the percentage of records filtered out."""
        if denom := len(self.selected) + len(self.filtered):
            return len(self.filtered) / denom
        return 0.0

    @property
    def selected_ratio(self) -> float:
        """Return the percentage of records selected."""
        return 1 - self.filter_ratio
