from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar

Record = TypeVar("Record", str, dict[str, str])


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
        deduplicated: List of deduplicated records after removing duplicates.
        duplicates: List of DuplicateRecord objects containing details about duplicates of an original record.
        threshold: The similarity threshold used for deduplication.

    """

    deduplicated: list[Record]
    duplicates: list[DuplicateRecord]
    threshold: float

    @property
    def duplicate_ratio(self) -> float:
        """Return the percentage of records dropped."""
        if denom := len(self.deduplicated) + len(self.duplicates):
            return 1.0 - len(self.deduplicated) / denom
        return 0.0

    @property
    def exact_duplicate_ratio(self) -> float:
        """Return the percentage of records dropped due to an exact match."""
        if denom := len(self.deduplicated) + len(self.duplicates):
            return len([dup for dup in self.duplicates if dup.exact]) / denom
        return 0.0

    def get_least_similar_from_duplicates(self, n: int = 1) -> list[tuple[Record, Record, float]]:
        """
        Return the N least similar duplicate pairs.

        :param n: The number of least similar pairs to return.
        :return: A list of tuples consisting of (original_record, duplicate_record, score).
        """
        all_pairs = [(dup.record, d, score) for dup in self.duplicates for d, score in dup.duplicates]
        sorted_pairs = sorted(all_pairs, key=lambda x: x[2])  # Sort by score
        return sorted_pairs[:n]

    def rethreshold(self, threshold: float) -> None:
        """Rethreshold the duplicates."""
        if self.threshold > threshold:
            raise ValueError("Threshold is smaller than the given value.")
        for dup in self.duplicates:
            dup._rethreshold(threshold)
            if not dup.duplicates:
                self.duplicates.remove(dup)
                self.deduplicated.append(dup.record)
        self.threshold = threshold


@dataclass
class FilterResult(Generic[Record]):
    """
    Result of filtering operations.

    Attributes
    ----------
        selected: List of records that passed the filter criteria.
        filtered: List of records that were filtered out.
        scores: Optional dictionary mapping records to their scores.

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

    def get_lowest_scoring(self, n: int = 1) -> list[tuple[Record, float]]:
        """
        Return the N lowest scoring records.

        :param n: The number of lowest scoring records to return.
        :return: A list of tuples consisting of (record, score).
        """
        lowest_filtered = [(record, score) for record, score in zip(self.filtered, self.scores_filtered)]
        lowest_selected = [(record, score) for record, score in zip(self.selected, self.scores_selected)]
        return lowest_filtered[:n] + lowest_selected[:n]

    def get_highest_scoring(self, n: int = 1) -> list[tuple[Record, float]]:
        """
        Return the N highest scoring records.

        :param n: The number of highest scoring records to return.
        :return: A list of tuples consisting of (record, score).
        """
        highest_selected = [(record, score) for record, score in zip(self.selected, self.scores_selected)]
        highest_filtered = [(record, score) for record, score in zip(self.filtered, self.scores_filtered)]
        return highest_selected[:n] + highest_filtered[:n]
