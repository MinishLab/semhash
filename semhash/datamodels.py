from dataclasses import dataclass, field
from typing import Generic, TypeVar

Record = TypeVar("Record", str, dict[str, str])


@dataclass
class DuplicateRecord(Generic[Record]):
    """Duplicate record."""

    record: Record
    exact: bool
    duplicates: list[Record] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    def least_similar(self, n: int = 1) -> list[tuple[Record, float]]:
        """Return the least similar duplicate."""
        ascending_by_score = sorted(zip(self.duplicates, self.scores), key=lambda x: x[1])

        return ascending_by_score[:n]

    def _rethreshold(self, threshold: float) -> None:
        """Rethreshold the duplicates."""
        for i, score in enumerate(self.scores):
            if score < threshold:
                self.duplicates.pop(i)
                self.scores.pop(i)


@dataclass
class DeduplicationResult(Generic[Record]):
    """Deduplication result."""

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

    def get_least_similar_from_duplicates(self, n: int = 1) -> list[tuple[Record, list[tuple[Record, float]]]]:
        """Return the least similar duplicates."""
        return [(dup.record, dup.least_similar(n)) for dup in self.duplicates]

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
