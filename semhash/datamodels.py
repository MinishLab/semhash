from dataclasses import dataclass, field
from typing import Generic, TypeVar

Record = TypeVar("Record", str, dict[str, str])


@dataclass
class DuplicateRecord(Generic[Record]):
    """Duplicate record."""

    record: Record
    duplicates: list[Record]
    exact: bool
    # This is not implemented yet.
    score: list[float] = field(default_factory=list)


@dataclass
class DeduplicationResult(Generic[Record]):
    """Deduplication result."""

    deduplicated: list[Record]
    duplicates: list[DuplicateRecord]

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
