from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Generic

from frozendict import frozendict

from semhash.utils import DuplicateList, Record, to_frozendict


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
    duplicates: DuplicateList = field(default_factory=list)

    def _rethreshold(self, threshold: float) -> None:
        """Rethreshold the duplicates."""
        self.duplicates = [(d, score) for d, score in self.duplicates if score >= threshold]


@dataclass
class SelectedWithDuplicates(Generic[Record]):
    """
    A record that has been selected along with its duplicates.

    Attributes
    ----------
        record: The original record being selected.
        duplicates: List of tuples consisting of duplicate records and their associated scores.

    """

    record: Record
    duplicates: DuplicateList = field(default_factory=list)


@dataclass
class DeduplicationResult(Generic[Record]):
    """
    Deduplication result.

    Attributes
    ----------
        selected: List of deduplicated records after removing duplicates.
        filtered: List of DuplicateRecord objects containing details about duplicates of an original record.
        threshold: The similarity threshold used for deduplication.
        columns: Columns used for deduplication.

    """

    selected: list[Record] = field(default_factory=list)
    filtered: list[DuplicateRecord] = field(default_factory=list)
    threshold: float = field(default=0.9)
    columns: Sequence[str] | None = field(default=None)

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
        # Invalidate cached property before modifying data
        self.__dict__.pop("selected_with_duplicates", None)
        # Rethreshold duplicates and move records without duplicates to selected
        for dup in list(self.filtered):
            dup._rethreshold(threshold)
            if not dup.duplicates:
                self.filtered.remove(dup)
                self.selected.append(dup.record)
        self.threshold = threshold

    @cached_property
    def selected_with_duplicates(self) -> list[SelectedWithDuplicates[Record]]:
        """
        For every kept record, return the duplicates that were removed along with their similarity scores.

        :return: A list of tuples where each tuple contains a kept record
                and a list of its duplicates with their similarity scores.
        """

        def _to_hashable(record: Record) -> frozendict[str, str] | str:
            """Convert a record to a hashable representation."""
            if isinstance(record, dict) and self.columns is not None:
                # Convert dict to frozendict for immutability and hashability
                return to_frozendict(record, set(self.columns))
            return str(record)

        # Build a mapping from original-record  to  [(duplicate, score), …]
        buckets: defaultdict[Hashable, DuplicateList] = defaultdict(list)
        for duplicate_record in self.filtered:
            for original_record, score in duplicate_record.duplicates:
                buckets[_to_hashable(original_record)].append((duplicate_record.record, float(score)))

        result: list[SelectedWithDuplicates[Record]] = []
        for selected in self.selected:
            # Get the list of duplicates for the selected record
            raw_list = buckets.get(_to_hashable(selected), [])
            # Ensure we don't have duplicates in the list
            # Use full-record canonical JSON for dicts so that unhashable values are handled correctly
            deduped = {
                (
                    json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                    if isinstance(rec, dict)
                    else rec
                ): (rec, score)
                for rec, score in raw_list
            }
            result.append(SelectedWithDuplicates(record=selected, duplicates=list(deduped.values())))

        return result


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
