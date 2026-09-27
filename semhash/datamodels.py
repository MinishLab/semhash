from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Generic

import numpy as np
from frozendict import frozendict

from semhash.utils import DuplicateList, Neighbors, Record, select_canonicals, to_frozendict


@dataclass
class DuplicateRecord(Generic[Record]):
    """
    A record that was filtered as a duplicate of another record.

    Attributes
    ----------
        record: The original record being deduplicated.
        exact: Whether the record was identified as an exact match.
        duplicate_of: The record that this record is a duplicate of.
        score: The similarity score between record and duplicate_of.
        duplicates: Deprecated, use duplicate_of and score instead.

    """

    record: Record
    exact: bool
    duplicate_of: Record
    score: float

    @property
    def duplicates(self) -> DuplicateList:
        """Deprecated, use duplicate_of and score instead."""
        warnings.warn(
            "'duplicates' is deprecated and will be removed in a future release. Use 'duplicate_of' and 'score' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return [(self.duplicate_of, self.score)]


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

    # Inputs of self_deduplicate, so rethreshold can rerun the selection at a higher threshold.
    _selection_inputs: tuple[list[list[Record]], list[Neighbors], np.ndarray] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @classmethod
    def _from_groups(
        cls,
        groups: list[list[Record]],
        neighbors: list[Neighbors],
        vectors: np.ndarray,
        threshold: float,
        columns: Sequence[str] | None,
    ) -> DeduplicationResult[Record]:
        """Build a self-deduplication result where every filtered record points to a directly matching selected record."""
        result = cls(threshold=threshold, columns=columns)
        canonicals = select_canonicals(vectors=vectors, neighbors=neighbors, threshold=threshold)
        for i, (group, (canonical, score)) in enumerate(zip(groups, canonicals)):
            is_selected = canonical == i
            if is_selected:
                result.selected.append(group[0])
            # The rest of a selected group are exact copies of its first record.
            filtered_records = group[1:] if is_selected else group
            result.filtered.extend(
                DuplicateRecord(record=record, exact=is_selected, duplicate_of=groups[canonical][0], score=score)
                for record in filtered_records
            )
        result._selection_inputs = (groups, neighbors, vectors)
        return result

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
        all_pairs = [(dup.record, dup.duplicate_of, dup.score) for dup in self.filtered]
        sorted_pairs = sorted(all_pairs, key=lambda x: x[2])  # Sort by score
        return sorted_pairs[:n]

    def rethreshold(self, threshold: float) -> None:
        """Rethreshold the duplicates."""
        if self.threshold > threshold:
            raise ValueError("Threshold is smaller than the given value.")
        # Invalidate cached property before modifying data
        self.__dict__.pop("selected_with_duplicates", None)
        if (inputs := self._selection_inputs) is not None:
            # Rerun the selection, since a record that is no longer filtered can become the canonical of later records.
            groups, neighbors, vectors = inputs
            result = self._from_groups(
                groups=groups, neighbors=neighbors, vectors=vectors, threshold=threshold, columns=self.columns
            )
            self.selected, self.filtered = result.selected, result.filtered
        else:
            filtered = []
            for dup in self.filtered:
                if dup.score >= threshold:
                    filtered.append(dup)
                else:
                    self.selected.append(dup.record)
            self.filtered = filtered
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
            buckets[_to_hashable(duplicate_record.duplicate_of)].append(
                (duplicate_record.record, duplicate_record.score)
            )

        result: list[SelectedWithDuplicates[Record]] = []
        for selected in self.selected:
            # Preserve occurrences, even when multiple input records have identical values.
            result.append(SelectedWithDuplicates(record=selected, duplicates=buckets.get(_to_hashable(selected), [])))

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
