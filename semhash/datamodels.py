from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Generic

import numpy as np
from frozendict import frozendict

from semhash.index import MAX_NEIGHBORS
from semhash.utils import DuplicateList, Record, to_frozendict


@dataclass
class DuplicateRecord(Generic[Record]):
    """
    A single record with its duplicates.

    Attributes
    ----------
        record: The original record being deduplicated.
        exact: Whether the record matches its canonical exactly on the deduplication columns.
        duplicates: The canonical record and its similarity score, stored as a one-element list.

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

    def __post_init__(self) -> None:
        """Initialize the cache used for rethresholding."""
        self._self_deduplication: tuple[list[list[Record]], list[list[tuple[int, float]]], np.ndarray] | None = None

    @classmethod
    def _from_groups(
        cls,
        groups: list[list[Record]],
        results: list[list[tuple[int, float]]],
        vectors: np.ndarray,
        threshold: float,
        columns: Sequence[str] | None,
    ) -> DeduplicationResult[Record]:
        """Assign each record to a directly matching kept canonical, preserving input group order."""
        result = cls(threshold=threshold, columns=columns)
        norms = np.linalg.norm(vectors, axis=1)
        # Normalized vectors of selected records, in selection order, for comparing against all of them at once.
        selected_vectors = np.empty(vectors.shape, dtype=np.float32)
        selected_indices: list[int] = []
        canonical_indices: dict[int, int] = {}
        # Canonicals proposed by earlier neighbors, so a match missed by one query is still found through the other.
        proposals: defaultdict[int, set[int]] = defaultdict(set)

        def closest(i: int, candidates: list[int], candidate_vectors: np.ndarray) -> tuple[int, float] | None:
            if not candidates:
                return None
            scores = candidate_vectors @ (vectors[i] / norms[i])
            best = int(np.argmax(scores))
            return (candidates[best], float(scores[best])) if scores[best] >= threshold else None

        for i, group in enumerate(groups):
            matches = [j for j, score in results[i] if score >= threshold]
            # Neighbors propose their canonical, whose similarity is checked directly to avoid transitive matches.
            candidates = list(
                proposals.pop(i, set()).union(canonical_indices[j] for j in matches if j in canonical_indices)
            )
            best_match = closest(i, candidates, vectors[candidates] / norms[candidates, None])
            if best_match is None and len(matches) >= MAX_NEIGHBORS:
                # The neighbors may be truncated, so compare against every selected record directly.
                best_match = closest(i, selected_indices, selected_vectors[: len(selected_indices)])
            if best_match is None:
                canonical_index, score = i, 1.0
                selected_vectors[len(selected_indices)] = vectors[i] / norms[i]
                selected_indices.append(i)
                result.selected.append(group[0])
                filtered_records = group[1:]
            else:
                canonical_index, score = best_match
                filtered_records = group
            canonical_indices[i] = canonical_index
            canonical_record = groups[canonical_index][0]
            for j in matches:
                if j > i:
                    proposals[j].add(canonical_index)
            result.filtered.extend(
                DuplicateRecord(record=record, exact=best_match is None, duplicates=[(canonical_record, score)])
                for record in filtered_records
            )
        result._self_deduplication = (groups, results, vectors)
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
        all_pairs = [(dup.record, d, score) for dup in self.filtered for d, score in dup.duplicates]
        sorted_pairs = sorted(all_pairs, key=lambda x: x[2])  # Sort by score
        return sorted_pairs[:n]

    def rethreshold(self, threshold: float) -> None:
        """Rethreshold the duplicates."""
        if self.threshold > threshold:
            raise ValueError("Threshold is smaller than the given value.")
        # Invalidate cached property before modifying data
        self.__dict__.pop("selected_with_duplicates", None)
        if (state := getattr(self, "_self_deduplication", None)) is not None:
            # Replay selection over cached group matches; filtered records must not keep each other filtered.
            groups, results, vectors = state
            result = self._from_groups(groups, results, vectors, threshold, self.columns)
            self.selected, self.filtered = result.selected, result.filtered
        else:
            filtered = []
            for dup in self.filtered:
                dup._rethreshold(threshold)
                if not dup.duplicates:
                    self.selected.append(dup.record)
                else:
                    filtered.append(dup)
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
            for original_record, score in duplicate_record.duplicates:
                buckets[_to_hashable(original_record)].append((duplicate_record.record, float(score)))

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
