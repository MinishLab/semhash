from __future__ import annotations

import imp
from collections import defaultdict
from re import T
from typing import Generic, Sequence, Union

import numpy as np
from frozendict import frozendict
from model2vec import StaticModel
from vicinity import Backend

from semhash.datamodels import DeduplicationResult, DuplicateRecord, FilterResult, Record
from semhash.index import Index
from semhash.records import add_scores_to_records, map_deduplication_result_to_strings, to_frozendict
from semhash.utils import Encoder, entropy_from_distances


class SemHash(Generic[Record]):
    def __init__(self, index: Index, model: Encoder, columns: Sequence[str], was_string: bool) -> None:
        """
        Initialize SemHash.

        :param index: An index.
        :param model: A model to use for featurization.
        :param columns: Columns of the records.
        :param was_string: Whether the records were strings. Used for mapping back to strings.
        """
        self.index = index
        self.model = model
        self.columns = columns
        self._was_string = was_string

    @staticmethod
    def _featurize(
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

    @classmethod
    def _remove_exact_duplicates(
        cls,
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

    @classmethod
    def from_records(
        cls,
        records: Sequence[Record],
        columns: Sequence[str] | None = None,
        use_ann: bool = True,
        model: Encoder | None = None,
    ) -> SemHash:
        """
        Initialize a SemHash instance from records.

        This removes exact duplicates, featurizes the records, and fits a vicinity index.

        :param records: A list of records (strings or dictionaries).
        :param columns: Columns to featurize if records are dictionaries.
        :param use_ann: Whether to use approximate nearest neighbors (True) or basic search (False). Default is True.
        :param model: (Optional) An Encoder model. If None, the default model is used (minishlab/potion-base-8M).
        :return: A SemHash instance with a fitted vicinity index.
        :raises ValueError: If columns are not provided for dictionary records.
        """
        if columns is None and isinstance(records[0], dict):
            raise ValueError("Columns must be specified when passing dictionaries.")

        if isinstance(records[0], str):
            # If records are strings, convert to dictionaries with a single column
            columns = ["text"]
            dict_records: list[dict[str, str]] = [{"text": record} for record in records]
            was_string = True
        else:
            dict_records = list(records)
            was_string = False

        # If no model is provided, load the default model
        if model is None:
            model = StaticModel.from_pretrained("minishlab/potion-base-8M")

        # Remove exact duplicates
        deduplicated_records, duplicates = cls._remove_exact_duplicates(dict_records, columns)

        col_set = set(columns)
        duplicate_map = defaultdict(list)
        for x, _ in duplicates:
            frozen_record = to_frozendict(x, col_set)
            duplicate_map[frozen_record].append(x)

        items: list[list[dict[str, str]]] = []
        for record in deduplicated_records:
            i = [record]
            frozen_record = to_frozendict(record, set(columns))
            i.extend(duplicate_map[frozen_record])
            items.append(i)

        # Create embeddings and unpack records
        embeddings = cls._featurize(deduplicated_records, columns, model)

        # Build the Vicinity index
        backend = Backend.USEARCH if use_ann else Backend.BASIC
        index = Index.from_vectors_and_items(
            vectors=embeddings,
            items=items,
            backend_type=backend,
        )

        return cls(index=index, columns=columns, model=model, was_string=was_string)

    def deduplicate(
        self,
        records: Sequence[Record],
        threshold: float = 0.9,
    ) -> DeduplicationResult:
        """
        Perform deduplication against the fitted index.

        This method assumes you have already fit on a reference dataset (e.g., a train set) with from_records.
        It will remove any items from 'records' that are similar above a certain threshold
        to any item in the fitted dataset.

        :param records: A new set of records (e.g., test set) to deduplicate against the fitted dataset.
        :param threshold: Similarity threshold for deduplication.
        :return: A deduplicated list of records.
        """
        dict_records = self._validate_if_strings(records)

        # Remove exact duplicates before embedding
        dict_records, exact_duplicates = self._remove_exact_duplicates(
            records=dict_records, columns=self.columns, reference_records=self.index.items
        )
        duplicate_records = []
        for record, duplicates in exact_duplicates:
            duplicated_with_score = add_scores_to_records(duplicates)
            duplicate_record = DuplicateRecord(record=record, duplicates=duplicated_with_score, exact=True)
            duplicate_records.append(duplicate_record)

        # If no records are left after removing exact duplicates, return early
        if not dict_records:
            return DeduplicationResult(deduplicated=[], duplicates=duplicate_records, threshold=threshold)

        # Compute embeddings for the new records
        embeddings = self._featurize(records=dict_records, columns=self.columns, model=self.model)
        # Query the fitted index
        results = self.index.query_threshold(embeddings, threshold=threshold)

        deduplicated_records = []
        for record, similar_items in zip(dict_records, results):
            if not similar_items:
                # No duplicates found, keep this record
                deduplicated_records.append(record)
            else:
                duplicate_records.append(
                    DuplicateRecord(
                        record=record,
                        duplicates=[(item, score) for item, score in similar_items],
                        exact=False,
                    )
                )

        result = DeduplicationResult(
            deduplicated=deduplicated_records, duplicates=duplicate_records, threshold=threshold
        )

        if self._was_string:
            # Convert records back to strings if the records were originally strings
            return map_deduplication_result_to_strings(result, columns=self.columns)

        return result

    def self_deduplicate(
        self,
        threshold: float = 0.9,
    ) -> DeduplicationResult:
        """
        Deduplicate within the same dataset. This can be used to remove duplicates from a single dataset.

        :param threshold: Similarity threshold for deduplication.
        :return: A deduplicated list of records.
        """
        # Query the fitted index
        results = self.index.query_threshold(self.index.vectors, threshold=threshold)
        column_set = set(self.columns)

        duplicate_records = []

        deduplicated_records = []
        seen_items: set[frozendict[str, str]] = set()
        for item, similar_items in zip(self.index.items, results):
            # Items is a list of items which are exact duplicates of each other
            # So if the an item has more than one record, it is an exact duplicate
            # Crucially, we should count each instance separately.
            record, *duplicates = item
            # We need to compare all duplicates to all _items_.
            # The first item in a list of duplicate is not duplicated, because otherwise
            # we would remove the whole cluster. But it is a duplicate for the other items.

            # Iterate from index 1.
            for index, curr_record in enumerate(duplicates, 1):
                # The use of indexing is intentional here, we want to check if the object is the same
                # not if they have the same values. If we did != or is we would probably ignore lots
                # of items.
                items_to_keep = item[:index] + item[index + 1 :]
                items_with_score = add_scores_to_records(items_to_keep)
                duplicate_records.append(DuplicateRecord(record=curr_record, duplicates=items_with_score, exact=True))

            # If we don't see any similar_items, we know the record is not a duplicate.
            # in rare cases, the item itself might not be a duplicate of itself.
            if not similar_items:
                deduplicated_records.append(record)
                continue
            items, _ = zip(*similar_items)
            frozen_items = [to_frozendict(item, column_set) for item in items]
            # similar_items includes 'record' itself
            # If we've seen any of these items before, this is a duplicate cluster.
            if any(item in seen_items for item in frozen_items):
                duplicate_records.append(
                    DuplicateRecord(
                        record=record,
                        duplicates=[(item, score) for item, score in similar_items if item != record],
                        exact=False,
                    )
                )
                continue
            # This is the first time we see this cluster of similar items
            deduplicated_records.append(record)
            # Mark all items in this cluster as seen
            seen_items.update(frozen_items)

        result = DeduplicationResult(
            deduplicated=deduplicated_records, duplicates=duplicate_records, threshold=threshold
        )

        if self._was_string:
            # Convert records back to strings if the records were originally strings
            return map_deduplication_result_to_strings(result, columns=self.columns)

        return result

    def _validate_if_strings(self, records: Sequence[dict[str, str] | str]) -> Sequence[dict[str, str]]:
        """
        Validate if the records are strings.

        If the records are strings, they are converted to dictionaries with a single column.

        :param records: The records to validate.
        :return: The records as a list of dictionaries.
        :raises ValueError: If the records are strings but were not originally strings.
        :raises ValueError: If the records are not all strings or dictionaries.
        """
        if isinstance(records[0], str):
            if not self._was_string:
                raise ValueError("Records were not originally strings, but you passed strings.")
            dict_records = [{"text": record} for record in records if isinstance(record, str)]
        else:
            dict_records = [record for record in records if isinstance(record, dict)]
        if dict_records != records:
            raise ValueError("Records must be either strings or dictionaries.")
        return dict_records

    def _validate_filter_budget(
        self, budget: float | int | None, records: Sequence[dict[str, str] | str], k: int
    ) -> int:
        """
        Validate the filter budget.

        :param budget: The budget to validate, either as a percentage (0 to 1) or an absolute number.
        :param records: The records to validate against.
        :param k: The number of top-k records to keep.
        :return: The validated budget as an integer.
        :raises ValueError: If the budget is not within the valid range or less than k.
        """
        if budget is None:
            budget = 0.9

        if not (0 <= budget <= 1 or 0 <= budget <= len(records)):
            raise ValueError("Budget must be between 0 and 1 (as a percentage) or between 0 and the number of records.")

        budget = int(budget) if budget > 1 else int(len(records) * budget)

        if budget < k:
            raise ValueError("Budget must be greater than or equal to the number of top-k records to keep.")

        return budget

    def _filter_by_entropy(
        self,
        records: Sequence[dict[str, str]],
        vectors: np.ndarray,
        budget: float | int | None,
        descending: bool = True,
    ) -> FilterResult:
        """
        Filter records based on their entropy scores.

        :param records: The records to filter.
        :param vectors: The vectors corresponding to the records.
        :param budget: The maximum number of records to keep.
        :param descending: Whether to sort in descending order of entropy.
        :return: A FilterResult containing selected and filtered records.
        """
        budget = self._validate_filter_budget(budget=budget, records=records, k=100)

        # compute entropy scores
        scores = [
            (idx, record, entropy_from_distances(results[0][-1]), results[0][0], results[0][-1])
            for idx, (record, vectors) in enumerate(zip(records, vectors))
            for results in [self.index.query_top_k(vectors.reshape(1, -1), k=100)]
        ]

        # sort scores
        scores.sort(key=lambda x: x[2], reverse=descending)

        # select records
        selected: list[Union[dict[str, str], str]] = []
        selected_indices: set[int] = set()
        scores_selected: list[float] = []
        for idx, record, entropy, _, _ in scores:
            if len(selected) >= budget:
                break
            if idx not in selected_indices:
                selected.append(record)
                selected_indices.add(idx)
                scores_selected.append(entropy)

        # filter records
        filtered = [record for idx, record, _, _, _ in scores if idx not in selected_indices]
        scores_filtered: list[float] = [entropy for idx, record, entropy, _, _ in scores if idx not in selected_indices]

        return FilterResult(
            selected=selected,
            filtered=filtered,
            scores_selected=scores_selected,
            scores_filtered=scores_filtered,
        )

    def filter_by_entropy(
        self,
        records: Sequence[Record],
        budget: float | int | None = 0.9,
        descending: bool = True,
    ) -> FilterResult:
        """
        Filter records based on their entropy. Entropy is computed based on mean cosine similarity of the top-100 records.

        :param records: Records to filter.
        :param budget: Maximum number of records to keep.
            If a float is passed, it is interpreted as a percentage of the total number of records.
        :param descending: Whether to sort in descending order, from high entropy to low entropy.
            Higher entropy means more diverse records, lower entropy means more similar records.
        :return: FilterResult containing selected and filtered records.
        """
        dict_records = self._validate_if_strings(records)

        # featurize the new records
        embeddings = self._featurize(records=dict_records, columns=self.columns, model=self.model)

        # execute filtering
        return self._filter_by_entropy(
            records=dict_records,
            vectors=embeddings,
            budget=budget,
            descending=descending,
        )

    def self_filter_by_entropy(
        self,
        budget: float | int | None = 0.9,
        descending: bool = True,
    ) -> FilterResult:
        """
        Filter records based on their entropy. Entropy is computed based on mean cosine similarity of the top-100 records.

        This is similar to filter_by_entropy, but it filters within the same dataset.

        :param budget: Maximum number of records to keep.
            If a float is passed, it is interpreted as a percentage of the total number of records.
        :param descending: Whether to sort in descending order, from high entropy to low entropy.
            Higher entropy means more diverse records, lower entropy means more similar records.
        :return: FilterResult containing selected and filtered records.
        """
        dict_records = [record[0] for record in self.index.items]
        return self._filter_by_entropy(
            records=dict_records,
            vectors=self.index.vectors,
            budget=budget,
            descending=descending,
        )
