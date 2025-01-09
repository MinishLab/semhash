from __future__ import annotations

from typing import Generic, Sequence

import numpy as np
from frozendict import frozendict
from model2vec import StaticModel
from vicinity import Backend

from semhash.datamodels import DeduplicationResult, DuplicateRecord, Record
from semhash.index import Index
from semhash.records import map_deduplication_result_to_strings, to_frozendict, unpack_record
from semhash.utils import Encoder


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
        reference_records: list[dict[str, str]] | None = None,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
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
        seen = {to_frozendict(x, column_set) for x in reference_records} if reference_records else set()
        in_one_set = reference_records is None

        for record in records:
            frozen_record = frozendict({k: v for k, v in record.items() if k in column_set})
            if frozen_record not in seen:
                deduplicated.append(record)
                # Only add current documents to seen if no reference set is used
                if in_one_set:
                    seen.add(frozen_record)
            else:
                duplicates.append(record)

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
        deduplicated_records, _ = cls._remove_exact_duplicates(dict_records, columns)

        # Create embeddings and unpack records
        embeddings = cls._featurize(deduplicated_records, columns, model)

        # Build the Vicinity index
        backend = Backend.USEARCH if use_ann else Backend.BASIC
        index = Index.from_vectors_and_items(
            vectors=embeddings,
            items=[(x,) for x in deduplicated_records],
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
        :raises: ValueError if passed records are strings and the original records were not strings.
        """
        if isinstance(records[0], str):
            if not self._was_string:
                raise ValueError("Records were not originally strings, but you passed strings.")
            # If records are strings, convert to dictionaries with a single column
            dict_records = [{"text": record} for record in records]
        else:
            dict_records = records

        # Remove exact duplicates before embedding
        dict_records, exact_duplicates = self._remove_exact_duplicates(
            records=dict_records, columns=self.columns, reference_records=self.index.items_as_sequence()
        )
        duplicate_records = [DuplicateRecord(record=record, duplicates=[], exact=True) for record in exact_duplicates]

        # If no records are left after removing exact duplicates, return early
        if not dict_records:
            return DeduplicationResult(deduplicated=[], duplicates=duplicate_records, at_threshold=threshold)

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
                items, scores = zip(*similar_items)
                duplicate_records.append(
                    DuplicateRecord(record=record, duplicates=list(items), scores=list(scores), exact=False)
                )

        if self._was_string:
            # Convert records back to strings if the records were originally strings
            deduplicated_str = [unpack_record(r, self.columns) for r in deduplicated_records]
            duplicates_str = map_deduplication_result_to_strings(duplicate_records, self.columns)
            return DeduplicationResult(deduplicated=deduplicated_str, duplicates=duplicates_str, at_threshold=threshold)

        return DeduplicationResult(
            deduplicated=deduplicated_records, duplicates=duplicate_records, at_threshold=threshold
        )

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
        dict_records = self.index.items_as_sequence()

        column_set = set(self.columns)

        duplicate_records = []
        deduplicated_records = []
        seen_items: set[frozendict[str, str]] = set()
        for record, similar_items in zip(dict_records, results):
            # If we don't see any similar_items, we know the record is not a duplicate.
            # in rare cases, the item itself might not be a duplicate of itself.
            if not similar_items:
                deduplicated_records.append(record)
                continue
            items, _ = zip(*similar_items)
            frozen_items = [to_frozendict(item, column_set) for item in items]
            # similar_items includes 'record' itself
            # If we've seen any of these items before, this is a duplicate cluster.
            if similar_items and any(item in seen_items for item in frozen_items):
                frozen_record = to_frozendict(record, column_set)
                if frozen_record in seen_items:
                    continue
                duplicates: tuple[dict[str, str], ...]
                scores: tuple[float, ...]
                duplicates, scores = zip(*[(item, score) for item, score in similar_items if item != record])
                duplicate_records.append(
                    DuplicateRecord(record=record, duplicates=list(duplicates), scores=list(scores), exact=False)
                )
                continue
            # This is the first time we see this cluster of similar items
            deduplicated_records.append(record)
            # Mark all items in this cluster as seen
            seen_items.update(frozen_items)

        if self._was_string:
            # Convert records back to strings if the records were originally strings
            deduplicated_str = [unpack_record(r, self.columns) for r in deduplicated_records]
            duplicates_str = map_deduplication_result_to_strings(duplicate_records, self.columns)
            return DeduplicationResult(deduplicated=deduplicated_str, duplicates=duplicates_str, at_threshold=threshold)

        return DeduplicationResult(
            deduplicated=deduplicated_records, duplicates=duplicate_records, at_threshold=threshold
        )
