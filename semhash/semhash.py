from __future__ import annotations

from typing import Sequence, Union

import numpy as np
from model2vec import StaticModel
from vicinity import Backend, Vicinity

Record = Union[str, dict[str, str]]


class SemHash:
    def __init__(self, model: StaticModel, columns: list[str] | None = None) -> None:
        """
        Initialize SemHash.

        :param model: A model to use for featurization.
        :param columns: Columns to featurize. Required if records are dictionaries.
        """
        self.model = model
        self.columns = columns
        self.vicinity: Vicinity | None = None

    def _featurize(self, record: Record) -> np.ndarray:
        """
        Featurize a record using the model.

        :param record: A record to featurize. Can be a dictionary or a string.
        :return: The featurized record.
        :raises ValueError: If columns are not specified when records are dictionaries.
        """
        if isinstance(record, dict):
            if self.columns is None:
                raise ValueError("Columns must be specified when passing dictionaries.")
            vectors = [np.asarray(self.model.encode(record[column])) for column in self.columns]
            return np.concatenate(vectors)
        else:
            return self.model.encode(record)

    def fit(self, records: Sequence[Record]) -> None:
        """
        Embed the records and fit a a vicinity index on the embeddings.

        :param records: The dataset to fit on. Can be a list of dictionaries or a list of strings.
        :raises ValueError: If columns are not specified when records are dictionaries.
        """
        if self.columns is None and isinstance(records[0], dict):
            raise ValueError("Columns must be specified when passing dictionaries.")

        embeddings = np.array([self._featurize(record) for record in records])
        self.vicinity = Vicinity.from_vectors_and_items(vectors=embeddings, items=records, backend_type=Backend.BASIC)  # type: ignore

    def deduplicate(
        self,
        records: Sequence[Record],
        threshold: float = 0.9,
    ) -> Sequence[Record]:
        """
        Perform deduplication against the fitted index.

        This method assumes you have already fit on a reference dataset (e.g., a train set).
        It will remove any items from 'records' that are similar above a certain threshold
        to any item in the fitted dataset.

        :param records: A new set of records (e.g., test set) to deduplicate against the fitted dataset.
        :param threshold: Similarity threshold for deduplication.
        :return: A deduplicated list of records.
        :raises ValueError: If no fitted index is found.
        """
        if self.vicinity is None:
            raise ValueError("No fitted index found. Call semhash.fit(records) before calling deduplicate.")

        # Compute embeddings for the new records
        embeddings = np.array([self._featurize(record) for record in records])

        # Query the fitted index
        results = self.vicinity.query_threshold(embeddings, threshold=1 - threshold)

        # Keep only those records for which no similar item was found
        deduplicated_records = []
        for record, similar_items in zip(records, results):
            if len(similar_items) == 0:
                # No duplicates found, keep this record
                deduplicated_records.append(record)

        return deduplicated_records

    def fit_deduplicate(
        self,
        records: Sequence[Record],
        threshold: float = 0.9,
    ) -> Sequence[Record]:
        """
        Fit and deduplicate a single dataset.

        This method removes any items that have duplicates within the same dataset.

        :param records: The dataset to fit and deduplicate.
        :param threshold: Similarity threshold for deduplication.
        :return: A deduplicated list of records.
        """
        self.fit(records)
        assert self.vicinity is not None

        embeddings = np.array([self._featurize(record) for record in records])
        results = self.vicinity.query_threshold(embeddings, threshold=1 - threshold)

        deduplicated_records = []
        seen_items = set()
        for record, similar_items in zip(records, results):
            # similar_items includes the record itself (since we are querying the same set)
            # If we haven't chosen any of these similar items yet, this is a new unique item.
            # If we have chosen one before, this is a duplicate.
            item_ids = [id(item) for item in similar_items]
            # Check if any similar item is already in seen_items
            if any(item in seen_items for item in item_ids):
                # Duplicate found, skip this record
                continue
            else:
                # This is the first time we see this cluster of similar items
                deduplicated_records.append(record)
                # Mark all similar items as seen to handle subsequent duplicates
                for item in item_ids:
                    seen_items.add(item)

        return deduplicated_records
