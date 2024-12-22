from __future__ import annotations

from typing import Sequence, Union, cast

import numpy as np
from model2vec import StaticModel
from vicinity import Backend, Vicinity

from semhash.utils import Encoder

Record = Union[str, dict[str, str]]


class SemHash:
    def __init__(self, model: Encoder | None = None, columns: list[str] | None = None, ann: bool = False) -> None:
        """
        Initialize SemHash.

        :param model: A model to use for featurization. Defaults to minishlab/potion-base-8M.
        :param columns: Columns to featurize. Required if records are dictionaries.
        :param ann: Whether to use approximate nearest neighbors for deduplication. Default is False.
        """
        self.model = model if model else StaticModel.from_pretrained("minishlab/potion-base-8M")
        self.backend = Backend.USEARCH if ann else Backend.BASIC
        self.columns = columns
        self.vicinity: Vicinity | None = None

    def _featurize(self, records: Sequence[Record]) -> np.ndarray:
        """
        Featurize a list of records using the model.

        :param records: A list of records (either strings or dictionaries).
        :return: The embeddings of the records.
        :raises ValueError: If columns are not specified when passing dictionaries.
        """
        if isinstance(records[0], dict):
            if self.columns is None:
                raise ValueError("Columns must be specified when passing dictionaries.")

            records = cast(Sequence[dict[str, str]], records)
            # Extract the embeddings for each column across all records
            embeddings_per_column = []
            for column in self.columns:
                column_texts = [r[column] for r in records]
                column_embeddings = self.model.encode(column_texts)
                embeddings_per_column.append(np.asarray(column_embeddings))

            return np.concatenate(embeddings_per_column, axis=1)

        else:
            # Records is a list of strings
            records = cast(Sequence[str], records)
            embeddings = self.model.encode(records)
            return embeddings

    def _unpack_record(self, record: Record) -> str:
        r"""
        Unpack a record into a single string.

        If the record is a dictionary, it uses self.columns to determine the order of the text segments.
        Each text is cleaned by replacing '\t' with ' '. The texts are then joined by '\t'.

        If the record is a string, it just replaces '\t' with ' ' and returns it.
        """
        if isinstance(record, dict):
            if self.columns is None:
                raise ValueError("Columns must be specified when passing dictionaries.")

            column_texts = []
            for column in self.columns:
                text = record[column].replace("\t", " ")
                column_texts.append(text)
            return "\t".join(column_texts)
        else:
            # Record is a string
            return record.replace("\t", " ")

    def fit(self, records: Sequence[Record]) -> np.ndarray:
        """
        Embed the records and fit a vicinity index on the embeddings.

        :param records: The dataset to fit on. Can be a list of dictionaries or a list of strings.
        :return: The embeddings of the records.
        :raises ValueError: If columns are not specified when records are dictionaries.
        """
        if self.columns is None and isinstance(records[0], dict):
            raise ValueError("Columns must be specified when passing dictionaries.")

        # Compute embeddings for the records and unpack the records
        embeddings = self._featurize(records)
        items = [self._unpack_record(record) for record in records]
        # Fit the index
        self.vicinity = Vicinity.from_vectors_and_items(vectors=embeddings, items=items, backend_type=self.backend)
        return embeddings

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
        embeddings = self._featurize(records)

        # Query the fitted index
        results = self.vicinity.query_threshold(embeddings, threshold=1 - threshold)

        # Keep only those records for which no similar item was found
        deduplicated_records = []
        for record, similar_items in zip(records, results):
            if not similar_items:
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
        # Create embeddings and fit the index
        embeddings = self.fit(records)

        # Get similar items for each record
        results = self.vicinity.query_threshold(embeddings, threshold=1 - threshold)  # type: ignore

        deduplicated_records = []
        seen_items = set()
        for record, similar_items in zip(records, results):
            # similar_items includes 'record' itself
            # If we've seen any of these items before, this is a duplicate cluster.
            if any(item in seen_items for item in similar_items):
                continue
            else:
                # This is the first time we see this cluster of similar items
                deduplicated_records.append(record)
                # Mark all items in this cluster as seen
                seen_items.update(similar_items)

        return deduplicated_records
