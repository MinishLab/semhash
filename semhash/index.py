from __future__ import annotations

from typing import List

import numpy as np
from vicinity import Backend
from vicinity.backends import AbstractBackend, get_backend_class
from vicinity.datatypes import SingleQueryResult

DocScore = tuple[dict[str, str], float]
DocScores = list[DocScore]
DictItem = list[dict[str, str]]


class Index:
    def __init__(self, vectors: np.ndarray, items: list[DictItem], backend: AbstractBackend) -> None:
        """
        An index that maps vectors to items.

        This index has an efficient backend for querying, but also explicitly stores the vectors in memory.

        :param vectors: The vectors of the items.
        :param items: The items in the index. This is a list of lists. Each sublist contains one or more dictionaries
            that represent records. These records are exact duplicates of each other.
        :param backend: The backend to use for querying.
        """
        self.items = items
        self.backend = backend
        self.vectors = vectors

    @classmethod
    def from_vectors_and_items(cls, vectors: np.ndarray, items: list[DictItem], backend_type: Backend) -> Index:
        """
        Load the index from vectors and items.

        :param vectors: The vectors of the items.
        :param items: The items in the index.
        :param backend_type: The type of backend to use.
        :return: The index.
        """
        backend_class = get_backend_class(backend_type)
        backend = backend_class.from_vectors(vectors)

        return cls(vectors, items, backend)

    def query_threshold(self, vectors: np.ndarray, threshold: float) -> list[DocScores]:
        """
        Query the index with a threshold.

        :param vectors: The vectors to query.
        :param threshold: The similarity threshold.
        :return: The query results.
        """
        out: list[DocScores] = []
        for result in self.backend.threshold(vectors, threshold=1 - threshold, max_k=100):
            intermediate = []
            for index, distance in zip(*result):
                # Every item in the index contains one or more records.
                # These are all exact duplicates, so they get the same score.
                for record in self.items[index]:
                    # The score is the cosine similarity.
                    # The backend returns distances, so we need to convert.
                    intermediate.append((record, 1 - distance))
            out.append(intermediate)

        return out

    def query_top_k(self, vectors: np.ndarray, k: int) -> List[SingleQueryResult]:
        """
        Query the index with a top-k threshold.

        :param vectors: The vectors to query.
        :param k: Maximum number of top-k records to keep.
        :return: The query results.
        """
        results = []
        for x, y in self.backend.query(vectors=vectors, k=k + 1):  # include the query vector
            results.append((x[-k:], (y + 1e-10)[-k:]))  # add a small epsilon to avoid division by zero
        return results
