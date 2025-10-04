from __future__ import annotations

from typing import Any

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
    def from_vectors_and_items(
        cls, vectors: np.ndarray, items: list[DictItem], backend_type: Backend | str, **kwargs: Any
    ) -> Index:
        """
        Load the index from vectors and items.

        :param vectors: The vectors of the items.
        :param items: The items in the index.
        :param backend_type: The type of backend to use.
        :param **kwargs: Additional arguments to pass to the backend.
        :return: The index.
        """
        backend_class = get_backend_class(backend_type)
        arguments = backend_class.argument_class(**kwargs)
        backend = backend_class.from_vectors(vectors, **arguments.dict())

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

    def query_top_k(self, vectors: np.ndarray, k: int, vectors_are_in_index: bool) -> list[SingleQueryResult]:
        """
        Query the index with a top-k.

        :param vectors: The vectors to query.
        :param k: Maximum number of top-k records to keep.
        :param vectors_are_in_index: Whether the vectors are in the index. If this is set to True, we retrieve k + 1
            records, and do not consider the first one, as it is the query vector itself.
        :return: The query results. Each result is a tuple where the first element is the list of neighbor records,
                 and the second element is a NumPy array of cosine similarity scores.
        """
        results = []
        offset = int(vectors_are_in_index)
        for x, y in self.backend.query(vectors=vectors, k=k + offset):
            # Convert returned distances to cosine similarities.
            similarities = 1 - y[offset:]
            results.append((x[offset:], similarities))
        return results
