from __future__ import annotations

from typing import Any

import numpy as np
from usearch.index import Index as UsearchIndex
from vicinity import Backend
from vicinity.backends import AbstractBackend, get_backend_class
from vicinity.backends.usearch import UsearchArgs, UsearchBackend
from vicinity.datatypes import SingleQueryResult
from vicinity.utils import Metric

DocScore = tuple[dict[str, str], float]
DocScores = list[DocScore]
DictItem = list[dict[str, str]]

# HNSW recall on binary signatures drops with dataset size at usearch's default expansion, so it is widened.
_BINARY_EXPANSION = 512


class Index:
    def __init__(
        self, vectors: np.ndarray, items: list[DictItem], backend: AbstractBackend, distance_scale: float = 1.0
    ) -> None:
        """
        An index that maps vectors to items.

        This index has an efficient backend for querying, but also explicitly stores the vectors in memory.

        :param vectors: The vectors of the items.
        :param items: The items in the index. This is a list of lists. Each sublist contains one or more dictionaries
            that represent records. These records are exact duplicates of each other.
        :param backend: The backend to use for querying.
        :param distance_scale: The distance at which similarity reaches zero: 1 for cosine, `n_bits / 2` for Hamming.
        """
        self.items = items
        self.backend = backend
        self.vectors = vectors
        self.distance_scale = distance_scale

    @classmethod
    def from_vectors_and_items(
        cls, vectors: np.ndarray, items: list[DictItem], backend_type: Backend | str, **kwargs: Any
    ) -> Index:
        """
        Load the index from vectors and items.

        Bit-packed uint8 vectors, such as MinHash signatures, always get a usearch Hamming index.

        :param vectors: The vectors of the items.
        :param items: The items in the index.
        :param backend_type: The type of backend to use.
        :param **kwargs: Additional arguments to pass to the backend.
        :return: The index.
        """
        if vectors.dtype != np.uint8:
            backend_class = get_backend_class(backend_type)
            arguments = backend_class.argument_class(**kwargs)
            return cls(vectors, items, backend_class.from_vectors(vectors, **arguments.dict()))

        # Vicinity takes the dimensionality from the array shape, but usearch counts binary dimensions in bits.
        num_bits = vectors.shape[1] * 8
        kwargs = {"expansion_add": _BINARY_EXPANSION, "expansion_search": _BINARY_EXPANSION, **kwargs}
        usearch_args = UsearchArgs(dim=num_bits, metric=Metric.HAMMING, **kwargs)
        usearch_index = UsearchIndex(ndim=num_bits, metric="hamming", dtype="b1x8", **kwargs)
        usearch_index.add(None, vectors)  # type: ignore[arg-type]  # None keys are allowed but not typed
        return cls(vectors, items, UsearchBackend(usearch_index, usearch_args), distance_scale=num_bits / 2)

    def query_threshold(self, vectors: np.ndarray, threshold: float) -> list[DocScores]:
        """
        Query the index with a threshold.

        :param vectors: The vectors to query.
        :param threshold: The similarity threshold.
        :return: The query results.
        """
        out: list[DocScores] = []
        distance_threshold = (1 - threshold) * self.distance_scale
        for result in self.backend.threshold(vectors, threshold=distance_threshold, max_k=100):
            intermediate = []
            for index, distance in zip(*result):
                # Every item in the index contains one or more records that are exact duplicates of each other.
                # Only the first is returned, since listing every copy grows with the size of the group.
                # The backend returns distances, so we need to convert.
                intermediate.append((self.items[index][0], 1 - distance / self.distance_scale))
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
            # Convert returned distances to similarities.
            similarities = 1 - y[offset:] / self.distance_scale
            results.append((x[offset:], similarities))
        return results
