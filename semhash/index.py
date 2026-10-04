from __future__ import annotations

from typing import Any

import numpy as np
from vicinity import Backend
from vicinity.backends import AbstractBackend, get_backend_class
from vicinity.datatypes import SingleQueryResult
from vicinity.utils import Metric

from semhash.utils import MAX_NEIGHBORS, NEIGHBORS_PER_QUERY, Neighbors

DictItem = list[dict[str, str]]

# HNSW recall on binary signatures drops with dataset size at usearch's default expansion, so it is widened.
_BINARY_EXPANSION = 512


class Index:
    def __init__(
        self, vectors: np.ndarray, items: list[DictItem], backend: AbstractBackend, binary: bool = False
    ) -> None:
        """
        An index that maps vectors to items.

        This index has an efficient backend for querying, but also explicitly stores the vectors in memory.

        :param vectors: The vectors of the items.
        :param items: The items in the index. This is a list of lists. Each sublist contains one or more dictionaries
            that represent records. These records are exact duplicates of each other.
        :param backend: The backend to use for querying.
        :param binary: Whether the vectors are bit-packed binary signatures, compared with Hamming distance.
        """
        self.items = items
        self.backend = backend
        self.vectors = vectors
        self.binary = binary
        # The distance at which similarity reaches zero. Unrelated binary signatures differ in half of their bits.
        self.distance_scale = vectors.shape[1] * 8 / 2 if binary else 1.0

    @classmethod
    def from_vectors_and_items(
        cls,
        vectors: np.ndarray,
        items: list[DictItem],
        backend_type: Backend | str,
        binary: bool = False,
        **kwargs: Any,
    ) -> Index:
        """
        Load the index from vectors and items.

        :param vectors: The vectors of the items.
        :param items: The items in the index.
        :param backend_type: The type of backend to use.
        :param binary: Whether the vectors are bit-packed binary signatures, such as MinHash signatures, which are
            compared with Hamming distance. This requires the usearch backend.
        :param **kwargs: Additional arguments to pass to the backend.
        :return: The index.
        :raises ValueError: If binary vectors are combined with another backend or a metric.
        """
        if binary:
            # Only usearch supports Hamming distance.
            if Backend(backend_type) != Backend.USEARCH or "metric" in kwargs:
                raise ValueError("Binary vectors always use the usearch backend with Hamming distance")
            kwargs = {"expansion_add": _BINARY_EXPANSION, "expansion_search": _BINARY_EXPANSION, **kwargs}
            kwargs["metric"] = Metric.HAMMING

        backend_class = get_backend_class(backend_type)
        arguments = backend_class.argument_class(**kwargs)
        backend = backend_class.from_vectors(vectors, **arguments.dict())

        return cls(vectors, items, backend, binary)

    def similarity_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Convert vectors to ones whose cosine similarity matches the similarity scores of this index.

        :param vectors: The vectors to convert.
        :return: The vectors, with binary signatures unpacked to +-1 vectors.
        """
        if self.binary:
            return np.unpackbits(vectors, axis=-1).astype(np.float32) * 2 - 1
        return vectors

    def query_threshold(self, vectors: np.ndarray, threshold: float) -> list[Neighbors]:
        """
        Query the index with a threshold.

        :param vectors: The vectors to query.
        :param threshold: The similarity threshold.
        :return: Arrays of group indices and similarity scores for each query.
        """
        distance_threshold = (1 - threshold) * self.distance_scale
        neighbors = [
            (indices, 1 - distances / self.distance_scale)
            for indices, distances in self.backend.threshold(
                vectors, threshold=distance_threshold, max_k=NEIGHBORS_PER_QUERY
            )
        ]
        # Query rows that hit the limit again with a larger one, so dense clusters rarely need a brute-force fallback.
        if capped := [i for i, (indices, _) in enumerate(neighbors) if len(indices) >= NEIGHBORS_PER_QUERY]:
            results = self.backend.threshold(vectors[capped], threshold=distance_threshold, max_k=MAX_NEIGHBORS)
            for i, (indices, distances) in zip(capped, results):
                neighbors[i] = (indices, 1 - distances / self.distance_scale)
        return neighbors

    def query_top_k(self, vectors: np.ndarray, k: int, vectors_are_in_index: bool) -> list[SingleQueryResult]:
        """
        Query the index with a top-k.

        :param vectors: The vectors to query.
        :param k: Maximum number of top-k records to keep.
        :param vectors_are_in_index: Whether the vectors are in the index. If this is set to True, we retrieve k + 1
            records, and do not consider the first one, as it is the query vector itself.
        :return: The query results. Each result is a tuple where the first element is the list of neighbor records,
                 and the second element is a NumPy array of similarity scores.
        """
        results = []
        offset = int(vectors_are_in_index)
        for x, y in self.backend.query(vectors=vectors, k=k + offset):
            # Convert returned distances to similarities.
            similarities = 1 - y[offset:] / self.distance_scale
            results.append((x[offset:], similarities))
        return results
