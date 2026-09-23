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

# HNSW recall on binary signatures degrades with dataset size at usearch's defaults (128/64), dropping to
# 0.61 at a million records. These wider values hold recall above 0.96 from 200k to 1M records.
_BINARY_EXPANSION = 512


class Index:
    def __init__(
        self,
        vectors: np.ndarray,
        items: list[DictItem],
        backend: AbstractBackend,
        distance_scale: float = 1.0,
    ) -> None:
        """
        An index that maps vectors to items.

        This index has an efficient backend for querying, but also explicitly stores the vectors in memory.

        :param vectors: The vectors of the items.
        :param items: The items in the index. This is a list of lists. Each sublist contains one or more dictionaries
            that represent records. These records are exact duplicates of each other.
        :param backend: The backend to use for querying.
        :param distance_scale: The distance at which similarity reaches zero, so that a distance `d` corresponds to
            a similarity of `1 - d / distance_scale`. This is 1 for cosine distance. For the bit counts returned by
            a Hamming backend over `n` bit signatures it is `n / 2`, which turns the count into estimated Jaccard.
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

    @classmethod
    def from_binary_vectors_and_items(cls, vectors: np.ndarray, items: list[DictItem], **kwargs: Any) -> Index:
        """
        Load the index from bit-packed vectors and items, using Hamming distance.

        Vicinity derives the number of dimensions from the shape of the array, but usearch counts dimensions in
        bits for its binary metrics, so its backend cannot build a Hamming index from packed bytes directly. This
        builds the usearch index with the right dimensionality and wraps it in the vicinity backend.

        :param vectors: The bit-packed vectors of the items, as a uint8 array of shape (n_items, n_bits // 8).
        :param items: The items in the index.
        :param **kwargs: Additional arguments to pass to the usearch index.
        :return: The index.
        :raises ValueError: If the vectors are not bit-packed into uint8.
        """
        if vectors.dtype != np.uint8:
            raise ValueError(f"Binary vectors must be bit-packed into uint8, got dtype {vectors.dtype}")

        num_bits = vectors.shape[1] * 8
        arguments = UsearchArgs(
            dim=num_bits,
            metric=Metric.HAMMING,
            **{"expansion_add": _BINARY_EXPANSION, "expansion_search": _BINARY_EXPANSION, **kwargs},
        )
        usearch_index = UsearchIndex(
            ndim=num_bits,
            metric="hamming",
            dtype="b1x8",
            connectivity=arguments.connectivity,
            expansion_add=arguments.expansion_add,
            expansion_search=arguments.expansion_search,
        )
        usearch_index.add(None, vectors)  # type: ignore[arg-type]  # None keys are allowed but not typed

        return cls(vectors, items, UsearchBackend(usearch_index, arguments), distance_scale=num_bits / 2)

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
