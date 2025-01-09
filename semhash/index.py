from __future__ import annotations

import numpy as np
from vicinity import Backend
from vicinity.backends import AbstractBackend, get_backend_class

DocScore = tuple[dict[str, str], float]
DocScores = list[DocScore]
DictItem = list[dict[str, str]]


class Index:
    def __init__(self, vectors: np.ndarray, items: list[DictItem], backend: AbstractBackend) -> None:
        """Make the index."""
        self.items = items
        self.backend = backend
        self.vectors = vectors

    def items_as_sequence(self) -> list[dict[str, str]]:
        """Return all items as a single sequence."""
        return [item[0] for item in self.items]

    @classmethod
    def from_vectors_and_items(cls, vectors: np.ndarray, items: list[DictItem], backend_type: Backend) -> Index:
        """Load the index from vectors and items."""
        backend_class = get_backend_class(backend_type)
        backend = backend_class.from_vectors(vectors)

        return cls(vectors, items, backend)

    def query_threshold(self, vectors: np.ndarray, threshold: float) -> list[DocScores]:
        """Query the index with a threshold."""
        out: list[list[tuple[dict[str, str], float]]] = []
        for result in self.backend.threshold(vectors, threshold=1 - threshold, max_k=100):
            intermediate = []
            for index, distance in zip(*result):
                distances = []
                items = []
                for item in self.items[index]:
                    distances.append(1 - distance)
                    items.append(item)
                intermediate.extend(list(zip(items, distances)))
            out.append(intermediate)

        return out
