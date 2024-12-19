from __future__ import annotations

import numpy as np
from model2vec import StaticModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from vicinity import Backend, Vicinity


class SemHash:
    def __init__(self, columns: list[str] | None, model: SentenceTransformer | StaticModel) -> None:
        """Initialize SemHash."""
        self.columns = columns
        self.model = model

    def featurize(self, record: dict[str, str]) -> np.ndarray:
        v = []
        for column in self.columns:
            v.append(self.model.encode(record[column]))

        return np.concatenate(v)

    def fit(self, records: list[dict[str, str]] | list[str]) -> None:
        if self.columns is None and isinstance(records[0], dict):
            raise ValueError()

        self.X = np.array([self.featurize(record) for record in records])
        self.vicinity = Vicinity.from_vectors_and_items(vectors=X, items=records, backend_type=Backend.BASIC)

    def deduplicate_embeddings(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray | None = None,
        threshold: float = 0.9,
    ) -> tuple[np.ndarray, dict[int, int]]:
        """
        Deduplicate embeddings within one list or across two lists.

        :param embeddings1: Embeddings of the first list of texts.
        :param embeddings2: Optional embeddings of the second list of texts.
        :param threshold: Similarity threshold for deduplication.
        :return: Deduplicated indices and a mapping of duplicates to originals.
        """
        # Initialize BasicBackend with the embeddings
        items = list(range(len(embeddings1)))
        vicinity = Vicinity.from_vectors_and_items(vectors=embeddings1, items=items, backend_type=Backend.BASIC)  # type: ignore
        if embeddings2 is None:
            # Handle deduplication within one list
            deduplicated_indices = set(range(len(embeddings1)))
            duplicate_to_original_mapping = {}

            results = vicinity.query_threshold(embeddings1, threshold=1 - threshold)

            for i, similar_indices in enumerate(tqdm(results, total=len(embeddings1))):
                if i not in deduplicated_indices:
                    continue  # Skip already marked duplicates

                # Exclude the current index from similar items
                similar_indices = [sim_idx for sim_idx in similar_indices if sim_idx != i]

                for sim_idx in similar_indices:
                    if sim_idx in deduplicated_indices:
                        deduplicated_indices.remove(sim_idx)
                        duplicate_to_original_mapping[sim_idx] = i  # Map duplicate to original

            return np.array(list(deduplicated_indices)), duplicate_to_original_mapping
        else:
            # Handle deduplication across two lists
            deduplicated_indices_in_b = set()
            duplicate_to_original_mapping = {}

            results = vicinity.query_threshold(embeddings2, threshold=1 - threshold)

            for i, similar_indices in enumerate(tqdm(results, total=len(embeddings2))):
                if len(similar_indices) == 0:
                    deduplicated_indices_in_b.add(i)
                else:
                    # Map to the first similar item in embeddings1
                    duplicate_to_original_mapping[i] = similar_indices[0]

            return np.array(list(deduplicated_indices_in_b)), duplicate_to_original_mapping

    def deduplicate(
        self,
        texts1: list[str],
        texts2: list[str] | None = None,
        threshold: float = 0.9,
    ) -> tuple[np.ndarray, dict[int, int]]:
        """
        Perform deduplication on one or two lists of texts.

        :param texts1: List of strings for the first dataset.
        :param texts2: Optional list of strings for the second dataset.
        :param threshold: Similarity threshold for deduplication.
        :return: Deduplicated indices and a mapping of duplicates to originals.
        """
        embeddings1 = self.model.encode(texts1, show_progressbar=True)
        embeddings2 = self.model.encode(texts2, show_progressbar=True) if texts2 else None

        deduplicated_indices, duplicate_mapping = self.deduplicate_embeddings(
            embeddings1, embeddings2=embeddings2, threshold=threshold
        )
        return deduplicated_indices, duplicate_mapping
