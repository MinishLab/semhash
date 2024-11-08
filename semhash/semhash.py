from __future__ import annotations

import numpy as np
from model2vec import StaticModel
from nearest import Nearest
from nearest.datatypes import Backend
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class SemHash:
    def __init__(self, model: SentenceTransformer | StaticModel) -> None:
        """Initialize SemHash."""
        self.model = model

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
        # Initialize Nearest with the embeddings and a basic backend
        nearest = Nearest.from_vectors_and_items(
            vectors=embeddings1, items=[str(i) for i in range(len(embeddings1))], backend_type=Backend.BASIC
        )

        if embeddings2 is None:
            # Handle deduplication within one list
            deduplicated_indices = set(range(len(embeddings1)))
            duplicate_to_original_mapping = {}

            results = nearest.query_threshold(embeddings1, threshold=1 - threshold)

            for idx_in_batch, similar_items in enumerate(tqdm(results, total=len(embeddings1))):
                i = idx_in_batch  # Since we're processing embeddings1
                if i not in deduplicated_indices:
                    continue  # Skip already marked duplicates

                # Convert similar items (strings) back to integer indices
                similar_indices = [int(sim_item) for sim_item in similar_items if int(sim_item) != i]

                for sim_idx in similar_indices:
                    if sim_idx in deduplicated_indices:
                        deduplicated_indices.remove(sim_idx)
                        duplicate_to_original_mapping[sim_idx] = i  # Map duplicate to original

            return np.array(list(deduplicated_indices)), duplicate_to_original_mapping
        else:
            # Handle deduplication across two lists
            deduplicated_indices_in_b = set()
            duplicate_to_original_mapping = {}

            results = nearest.query_threshold(embeddings2, threshold=1 - threshold)

            for idx_in_batch, similar_items in enumerate(tqdm(results, total=len(embeddings2))):
                i = idx_in_batch  # Index in embeddings2
                if not similar_items:
                    deduplicated_indices_in_b.add(i)
                else:
                    # Map to the first similar item in embeddings1
                    duplicate_to_original_mapping[i] = int(similar_items[0])

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
