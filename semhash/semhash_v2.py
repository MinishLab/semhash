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
        self.vicinity = Vicinity.from_vectors_and_items(vectors=self.X, items=records, backend_type=Backend.BASIC)

    def deduplicate_embeddings(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
        threshold: float = 0.9,
    ) -> tuple[np.ndarray, dict[int, int]]:
        """
        Deduplicate embeddings2 against embeddings1.

        :param embeddings1: Embeddings of the reference set (already fitted).
        :param embeddings2: Embeddings of the records we want to deduplicate.
        :param threshold: Similarity threshold for deduplication.
        :return: Deduplicated indices of embeddings2 and a mapping of duplicates (in embeddings2) to their originals (in embeddings1).
        """
        items = list(range(len(embeddings1)))
        vicinity = Vicinity.from_vectors_and_items(vectors=embeddings1, items=items, backend_type=Backend.BASIC)  # type: ignore

        deduplicated_indices_in_b = set()
        duplicate_to_original_mapping = {}

        # query_threshold returns a list of lists of items from embeddings1 that are similar to each embeddings2 vector
        results = vicinity.query_threshold(embeddings2, threshold=1 - threshold)

        for i, similar_indices in enumerate(tqdm(results, total=len(embeddings2))):
            if len(similar_indices) == 0:
                # No duplicates found in embeddings1, so keep this one
                deduplicated_indices_in_b.add(i)
            else:
                # Map this query to the first found match in embeddings1
                duplicate_to_original_mapping[i] = similar_indices[0]

        return np.array(list(deduplicated_indices_in_b)), duplicate_to_original_mapping

    def deduplicate(
        self,
        records: list[dict[str, str]],
        threshold: float = 0.9,
    ) -> tuple[np.ndarray, dict[int, int]]:
        """
        Perform deduplication against the dataset on which the model was fitted.

        This function uses the embeddings from `self.X` (fitted records) as `embeddings1`
        and deduplicates a new set of `records` (queries) against them as `embeddings2`.

        If you want to deduplicate the same dataset you fit on, just call:
        fit(records=records)
        deduplicate(records=records)

        :param records: A new set of records (queries) to deduplicate against the fitted dataset.
        :param threshold: Similarity threshold for deduplication.
        :return: Deduplicated indices and a mapping of duplicates (in records) to their originals (in the fitted dataset).
        """
        if self.X is None:
            raise ValueError("Model must be fitted before deduplication.")

        embeddings = np.array([self.featurize(record) for record in records])

        deduplicated_indices, duplicate_mapping = self.deduplicate_embeddings(
            embeddings1=self.X, embeddings2=embeddings, threshold=threshold
        )
        return deduplicated_indices, duplicate_mapping

    def fit_deduplicate(
        self,
        records: list[dict[str, str]],
        threshold: float = 0.9,
    ) -> tuple[np.ndarray, dict[int, int]]:
        """
        Fit on the dataset and deduplicate it in one go, avoiding double embedding.

        Use this method if you want to deduplicate the exact same set of records
        that you're fitting on, and you do not want to run `deduplicate` separately
        (which would re-embed your data).

        :param records: The dataset to fit and deduplicate.
        :param threshold: Similarity threshold for deduplication.
        :return: Deduplicated indices and a mapping of duplicates to originals.
        """
        if self.columns is None and isinstance(records[0], dict):
            raise ValueError("Columns must be specified when passing dictionaries.")

        # Embed the records once
        self.X = np.array([self.featurize(record) for record in records])
        self.vicinity = Vicinity.from_vectors_and_items(vectors=self.X, items=records, backend_type=Backend.BASIC)

        # Deduplicate the same dataset against itself
        deduplicated_indices, duplicate_mapping = self.deduplicate_embeddings(
            embeddings1=self.X, embeddings2=self.X, threshold=threshold
        )
        return deduplicated_indices, duplicate_mapping


# from model2vec import StaticModel
# from semhash.semhash_v2 import SemHash
# from datasets import load_dataset

# # Load an embedding model
# model = StaticModel.from_pretrained("minishlab/potion-base-8M")

# # Load two datasets
# texts_train = load_dataset("ag_news", split="train")["text"]
# texts_test = load_dataset("ag_news", split="test")["text"]

# records = [{"text": text} for text in texts_train]
# queries = [{"text": text} for text in texts_test]
# semhash = SemHash(columns=["text"], model=model)
# semhash.fit(records=records)

# # Train/test deduplication
# semhash.deduplicate(records=queries)

# # Train/train deduplication
# semhash.fit_deduplicate(records=records)
